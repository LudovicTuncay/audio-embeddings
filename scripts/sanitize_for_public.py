#!/usr/bin/env python3
"""Apply deterministic sanitization rules before publishing to public."""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}


@dataclass(frozen=True)
class PatternRule:
    pattern: str
    flags: list[str] = field(default_factory=list)
    path_globs: list[str] = field(default_factory=list)

    def compile(self) -> re.Pattern[str]:
        bitmask = re.NOFLAG
        for flag in self.flags:
            key = flag.upper()
            if key not in FLAG_MAP:
                raise ValueError(f"Unsupported regex flag: {flag}")
            bitmask |= FLAG_MAP[key]
        return re.compile(self.pattern, bitmask)

    def applies_to(self, rel_path: str) -> bool:
        if not self.path_globs:
            return True
        return any(fnmatch.fnmatch(rel_path, glob) for glob in self.path_globs)


@dataclass(frozen=True)
class ReplacementRule(PatternRule):
    replacement: str = ""


@dataclass(frozen=True)
class SanitizeConfig:
    remove_paths: list[str]
    replacements: list[ReplacementRule]
    blocked_patterns: list[PatternRule]


@dataclass
class RunSummary:
    removed: int = 0
    replaced_files: int = 0
    replacement_count: int = 0
    pending_changes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=".public-sanitize.yml",
        help="Sanitizer config path (default: .public-sanitize.yml).",
    )
    parser.add_argument(
        "--target-root",
        default=".",
        help="Repository root to sanitize (default: current directory).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Do not write changes; fail if sanitization would change files.",
    )
    return parser.parse_args()


def load_config(path: Path) -> SanitizeConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    replacements: list[ReplacementRule] = []
    for item in raw.get("replacements", []):
        replacements.append(
            ReplacementRule(
                pattern=item["pattern"],
                replacement=item["replacement"],
                flags=item.get("flags", []),
                path_globs=item.get("path_globs", []),
            )
        )

    blocked_patterns: list[PatternRule] = []
    for item in raw.get("blocked_patterns", []):
        if isinstance(item, str):
            blocked_patterns.append(PatternRule(pattern=item))
        else:
            blocked_patterns.append(
                PatternRule(
                    pattern=item["pattern"],
                    flags=item.get("flags", []),
                    path_globs=item.get("path_globs", []),
                )
            )

    return SanitizeConfig(
        remove_paths=raw.get("remove_paths", []),
        replacements=replacements,
        blocked_patterns=blocked_patterns,
    )


def list_tracked_paths(repo_root: Path) -> list[str]:
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.output.decode("utf-8", errors="replace")
        raise RuntimeError(f"Unable to list tracked files:\n{message}") from exc
    return [entry.decode("utf-8") for entry in raw.split(b"\0") if entry]


def is_text_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return True


def matches_remove_rule(rel_path: str, rule: str) -> bool:
    normalized = rule.strip()
    if not normalized:
        return False

    if any(token in normalized for token in ["*", "?", "["]):
        return fnmatch.fnmatch(rel_path, normalized)

    if normalized.endswith("/"):
        prefix = normalized.rstrip("/")
        return rel_path == prefix or rel_path.startswith(f"{prefix}/")

    return rel_path == normalized


def resolve_remove_targets(
    tracked_rel_paths: Iterable[str], remove_rules: list[str]
) -> set[str]:
    targets: set[str] = set()
    for rel_path in tracked_rel_paths:
        for rule in remove_rules:
            if matches_remove_rule(rel_path, rule):
                targets.add(rel_path)
                break
    return targets


def run_sanitizer(
    repo_root: Path,
    config: SanitizeConfig,
    check_only: bool,
    excluded_rel_paths: set[str] | None = None,
) -> tuple[RunSummary, list[str]]:
    summary = RunSummary()
    excluded = excluded_rel_paths or set()
    tracked_rel_paths = list_tracked_paths(repo_root)
    remove_targets = resolve_remove_targets(tracked_rel_paths, config.remove_paths)
    remove_targets = {
        path
        for path in remove_targets
        if path not in excluded and (repo_root / path).exists()
    }
    summary.pending_changes += len(remove_targets)

    transformed_content: dict[str, str] = {}
    blocked_messages: list[str] = []

    if not check_only:
        for rel_path in sorted(remove_targets):
            target = repo_root / rel_path
            if target.is_dir():
                shutil.rmtree(target)
                summary.removed += 1
            elif target.exists():
                target.unlink()
                summary.removed += 1

    for rel_path in tracked_rel_paths:
        if rel_path in excluded:
            continue
        if rel_path in remove_targets:
            continue
        file_path = repo_root / rel_path
        if not is_text_file(file_path):
            continue

        original = file_path.read_text(encoding="utf-8")
        updated = original
        per_file_replacements = 0

        for rule in config.replacements:
            if not rule.applies_to(rel_path):
                continue
            regex = rule.compile()
            updated, replaced = regex.subn(rule.replacement, updated)
            per_file_replacements += replaced

        if per_file_replacements > 0:
            summary.pending_changes += 1
            summary.replaced_files += 1
            summary.replacement_count += per_file_replacements
            if not check_only:
                file_path.write_text(updated, encoding="utf-8")

        transformed_content[rel_path] = updated

    for rel_path in tracked_rel_paths:
        if rel_path in excluded:
            continue
        if rel_path in remove_targets:
            continue
        file_path = repo_root / rel_path
        if not is_text_file(file_path):
            continue

        content = transformed_content.get(
            rel_path, file_path.read_text(encoding="utf-8")
        )

        for rule in config.blocked_patterns:
            if not rule.applies_to(rel_path):
                continue
            regex = rule.compile()
            for match in regex.finditer(content):
                line_number = content.count("\n", 0, match.start()) + 1
                blocked_messages.append(
                    f"{rel_path}:{line_number}: blocked pattern `{rule.pattern}`"
                )

    return summary, blocked_messages


def main() -> int:
    args = parse_args()
    repo_root = Path(args.target_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 2

    try:
        excluded_rel_paths: set[str] = set()
        try:
            excluded_rel_paths.add(str(config_path.relative_to(repo_root).as_posix()))
        except ValueError:
            pass
        config = load_config(config_path)
        summary, blocked_messages = run_sanitizer(
            repo_root=repo_root,
            config=config,
            check_only=args.check_only,
            excluded_rel_paths=excluded_rel_paths,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Sanitizer failed: {exc}", file=sys.stderr)
        return 2

    mode = "CHECK" if args.check_only else "APPLY"
    print(
        (
            f"[{mode}] removed={summary.removed} "
            f"replaced_files={summary.replaced_files} "
            f"replacement_count={summary.replacement_count} "
            f"pending_changes={summary.pending_changes}"
        )
    )

    if blocked_messages:
        print("Blocked patterns detected:")
        for message in blocked_messages:
            print(f"- {message}")
        return 1

    if args.check_only and summary.pending_changes > 0:
        print("Sanitization drift detected. Run without --check-only to apply rules.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

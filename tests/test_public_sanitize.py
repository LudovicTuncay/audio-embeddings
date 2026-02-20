from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SANITIZER = REPO_ROOT / "scripts" / "sanitize_for_public.py"


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def _init_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    return repo


def test_sanitizer_applies_replacements_and_removals(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    (repo / "README.md").write_text(
        "data=/lustre/fsmisc/dataset\nups=/mnt/ups/audio\n", encoding="utf-8"
    )
    (repo / "slurm_scripts").mkdir()
    (repo / "slurm_scripts" / "submit.py").write_text(
        "print('private')\n", encoding="utf-8"
    )
    (repo / ".public-sanitize.yml").write_text(
        """
remove_paths:
  - slurm_scripts/
replacements:
  - pattern: /lustre/fsmisc/dataset
    replacement: ${oc.env:DATA_ROOT,/path/to/datasets}
  - pattern: /mnt/ups
    replacement: ${oc.env:UPS_DATA_ROOT,/path/to/ups}
blocked_patterns:
  - pattern: /lustre/
  - pattern: /mnt/
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)

    result = _run(
        [
            sys.executable,
            str(SANITIZER),
            "--target-root",
            str(repo),
            "--config",
            ".public-sanitize.yml",
        ],
        cwd=repo,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert not (repo / "slurm_scripts" / "submit.py").exists()
    updated_readme = (repo / "README.md").read_text(encoding="utf-8")
    assert "/lustre/" not in updated_readme
    assert "/mnt/" not in updated_readme


def test_sanitizer_check_only_detects_drift(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    (repo / "README.md").write_text("data=/lustre/fsmisc/dataset\n", encoding="utf-8")
    (repo / ".public-sanitize.yml").write_text(
        """
replacements:
  - pattern: /lustre/fsmisc/dataset
    replacement: ${oc.env:DATA_ROOT,/path/to/datasets}
blocked_patterns: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)

    result = _run(
        [
            sys.executable,
            str(SANITIZER),
            "--target-root",
            str(repo),
            "--config",
            ".public-sanitize.yml",
            "--check-only",
        ],
        cwd=repo,
    )

    assert result.returncode == 1
    assert "Sanitization drift detected" in result.stdout


def test_sanitizer_fails_on_blocked_pattern(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    (repo / "secrets.txt").write_text("password='abc123'\n", encoding="utf-8")
    (repo / ".public-sanitize.yml").write_text(
        """
replacements: []
blocked_patterns:
  - pattern: (?i)(api[_-]?key|secret|password)\\s*[:=]\\s*['\\"][^'\\"]+['\\"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)

    result = _run(
        [
            sys.executable,
            str(SANITIZER),
            "--target-root",
            str(repo),
            "--config",
            ".public-sanitize.yml",
            "--check-only",
        ],
        cwd=repo,
    )

    assert result.returncode == 1
    assert "Blocked patterns detected" in result.stdout

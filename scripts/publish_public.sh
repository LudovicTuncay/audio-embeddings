#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/publish_public.sh <version> [--dry-run]

Example:
  scripts/publish_public.sh v0.2.0
  scripts/publish_public.sh v0.2.0 --dry-run

Requirements:
  - Run from the repository root.
  - Current branch must be release/<version>.
  - Working tree must be clean.
  - Remote "public" must exist.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

VERSION="$1"
DRY_RUN=0

if [[ $# -eq 2 ]]; then
  [[ "$2" == "--dry-run" ]] || die "Unknown flag: $2"
  DRY_RUN=1
fi

[[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "Version must match vX.Y.Z"

CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "release/$VERSION" ]] || die "Expected branch release/$VERSION, got $CURRENT_BRANCH"

git diff --quiet || die "Working tree has unstaged changes"
git diff --cached --quiet || die "Working tree has staged changes"
git remote get-url public >/dev/null 2>&1 || die 'Remote "public" does not exist'
command -v uv >/dev/null 2>&1 || die "uv is required"
command -v rsync >/dev/null 2>&1 || die "rsync is required"

if git rev-parse "$VERSION" >/dev/null 2>&1; then
  die "Tag $VERSION already exists locally"
fi

if git ls-remote --exit-code --tags public "refs/tags/$VERSION" >/dev/null 2>&1; then
  die "Tag $VERSION already exists on remote public"
fi

TMP_ROOT="$(mktemp -d)"
SANITIZED_WT="$TMP_ROOT/sanitized"
PUBLIC_WT="$TMP_ROOT/public"

cleanup() {
  git worktree remove --force "$SANITIZED_WT" >/dev/null 2>&1 || true
  git worktree remove --force "$PUBLIC_WT" >/dev/null 2>&1 || true
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

echo "[1/5] Preparing sanitized release snapshot..."
git worktree add --detach "$SANITIZED_WT" HEAD >/dev/null

(
  cd "$SANITIZED_WT"
  uv run scripts/sanitize_for_public.py --config .public-sanitize.yml
  # First pass may apply autofixes; second pass enforces a clean result.
  uv run pre-commit run --all-files || true
  uv run pre-commit run --all-files
  uv run scripts/sanitize_for_public.py --config .public-sanitize.yml --check-only
)

echo "[2/5] Preparing public release worktree..."
if git ls-remote --exit-code --heads public master >/dev/null 2>&1; then
  git fetch public master >/dev/null
  git worktree add --detach "$PUBLIC_WT" FETCH_HEAD >/dev/null
else
  git worktree add --detach "$PUBLIC_WT" HEAD >/dev/null
  (
    cd "$PUBLIC_WT"
    git checkout --orphan public-bootstrap >/dev/null
    git rm -rf . >/dev/null 2>&1 || true
  )
fi

echo "[3/5] Syncing sanitized tree into public snapshot..."
rsync -a --delete --exclude '.git' "$SANITIZED_WT"/ "$PUBLIC_WT"/

(
  cd "$PUBLIC_WT"
  git add -A
  git diff --cached --quiet && die "No public changes to publish for $VERSION"
  git commit -m "release(public): $VERSION" >/dev/null
)

echo "[4/5] Public commit ready."
(
  cd "$PUBLIC_WT"
  git --no-pager show --stat --oneline --no-patch HEAD
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[5/5] Dry run requested. Skipping push/tag."
  exit 0
fi

echo "[5/5] Pushing commit and tag to public..."
(
  cd "$PUBLIC_WT"
  git push public HEAD:master
  git tag -a "$VERSION" -m "Public release $VERSION"
  git push public "refs/tags/$VERSION"
)

echo "Published $VERSION to remote 'public'."

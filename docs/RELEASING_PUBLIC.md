# Releasing to the Public Mirror

This repository uses a private-first workflow:

- `origin` is the private canonical repository.
- `public` is the public mirror repository.
- Public publishes are release-gated from `release/*` branches only.

## 1) One-time migration setup

Run these from a clean working tree.

```bash
# Optional rollback/audit marker before rewiring remotes.
git tag migration-public-baseline-$(date +%Y%m%d)

# Keep the current public repo as the "public" remote.
git remote rename origin public

# Add the private canonical repository as "origin".
git remote add origin git@github.com:LudovicTuncay/audio-embeddings-private.git

# Verify topology.
git remote -v
```

Expected:

- `origin` points to `audio-embeddings-private`.
- `public` points to `audio-embeddings`.

## 2) Security gate before first release

Run a full-history secret scan before the first public publish from private.

```bash
# Example with gitleaks (install separately if needed).
gitleaks detect --source . --verbose
```

If a real credential leak is found:

1. Rotate the credential immediately.
2. Rewrite git history to purge it.
3. Re-scan until clean.
4. Only then continue with public publishing.

## 3) Cut a release branch in private

```bash
git checkout master
git pull origin master
git checkout -b release/v0.2.0
```

`scripts/publish_public.sh` requires the current branch name to match
`release/<version>` exactly.

## 4) Dry-run publication

```bash
scripts/publish_public.sh v0.2.0 --dry-run
```

What this does:

1. Creates a temporary worktree from the release branch.
2. Applies `.public-sanitize.yml` via `scripts/sanitize_for_public.py`.
3. Runs `uv run pre-commit run --all-files`.
4. Enforces policy checks (blocked patterns and private absolute paths).
5. Builds a public commit snapshot on top of `public/master`.

No push or tag occurs in dry-run mode.

## 5) Publish for real

```bash
scripts/publish_public.sh v0.2.0
```

This pushes:

- The sanitized commit to `public/master`.
- Annotated tag `v0.2.0` to `public`.

## 6) OSS intake loop (public-first triage)

Public repository labels:

- `accepted`
- `needs-private-port`
- `scheduled-release`

Workflow:

1. Triage public issue/PR in the public repo.
2. Merge accepted PR in public for contributor visibility.
3. Port accepted change into private `master` (for example with `cherry-pick`).
4. Include that port in the next `release/<version>` publish.

## 7) Recommended branch protections

Private repository (`origin`, branch `master`):

- Require PRs.
- Require status checks once CI jobs are wired in.
- Restrict force-push/deletions.

Public repository (`public`, branch `master`):

- Maintainers-only push.
- Release script is the normal write path.

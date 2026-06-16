#!/bin/sh
# Commit the CI-generated SCORES.md + badges/ back to the repo, guarded so it never loops or hard-fails.
#  - skips cleanly if nothing changed,
#  - skips (with a notice) if BADGE_PUSH_TOKEN isn't set — the files are still kept as build artifacts,
#  - tags the commit [skip ci] so the scores commit does NOT re-trigger the pipeline,
#  - treats a rejected push (branch advanced mid-build) as a soft no-op (refreshed next run).
# BADGE_PUSH_TOKEN must be a secured repo variable holding a Repository Access Token with repository:write.
set -eu

git add SCORES.md badges 2>/dev/null || true

if git diff --cached --quiet 2>/dev/null; then
  echo "[publish] scores unchanged - nothing to commit"
  exit 0
fi

if [ -z "${BADGE_PUSH_TOKEN:-}" ]; then
  echo "[publish] BADGE_PUSH_TOKEN not set - SCORES.md/badges kept as artifacts, NOT committed back."
  echo "[publish] Set it: repo Access token (repository:write) -> secured repo variable BADGE_PUSH_TOKEN."
  exit 0
fi

git config user.name "maxwell-ci"
git config user.email "ci@novateur.com"
git commit -m "CI: update gallery scores [skip ci]"

REPO="bitbucket.org/Novateur/vlincs_reid_by_search.git"
if git push "https://x-token-auth:${BADGE_PUSH_TOKEN}@${REPO}" "HEAD:${BITBUCKET_BRANCH}"; then
  echo "[publish] pushed updated scores to ${BITBUCKET_BRANCH}"
else
  echo "[publish] push rejected (branch advanced?) - scores will refresh on the next run"
fi

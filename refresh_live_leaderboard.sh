#!/bin/zsh
set -euo pipefail

REPO_DIR="${0:A:h}"
cd "$REPO_DIR"

# Endpoint agents are captured before the target session and replayed from an
# immutable local log. The capture command refuses to call after the open.
if ! python3 capture_endpoint_orders.py; then
  # A remote endpoint that was not captured before the open is kept at its
  # previous comparable score by live_runner.py. It must not freeze every
  # local entrant or be treated as a no-op entrant decision.
  print -u2 "endpoint capture unavailable; continuing with stale endpoint rows preserved"
fi

# The private entrants live only on this scoring machine. Never publish or move
# their code; only the generated JSON artifacts are committed.
if ! python3 market_open.py; then
  exit 0
fi

# Never mix an automated board refresh with unrelated local edits.
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  print -u2 "tracked worktree changes present; leaderboard refresh skipped"
  exit 1
fi

git pull --ff-only origin main
python3 live_runner.py
git add leaderboard.json private_results.json history.json

if git diff --staged --quiet; then
  print "leaderboard unchanged"
  exit 0
fi

git commit -m "Refresh live leaderboard $(date -u +%FT%TZ)"
git push origin main

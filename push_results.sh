#!/usr/bin/env bash
# Commit and push the alignment results (CSVs + figures) to GitHub.
# Run after 03/04 finish:  bash push_results.sh
cd "$(dirname "$0")" || exit 1
git add results/*.csv results/*.png 2>/dev/null
if git diff --cached --quiet; then
    echo "No new results to push."
    exit 0
fi
git commit -m "results: alignment run $(date +%Y-%m-%d_%H%M)"
git push
echo "Pushed results to GitHub."

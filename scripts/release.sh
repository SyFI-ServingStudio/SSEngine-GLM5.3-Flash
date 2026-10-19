#!/bin/bash
# Release the dev branch to main: main = dev without research/ (README "Branches").
#
#   scripts/release.sh [<branch, default dev>]
#
# Run on a clean checkout of main. Merges the branch with --no-ff --no-commit, removes research/ from the merge
# (conflicts are resolved by deletion only under research/: a research file dev changed and main deleted), checks that
# nothing under research/ is left, runs the CPU tests and commits "Release from dev <sha>". On any failure the merge is
# aborted and main is left as it was. Nothing is pushed.
#   PYTHON   the interpreter of the tests   default .venv/bin/python
#   TRAILER  a trailer paragraph for the commit message (e.g. Co-Authored-By: ...)   default none
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SRC=${1:-dev}
PY=${PYTHON:-.venv/bin/python}
die() { echo "release: $*" >&2; exit 1; }

[ "$(git symbolic-ref --short HEAD 2>/dev/null)" = main ] || die "check out main first"
git diff --quiet && git diff --cached --quiet || die "main has uncommitted changes"
[ -z "$(git ls-files research)" ] || die "main tracks files under research/"
SHA=$(git rev-parse --short "$SRC^{commit}")
git merge-base --is-ancestor "$SRC" HEAD && die "$SRC ($SHA) is already in main"

abort() { echo "release: aborting the merge" >&2; git merge --abort 2>/dev/null || true; exit 1; }
git merge --no-ff --no-commit "$SRC" > /dev/null 2>&1 || true
git rev-parse -q --verify MERGE_HEAD > /dev/null || die "merging $SRC did not start (see git status)"

# conflicts: allowed only under research/, where deletion (main's side) wins
others=$(git diff --name-only --diff-filter=U | grep -v '^research/' || true)
[ -z "$others" ] || { echo "release: conflicts outside research/:" >&2; echo "$others" >&2; abort; }
git rm -r -q -f --ignore-unmatch research
[ -z "$(git ls-files research)" ] || { echo "release: research/ is still in the index" >&2; abort; }
[ -z "$(git diff --name-only --diff-filter=U)" ] || { echo "release: unresolved conflicts" >&2; abort; }

echo "release: CPU tests"
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q tests || { echo "release: CPU tests failed" >&2; abort; }

git commit -q -m "Release from dev $SHA" ${TRAILER:+-m "$TRAILER"}
echo "release: main = $(git rev-parse --short HEAD) (Release from dev $SHA); not pushed"

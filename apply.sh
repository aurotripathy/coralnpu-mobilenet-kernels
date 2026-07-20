#!/usr/bin/env bash
# Applies the CoralNPU MobileNet kernel + npusim-example changes onto a clean
# checkout of the upstream coralnpu repo.
#
# Usage:
#   ./apply.sh /path/to/coralnpu
#
# If no path is given, it defaults to ../coralnpu next to this repo.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORALNPU="${1:-$(cd "$HERE/.." && pwd)/coralnpu}"
BASE_COMMIT="$(cat "$HERE/BASE_COMMIT")"

if ! git -C "$CORALNPU" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: '$CORALNPU' is not a git checkout of coralnpu." >&2
  echo "Clone it first: git clone https://github.com/google-coral/coralnpu.git" >&2
  exit 1
fi

echo ">> Target coralnpu: $CORALNPU"
echo ">> Expected base commit: $BASE_COMMIT"

HEAD_COMMIT="$(git -C "$CORALNPU" rev-parse HEAD)"
if [ "$HEAD_COMMIT" != "$BASE_COMMIT" ]; then
  echo "WARNING: coralnpu HEAD is $HEAD_COMMIT, not the pinned base." >&2
  echo "         The patches were generated against $BASE_COMMIT." >&2
  echo "         For an exact reproduction run:" >&2
  echo "           git -C \"$CORALNPU\" checkout $BASE_COMMIT" >&2
  read -r -p "Continue anyway? [y/N] " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "Aborted."; exit 1; }
fi

echo ">> Checking patches apply cleanly..."
for p in "$HERE"/patches/*.patch; do
  git -C "$CORALNPU" apply --check "$p"
done

echo ">> Applying patches (modified upstream files)..."
for p in "$HERE"/patches/*.patch; do
  echo "   - $(basename "$p")"
  git -C "$CORALNPU" apply "$p"
done

echo ">> Copying new files (overlay)..."
cp -rv "$HERE/overlay/." "$CORALNPU/" >/dev/null
echo "   done."

echo ">> Regenerating the 10 validation images (needs numpy, pillow, network)..."
if python3 "$CORALNPU/tests/npusim_examples/prepare_val_images.py" --seed 42 --count 10; then
  echo "   images regenerated."
else
  echo "WARNING: image generation failed (offline or missing numpy/pillow)." >&2
  echo "         Run it manually before the verify step:" >&2
  echo "           python3 tests/npusim_examples/prepare_val_images.py --seed 42 --count 10" >&2
fi

cat <<EOF

Done. Now build and run the kernel verification from the coralnpu root:

  cd "$CORALNPU"
  bazel run //tests/npusim_examples:npusim_verify_val10

EOF

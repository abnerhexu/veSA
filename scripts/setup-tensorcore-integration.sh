#!/bin/sh
# Fetch only the locked Scala dependency graph; no compiler/toolchain or SoC checkout.
set -eu
TENSORCORE_SETUP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TENSORCORE_PROJECT_DIR=$(CDPATH= cd -- "$TENSORCORE_SETUP_DIR/.." && pwd)
TENSORCORE_DEPS_DIR="$TENSORCORE_PROJECT_DIR/out/integration-deps"
mkdir -p "$TENSORCORE_DEPS_DIR"
while read -r dependency repository revision; do
  case "$dependency" in ''|'#'*) continue ;; esac
  case "$dependency" in
    rocket-chip) source_dirs='src macros' ;;
    hardfloat) source_dirs='hardfloat/src' ;;
    cde) source_dirs='cde/src' ;;
    firesim) source_dirs='sim/midas/targetutils' ;;
    *) echo "Unknown locked dependency: $dependency" >&2; exit 2 ;;
  esac
  checkout="$TENSORCORE_DEPS_DIR/$dependency"
  if [ ! -e "$checkout" ]; then
    git init --quiet "$checkout"
    git -C "$checkout" remote add origin "$repository"
  fi
  if [ "$(git -C "$checkout" remote get-url origin)" != "$repository" ]; then
    echo "Refusing to change an unexpected dependency remote: $checkout" >&2
    exit 2
  fi
  if [ -n "$(git -C "$checkout" status --porcelain)" ]; then
    echo "Preserve modified dependency checkout; clean it manually: $checkout" >&2
    exit 2
  fi
  if ! git -C "$checkout" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$checkout" fetch --depth 1 --filter=blob:none origin "$revision"
  fi
  git -C "$checkout" sparse-checkout init --cone
  # Intentional splitting of the fixed, whitespace-free directory list above.
  # shellcheck disable=SC2086
  git -C "$checkout" sparse-checkout set $source_dirs
  git -C "$checkout" checkout --detach "$revision"
  test "$(git -C "$checkout" rev-parse HEAD)" = "$revision"
done < "$TENSORCORE_PROJECT_DIR/integration/dependencies.lock"

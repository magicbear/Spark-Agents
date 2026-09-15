#!/usr/bin/env bash
# Apply the 7-file DeepSeek-V4.1 overlay5 GB10 patch.
# Usage:
#   apply.sh bind   [PATCH_DIR] [SITE]
#   apply.sh copy   [PATCH_DIR] [SITE]
#   apply.sh patch  [UNIFIED.patch] [VLLM_ROOT]
#   apply.sh mounts [PATCH_DIR] [SITE]     # print docker -v lines only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CMD="${1:-}"
SITE_DEFAULT="/usr/local/lib/python3.12/dist-packages/vllm"

die() { echo "error: $*" >&2; exit 2; }

print_mounts() {
  local dir="$1" site="$2"
  test -f "$dir/mounts.txt" || die "missing $dir/mounts.txt"
  while read -r f rel; do
    [ -z "${f:-}" ] && continue
    test -f "$dir/$f" || die "missing $dir/$f"
    printf -- ' -v %s:%s/%s:ro' "$dir/$f" "$site" "$rel"
  done < "$dir/mounts.txt"
  echo
}

copy_files() {
  local dir="$1" site="$2"
  test -f "$dir/mounts.txt" || die "missing $dir/mounts.txt"
  while read -r f rel; do
    [ -z "${f:-}" ] && continue
    test -f "$dir/$f" || die "missing $dir/$f"
    dest="$site/$rel"
    mkdir -p "$(dirname "$dest")"
    cp -a "$dir/$f" "$dest"
    echo "copied $f -> $dest"
  done < "$dir/mounts.txt"
}

case "$CMD" in
  bind|mounts)
    DIR="${2:-$HERE/files}"
    SITE="${3:-$SITE_DEFAULT}"
    print_mounts "$DIR" "$SITE"
    ;;
  copy)
    DIR="${2:-$HERE/files}"
    SITE="${3:-$SITE_DEFAULT}"
    copy_files "$DIR" "$SITE"
    ;;
  patch)
    P="${2:-$HERE/overlay5-gb10.patch}"
    ROOT="${3:-.}"
    test -f "$P" || die "missing $P"
    test -d "$ROOT/vllm" || die "$ROOT/vllm not found (pass dist-packages or vLLM source root)"
    patch -d "$ROOT" -p1 --dry-run < "$P"
    patch -d "$ROOT" -p1 < "$P"
    echo "patched $ROOT"
    ;;
  *)
    cat <<EOF
usage:
  $0 bind   [files-dir] [site-packages/vllm]   # print docker -v lines (production)
  $0 copy   [files-dir] [site-packages/vllm]   # overwrite files in a tree
  $0 patch  [overlay5-gb10.patch] [vllm-root]  # patch -p1 onto source or dist-packages
  $0 mounts [files-dir] [site-packages/vllm]   # alias of bind
vllm-root must contain a vllm/ package directory.
EOF
    exit 2
    ;;
esac

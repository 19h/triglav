#!/bin/bash
# Compatibility entry point for the production-binary acceptance harness.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
case "${1:---local}" in
  --local)
    shift "$(( $# > 0 ? 1 : 0 ))"
    exec bash "$ROOT/validation/run.sh" "$@"
    ;;
  --help|-h)
    printf 'Usage: %s [--local] [fresh-evidence-directory]\n' "$0"
    printf 'Runs real production binaries in isolated Linux namespaces using Docker.\n'
    ;;
  *)
    printf 'Unsupported legacy option: %s\nSee validation/README.md for production validation and remote deployment.\n' "$1" >&2
    exit 2
    ;;
esac

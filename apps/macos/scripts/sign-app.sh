#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_BUNDLE="${1:-$APP_DIR/dist/Triglav.app}"
IDENTITY="${CODESIGN_IDENTITY:-${2:-}}"

if [[ -z "$IDENTITY" ]]; then
  echo "error: set CODESIGN_IDENTITY or pass it as the second argument" >&2
  exit 1
fi

if [[ ! -d "$APP_BUNDLE" ]]; then
  echo "error: app bundle not found at $APP_BUNDLE" >&2
  exit 1
fi

echo "==> Signing embedded CLI"
codesign --force --timestamp --options runtime --sign "$IDENTITY" "$APP_BUNDLE/Contents/Resources/bin/triglav"

if [[ -x "$APP_BUNDLE/Contents/Library/LaunchServices/TriglavPrivilegedHelper" ]]; then
  echo "==> Signing privileged helper"
  codesign --force --timestamp --options runtime --sign "$IDENTITY" "$APP_BUNDLE/Contents/Library/LaunchServices/TriglavPrivilegedHelper"
fi

shopt -s nullglob
for bundle in "$APP_BUNDLE/Contents/Resources"/*.bundle; do
  echo "==> Signing resource bundle $(basename "$bundle")"
  codesign --force --timestamp --options runtime --sign "$IDENTITY" "$bundle"
done
shopt -u nullglob

echo "==> Signing app bundle"
codesign --force --timestamp --options runtime --sign "$IDENTITY" "$APP_BUNDLE"

echo "==> Verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP_BUNDLE"
spctl --assess --type execute --verbose "$APP_BUNDLE"

echo "Signed $APP_BUNDLE"

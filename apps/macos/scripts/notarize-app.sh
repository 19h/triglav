#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_BUNDLE="${1:-$APP_DIR/dist/Triglav.app}"
ZIP_PATH="${2:-$APP_DIR/dist/Triglav.zip}"
PROFILE="${NOTARY_KEYCHAIN_PROFILE:-${3:-}}"

if [[ -z "$PROFILE" ]]; then
  echo "error: set NOTARY_KEYCHAIN_PROFILE or pass it as the third argument" >&2
  exit 1
fi

if [[ ! -d "$APP_BUNDLE" ]]; then
  echo "error: app bundle not found at $APP_BUNDLE" >&2
  exit 1
fi

echo "==> Creating notarization archive"
rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$APP_BUNDLE" "$ZIP_PATH"

echo "==> Submitting for notarization"
xcrun notarytool submit "$ZIP_PATH" --keychain-profile "$PROFILE" --wait

echo "==> Stapling ticket"
xcrun stapler staple "$APP_BUNDLE"
xcrun stapler validate "$APP_BUNDLE"

echo "Notarized $APP_BUNDLE"

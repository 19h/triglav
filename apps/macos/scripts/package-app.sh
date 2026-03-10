#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$APP_DIR/../.." && pwd)"
DIST_DIR="${DIST_DIR:-$APP_DIR/dist}"
APP_BUNDLE="$DIST_DIR/Triglav.app"
INFO_PLIST="$APP_DIR/Packaging/Info.plist"

mkdir -p "$DIST_DIR"

echo "==> Building TriglavGUI"
(cd "$APP_DIR" && swift build -c release)
GUI_BIN_DIR="$(cd "$APP_DIR" && swift build -c release --show-bin-path)"
GUI_BIN="$GUI_BIN_DIR/TriglavGUI"

TRIGLAV_BIN="${TRIGLAV_BIN:-$REPO_DIR/target/release/triglav}"
if [[ ! -x "$TRIGLAV_BIN" ]]; then
  echo "==> Building embedded triglav CLI"
  (cd "$REPO_DIR" && cargo build --release --bin triglav)
fi

if [[ ! -x "$TRIGLAV_BIN" ]]; then
  echo "error: embedded triglav binary not found at $TRIGLAV_BIN" >&2
  exit 1
fi

echo "==> Assembling app bundle"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources/bin"

cp "$GUI_BIN" "$APP_BUNDLE/Contents/MacOS/TriglavGUI"
cp "$TRIGLAV_BIN" "$APP_BUNDLE/Contents/Resources/bin/triglav"
cp "$INFO_PLIST" "$APP_BUNDLE/Contents/Info.plist"

shopt -s nullglob
for bundle in "$GUI_BIN_DIR"/*.bundle; do
  cp -R "$bundle" "$APP_BUNDLE/Contents/Resources/"
done
shopt -u nullglob

chmod 755 "$APP_BUNDLE/Contents/MacOS/TriglavGUI" "$APP_BUNDLE/Contents/Resources/bin/triglav"

echo "Created $APP_BUNDLE"
echo "Embedded CLI: $TRIGLAV_BIN"

if [[ -n "${CODESIGN_IDENTITY:-}" ]]; then
  echo "==> Signing bundle"
  "$APP_DIR/scripts/sign-app.sh" "$APP_BUNDLE"
fi

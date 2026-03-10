#!/usr/bin/env bash
set -euo pipefail

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "error: xcodebuild is unavailable. Install full Xcode and select it with xcode-select." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_PATH="$APP_DIR/Triglav.xcodeproj"
SCHEME="${SCHEME:-Triglav}"
CONFIGURATION="${CONFIGURATION:-Debug}"
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-$APP_DIR/.xcode-derived-data}"

xcodebuild \
  -project "$PROJECT_PATH" \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -derivedDataPath "$DERIVED_DATA_PATH" \
  build

echo "Built app bundle: $DERIVED_DATA_PATH/Build/Products/$CONFIGURATION/Triglav.app"

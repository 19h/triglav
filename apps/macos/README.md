# Triglav macOS app

This directory contains a native SwiftUI macOS control app for Triglav.

What it includes:

- main dashboard window for client configuration and logs
- status bar extra with quick connect/disconnect actions
- settings window
- option to hide the status bar icon while keeping the app reachable in the Dock
- proxy-mode process management in-app
- TUN-mode launchd daemon management with an administrator prompt
- live client status from a local HTTP status endpoint
- an Xcode app project with a blessed privileged helper target

## Build

Open the Xcode project for the full app + privileged helper setup:

```bash
open apps/macos/Triglav.xcodeproj
```

If you have full Xcode installed, you can build the project from the command line too:

```bash
./apps/macos/scripts/build-xcode-app.sh
```

Open the package in Xcode:

```bash
open apps/macos/Package.swift
```

Or run it directly with SwiftPM:

```bash
cd apps/macos
swift run
```

Use the SwiftPM path for the lightweight developer shell. Use the Xcode project when you want the real privileged helper target, app icon asset catalog, and full app-bundle signing settings.

## Triglav binary

The app looks for `triglav` in:

- the custom path from Settings
- an embedded app bundle binary at `Contents/Resources/bin/triglav`
- `PATH`
- nearby Cargo build outputs like `target/debug/triglav`

If you have not installed the CLI globally yet, build it first:

```bash
cargo build
```

Then either:

- leave auto-detection on if the app finds `target/debug/triglav`
- or point Settings to your built binary manually

## Notes

- `Proxy` mode is managed directly by the app and streams logs live.
- The Xcode app build includes a blessed privileged helper target that installs and removes the root `launchd` daemon for TUN mode.
- The SwiftPM build falls back to the direct administrator-prompt flow so local iteration still works without the helper target.
- When the status bar icon is hidden, the app switches to a normal Dock app so it never disappears completely.
- The GUI asks the CLI to expose status on `http://127.0.0.1:9091/status` and uses that for live metrics and uplink telemetry.

## Xcode helper setup

Before using the privileged helper from Xcode, set your signing team in the project and verify these identifiers if you customize them:

- app bundle id: `com.triglav.gui`
- helper bundle id: `com.triglav.gui.helper`
- helper label: `com.triglav.gui.helper`
- launch daemon label: `com.triglav.gui.tun`

The Xcode app target builds the Rust `triglav` binary into the app resources and embeds the helper into `Contents/Library/LaunchServices` during the build.

## Package a .app bundle

```bash
./apps/macos/scripts/package-app.sh
```

That produces `apps/macos/dist/Triglav.app` and embeds the `triglav` CLI inside the bundle.

If you already have a CLI binary you want to embed, point the script at it:

```bash
TRIGLAV_BIN=/path/to/triglav ./apps/macos/scripts/package-app.sh
```

## Sign and notarize

Sign with a Developer ID identity:

```bash
CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
  ./apps/macos/scripts/sign-app.sh
```

Notarize after storing credentials with `xcrun notarytool store-credentials`:

```bash
NOTARY_KEYCHAIN_PROFILE="triglav-notary" \
  ./apps/macos/scripts/notarize-app.sh
```

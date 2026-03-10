// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TriglavGUI",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(
            name: "TriglavGUI",
            targets: ["TriglavGUI"]
        ),
    ],
    targets: [
        .executableTarget(
            name: "TriglavGUI",
            path: "Sources/TriglavGUI"
        ),
    ]
)

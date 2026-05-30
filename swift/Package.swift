// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "GalliumCLI",
    platforms: [.macOS("26.0")],
    products: [
        .executable(name: "gallium", targets: ["GalliumCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/jpsim/Yams.git", from: "5.0.0"),
    ],
    targets: [
        // Main CLI binary: text mode + voice mode
        .executableTarget(
            name: "GalliumCLI",
            dependencies: [
                "AgentBridge",
                "Util",
                "TTS",
                "Audio",
                "ScreenCapture",
                "CEditline",
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // UniFFI-generated Swift bindings for gallium_agent Rust library
        .target(
            name: "AgentBridge",
            dependencies: ["AgentBridgeFFI"],
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [
                .unsafeFlags([
                    "-L../target/release",
                    "-lgallium_agent",
                ])
            ]
        ),
        // System library: bridging header + module map for the C FFI symbols
        .systemLibrary(
            name: "AgentBridgeFFI",
            path: "Sources/AgentBridgeFFI",
            pkgConfig: nil,
            providers: nil
        ),
        .target(
            name: "Util",
            dependencies: ["Yams"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "TTS",
            dependencies: ["Util"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "Audio",
            dependencies: ["Util"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .systemLibrary(
            name: "CEditline",
            path: "Sources/CEditline"
        ),
        .target(
            name: "ScreenCapture",
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [
                .linkedFramework("ScreenCaptureKit"),
                .linkedFramework("AppKit"),
            ]
        ),
    ]
)

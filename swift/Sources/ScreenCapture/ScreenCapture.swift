import Foundation
import ScreenCaptureKit
import AppKit

/// One-shot screen capture using ScreenCaptureKit (macOS 14+).
///
/// On first call, macOS will present a screen recording permission dialog.
/// After the user grants access, subsequent calls proceed without a prompt.
public enum ScreenCapture {

    public enum Error: Swift.Error, LocalizedError {
        case noDisplayFound
        case encodingFailed

        public var errorDescription: String? {
            switch self {
            case .noDisplayFound: return "No display available for capture"
            case .encodingFailed:  return "Failed to encode screenshot as JPEG"
            }
        }
    }

    /// Capture the main (first) display and return JPEG data.
    public static func captureMainDisplay() async throws -> Data {
        try await captureDisplay(index: 0)
    }

    /// Capture a specific display by zero-based index.
    public static func captureDisplay(index: Int = 0) async throws -> Data {
        let content = try await SCShareableContent.current
        guard index < content.displays.count else {
            throw Error.noDisplayFound
        }
        let display = content.displays[index]

        let filter = SCContentFilter(
            display: display,
            excludingApplications: [],
            exceptingWindows: []
        )
        let config = SCStreamConfiguration()
        config.width  = display.width
        config.height = display.height

        let cgImage = try await SCScreenshotManager.captureImage(
            contentFilter: filter,
            configuration: config
        )

        let rep = NSBitmapImageRep(cgImage: cgImage)
        guard let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.85]) else {
            throw Error.encodingFailed
        }
        return jpeg
    }

    /// Return human-readable labels for all connected displays.
    public static func listDisplays() async throws -> [String] {
        let content = try await SCShareableContent.current
        return content.displays.enumerated().map { i, d in
            "Display \(i + 1): \(d.width)×\(d.height)"
        }
    }
}

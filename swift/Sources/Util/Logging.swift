import Foundation

public enum LogLevel: String {
    case debug = "DEBUG"
    case info = "INFO"
    case warning = "WARN"
    case error = "ERROR"
}

public struct Logger {
    private let name: String
    private static var minLevel: LogLevel = .info

    public init(_ name: String) { self.name = name }

    public static func setLevel(_ level: LogLevel) { minLevel = level }

    public func debug(_ message: String) { log(.debug, message) }
    public func info(_ message: String)  { log(.info, message) }
    public func warning(_ message: String) { log(.warning, message) }
    public func error(_ message: String) { log(.error, message) }

    private func log(_ level: LogLevel, _ message: String) {
        guard shouldLog(level) else { return }
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let output = "[\(timestamp)] [\(level.rawValue)] [\(name)] \(message)"
        if level == .error || level == .warning {
            fputs(output + "\n", stderr)
        } else {
            print(output)
        }
    }

    private func shouldLog(_ level: LogLevel) -> Bool {
        let levels: [LogLevel] = [.debug, .info, .warning, .error]
        guard let current = levels.firstIndex(of: Self.minLevel),
              let requested = levels.firstIndex(of: level) else { return false }
        return requested >= current
    }
}

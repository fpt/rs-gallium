import Foundation
import Yams

/// Configuration loaded from a YAML file (default: configs/default.yaml).
public struct Config: Codable {
    public let llm: LLMConfig
    public let tts: TTSConfig?
    public let stt: STTConfig?
    public let mcpServers: [McpServer]?

    public struct McpServer: Codable {
        public let command: String
        public let args: [String]
    }

    public struct LLMConfig: Codable {
        public let baseURL: String
        public let model: String
        public let apiKey: String?
        public let temperature: Float?
        public let maxTokens: Int
        public let contextWindow: Int?
        public let reasoningEffort: String?
        public let systemPrompt: String?
    }

    public struct TTSConfig: Codable {
        public let enabled: Bool
        public let voice: String?
        public let rate: Float
        public let pitchMultiplier: Float
        public let volume: Float
    }

    public struct STTConfig: Codable {
        public let enabled: Bool
        public let locale: String?
        public let censor: Bool?
    }

    public static func load(from path: String) throws -> Config {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let yaml = String(data: data, encoding: .utf8) ?? ""
        let decoder = YAMLDecoder()
        return try decoder.decode(Config.self, from: yaml)
    }

    public static func `default`() -> Config {
        Config(
            llm: LLMConfig(
                baseURL: "https://api.openai.com/v1",
                model: "gpt-4o-mini",
                apiKey: ProcessInfo.processInfo.environment["OPENAI_API_KEY"],
                temperature: 0.7,
                maxTokens: 512,
                contextWindow: 32000,
                reasoningEffort: nil,
                systemPrompt: nil
            ),
            tts: nil,
            stt: nil,
            mcpServers: nil
        )
    }
}

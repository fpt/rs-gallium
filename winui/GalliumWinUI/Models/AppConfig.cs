using YamlDotNet.Serialization;

namespace GalliumWinUI.Models;

public sealed class AppConfig
{
    [YamlMember(Alias = "llm")]
    public LlmSection Llm { get; set; } = new();

    [YamlMember(Alias = "tts")]
    public TtsSection? Tts { get; set; }

    [YamlMember(Alias = "stt")]
    public SttSection? Stt { get; set; }

    [YamlMember(Alias = "mcpServers")]
    public List<McpServerEntry>? McpServers { get; set; }

    public sealed class LlmSection
    {
        [YamlMember(Alias = "baseURL")]
        public string BaseUrl { get; set; } = "https://api.openai.com/v1";

        [YamlMember(Alias = "model")]
        public string Model { get; set; } = "gpt-4o-mini";

        [YamlMember(Alias = "apiKey")]
        public string? ApiKey { get; set; }

        [YamlMember(Alias = "temperature")]
        public float? Temperature { get; set; }

        [YamlMember(Alias = "maxTokens")]
        public int MaxTokens { get; set; } = 512;

        [YamlMember(Alias = "contextWindow")]
        public int? ContextWindow { get; set; }

        [YamlMember(Alias = "reasoningEffort")]
        public string? ReasoningEffort { get; set; }

        [YamlMember(Alias = "systemPrompt")]
        public string? SystemPrompt { get; set; }
    }

    public sealed class TtsSection
    {
        [YamlMember(Alias = "enabled")]
        public bool Enabled { get; set; }
    }

    public sealed class SttSection
    {
        [YamlMember(Alias = "enabled")]
        public bool Enabled { get; set; }

        [YamlMember(Alias = "locale")]
        public string? Locale { get; set; }
    }

    public sealed class McpServerEntry
    {
        [YamlMember(Alias = "command")]
        public string Command { get; set; } = "";

        [YamlMember(Alias = "args")]
        public List<string> Args { get; set; } = [];
    }

    /// Load from the first candidate path that exists, falling back to defaults.
    public static AppConfig Load()
    {
        var candidates = new[]
        {
            Environment.GetEnvironmentVariable("GALLIUM_CONFIG"),
            Path.Combine(Directory.GetCurrentDirectory(), "configs", "default.yaml"),
            FindInAncestors(AppContext.BaseDirectory, Path.Combine("configs", "default.yaml")),
        };

        foreach (var path in candidates)
        {
            if (path is not null && File.Exists(path))
            {
                var yaml = File.ReadAllText(path);
                return new DeserializerBuilder()
                    .IgnoreUnmatchedProperties()
                    .Build()
                    .Deserialize<AppConfig>(yaml) ?? new AppConfig();
            }
        }

        return new AppConfig();
    }

    private static string? FindInAncestors(string start, string relative)
    {
        for (var dir = new DirectoryInfo(start); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }
}

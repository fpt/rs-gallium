using GalliumWinUI.Models;
using uniffi.gallium_agent;

namespace GalliumWinUI.Services;

internal sealed class AgentService : IDisposable
{
    private Agent? _agent;

    public bool IsReady => _agent != null;

    public async Task InitializeAsync(AppConfig cfg)
    {
        var apiKey = cfg.Llm.ApiKey
            ?? Environment.GetEnvironmentVariable("OPENAI_API_KEY")
            ?? throw new InvalidOperationException(
                "No API key found. Set OPENAI_API_KEY or add apiKey to configs/default.yaml.");

        var config = new CloudAgentConfig(
            baseUrl:         cfg.Llm.BaseUrl,
            model:           cfg.Llm.Model,
            apiKey:          apiKey,
            temperature:     cfg.Llm.Temperature,
            maxTokens:       (uint)cfg.Llm.MaxTokens,
            contextWindow:   (uint)(cfg.Llm.ContextWindow ?? 32000),
            workingDir:      Directory.GetCurrentDirectory(),
            reasoningEffort: cfg.Llm.ReasoningEffort,
            systemPrompt:    cfg.Llm.SystemPrompt,
            mcpServers:      (cfg.McpServers ?? [])
                                 .Select(s => new McpServerConfig(s.Command, s.Args))
                                 .ToList()
        );

        _agent = await Task.Run(() => GalliumAgentMethods.AgentNew(config));
    }

    public async Task<AgentResult> StepAsync(string input)
    {
        if (_agent is null) throw new InvalidOperationException("Agent not initialized.");
        var resp = await Task.Run(() => _agent.Step(input));
        return new AgentResult(resp.content, resp.inputTokens, resp.outputTokens, resp.totalTokens, resp.contextPercent);
    }

    public void Reset() => _agent?.Reset();

    public void SetSystemPrompt(string prompt) => _agent?.SetSystemPrompt(prompt);

    public void Dispose() => _agent?.Dispose();
}

internal sealed record AgentResult(
    string Content,
    ulong InputTokens,
    ulong OutputTokens,
    ulong TotalTokens,
    float ContextPercent);

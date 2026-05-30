import Foundation
import Util
import TTS
import Audio
import CEditline

let logger = Logger("Main")

// ─── readline (libedit) callback globals ─────────────────────────────────────
nonisolated(unsafe) var _rlCompletedLine: UnsafeMutablePointer<CChar>? = nil
nonisolated(unsafe) var _rlLineReady = false
nonisolated(unsafe) var _rlGotEOF = false

private func rlLineCallback(_ line: UnsafeMutablePointer<CChar>?) {
    if line != nil { _rlCompletedLine = line; _rlLineReady = true }
    else { _rlGotEOF = true }
}

// ─── Entry point ─────────────────────────────────────────────────────────────
@main
struct GalliumCLI {
    static func main() async { await runMain() }
}

@MainActor
func runMain() async {
    let args = CommandLine.arguments

    var configPath = "configs/default.yaml"
    var voiceMode = false

    var i = 1
    while i < args.count {
        switch args[i] {
        case "--config" where i + 1 < args.count:
            configPath = args[i + 1]; i += 2
        case "--voice":
            voiceMode = true; i += 1
        case "--verbose", "-v":
            Logger.setLevel(.debug); i += 1
        case "--help", "-h":
            printHelp(); exit(0)
        default:
            i += 1
        }
    }

    // ── Load config ──────────────────────────────────────────────────────────
    let config: Config
    do {
        if FileManager.default.fileExists(atPath: configPath) {
            config = try Config.load(from: configPath)
            logger.info("Loaded config from \(configPath)")
        } else {
            config = Config.default()
        }
    } catch {
        logger.error("Failed to load config: \(error)")
        config = Config.default()
    }

    // ── Resolve API key ──────────────────────────────────────────────────────
    let apiKey = config.llm.apiKey
        ?? ProcessInfo.processInfo.environment["OPENAI_API_KEY"]
        ?? ""

    if apiKey.isEmpty {
        fputs("Error: No OpenAI API key. Set OPENAI_API_KEY or add apiKey to config.\n", stderr)
        fputs("  For local gallium model inference, use the gallium-agent binary.\n", stderr)
        exit(1)
    }

    // ── Build CloudAgentConfig ───────────────────────────────────────────────
    let agentConfig = CloudAgentConfig(
        baseUrl: config.llm.baseURL,
        model: config.llm.model,
        apiKey: apiKey,
        temperature: config.llm.temperature,
        maxTokens: UInt32(config.llm.maxTokens),
        contextWindow: UInt32(config.llm.contextWindow ?? 32000),
        workingDir: FileManager.default.currentDirectoryPath,
        reasoningEffort: config.llm.reasoningEffort,
        systemPrompt: config.llm.systemPrompt,
        mcpServers: (config.mcpServers ?? []).map {
            McpServerConfig(command: $0.command, args: $0.args)
        }
    )

    let agent: Agent
    do {
        agent = try agentNew(config: agentConfig)
    } catch {
        fputs("Failed to create agent: \(error)\n", stderr)
        exit(1)
    }

    // ── TTS ──────────────────────────────────────────────────────────────────
    let ttsEnabled = config.tts?.enabled ?? false
    let tts: TextToSpeech? = ttsEnabled ? TextToSpeech(config: TextToSpeech.Config(
        enabled: true,
        voice: config.tts?.voice,
        rate: config.tts?.rate ?? 0.52,
        pitchMultiplier: config.tts?.pitchMultiplier ?? 1.0,
        volume: config.tts?.volume ?? 1.0
    )) : nil

    // ── STT ──────────────────────────────────────────────────────────────────
    let sttEnabled = voiceMode && (config.stt?.enabled ?? false)
    let audioCapture: AudioCapture? = sttEnabled ? AudioCapture(config: AudioCapture.Config(
        enabled: true,
        locale: Locale(identifier: config.stt?.locale ?? Locale.current.identifier(.bcp47)),
        censor: config.stt?.censor ?? false
    )) : nil

    if let capture = audioCapture {
        do {
            guard await capture.requestMicrophonePermission() else {
                fputs("Microphone permission denied.\n", stderr); exit(1)
            }
            try await capture.initialize()
        } catch {
            fputs("STT init failed: \(error). Falling back to text mode.\n", stderr)
        }
    }

    // ── Mode banner ──────────────────────────────────────────────────────────
    let modeName = sttEnabled ? "voice" : "text"
    print("gallium | \(config.llm.model) | mode: \(modeName) | /help for commands")

    // ── Run REPL ─────────────────────────────────────────────────────────────
    if sttEnabled, let capture = audioCapture {
        await runVoiceREPL(agent: agent, tts: tts, capture: capture)
    } else {
        await runTextREPL(agent: agent, tts: tts)
    }
}

// ─── Text REPL ────────────────────────────────────────────────────────────────
@MainActor
func runTextREPL(agent: Agent, tts: TextToSpeech?) async {
    let el = el_init("gallium", stdin, stdout, stderr)
    el_set(el, EL_PROMPT, { _ in UnsafePointer<CChar>(strdup("> ")) })
    el_set(el, EL_EDITOR, "emacs")
    let hist = history_init()
    var ev = HistEvent()
    history(hist, &ev, H_SETSIZE, 1000)
    el_set(el, EL_HIST, history, hist)

    while true {
        var count: Int32 = 0
        guard let cline = el_gets(el, &count), count > 0 else { break }
        let line = String(cString: cline).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty else { continue }
        history(hist, &ev, H_ENTER, cline)

        if let cmd = handleCommand(line, agent: agent) {
            if cmd == .quit { break }
            continue
        }

        await processInput(line, agent: agent, tts: tts)
    }

    el_end(el)
    history_end(hist)
}

// ─── Voice REPL ───────────────────────────────────────────────────────────────
@MainActor
func runVoiceREPL(agent: Agent, tts: TextToSpeech?, capture: AudioCapture) async {
    let voiceQueue = VoiceQueue()

    capture.onFinalResult = { text in
        Task { @MainActor in
            voiceQueue.enqueue(text)
        }
    }

    do { try await capture.start() }
    catch { fputs("Failed to start mic: \(error)\n", stderr) }

    print("🎤 Listening... (speak to interact, /quit to exit)")

    while true {
        try? await Task.sleep(nanoseconds: 50_000_000)
        if let text = voiceQueue.dequeue() {
            if text.lowercased().hasPrefix("/quit") { break }
            print("\n[You] \(text)")
            capture.mute()
            await processInput(text, agent: agent, tts: tts)
            capture.unmute()
        }
    }

    capture.stop()
}

// ─── Shared input handler ─────────────────────────────────────────────────────
enum ReplCommand { case quit }

func handleCommand(_ line: String, agent: Agent) -> ReplCommand? {
    switch line {
    case "/quit", "/exit", "/q": return .quit
    case "/reset":
        agent.reset()
        print("[Conversation reset]")
    case "/help":
        printHelp()
    default:
        if line.hasPrefix("/") {
            print("Unknown command. Type /help.")
        } else {
            return nil
        }
    }
    return nil  // consumed but not quit
}

@MainActor
func processInput(_ text: String, agent: Agent, tts: TextToSpeech?) async {
    do {
        let resp = try agent.step(userInput: text)
        print(resp.content)
        if let tts = tts {
            await tts.speakAsync(resp.content)
        }
        if resp.totalTokens > 0 {
            fputs("[in=\(resp.inputTokens) out=\(resp.outputTokens) ctx=\(String(format: "%.0f", resp.contextPercent))%]\n", stderr)
        }
    } catch {
        fputs("[Error] \(error)\n", stderr)
    }
}

// ─── Voice queue (MainActor → async drain) ────────────────────────────────────
final class VoiceQueue: @unchecked Sendable {
    private var queue: [String] = []
    private var lock = os_unfair_lock()

    func enqueue(_ text: String) {
        os_unfair_lock_lock(&lock)
        queue.append(text)
        os_unfair_lock_unlock(&lock)
    }

    func dequeue() -> String? {
        os_unfair_lock_lock(&lock)
        let v = queue.isEmpty ? nil : queue.removeFirst()
        os_unfair_lock_unlock(&lock)
        return v
    }
}

// ─── Help text ────────────────────────────────────────────────────────────────
func printHelp() {
    print("""
    gallium — local LLM agent with voice support

    Options:
      --config PATH    Config file (default: configs/default.yaml)
      --voice          Enable voice input (STT) + output (TTS)
      --verbose        Verbose logging

    Session commands:
      /reset           Clear conversation history
      /quit            Exit
      /help            Show this help

    For local model inference (GPT-OSS, Gemma 4, Qwen 3.5), run:
      gallium-agent --arch gemma4 --hf-repo google/gemma-4-E4B ...
    """)
}

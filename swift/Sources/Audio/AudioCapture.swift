import Foundation
import AVFoundation
import Speech
import Util

/// Live microphone transcription using Apple SpeechTranscriber (macOS 26+).
@MainActor
public class AudioCapture {
    private let logger = Logger("Audio")
    private let config: Config

    private var transcriber: SpeechTranscriber?
    private var analyzer: SpeechAnalyzer?
    private var analyzerFormat: AVAudioFormat?
    private let audioEngine = AVAudioEngine()
    nonisolated(unsafe) private var converter: AVAudioConverter?
    nonisolated(unsafe) private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    private var resultsTask: Task<Void, any Error>?
    private var isRunning = false
    nonisolated(unsafe) private var _muted = false
    nonisolated(unsafe) private var _muteGeneration: UInt64 = 0

    public var onVolatileResult: ((String) -> Void)?
    public var onFinalResult: ((String) -> Void)?

    public struct Config {
        public let enabled: Bool
        public let locale: Locale
        public let censor: Bool

        public init(enabled: Bool, locale: Locale = .current, censor: Bool = false) {
            self.enabled = enabled
            self.locale = locale
            self.censor = censor
        }
    }

    public enum AudioCaptureError: Error {
        case speechTranscriberNotAvailable
        case unsupportedLocale(String)
        case notInitialized
        case permissionDenied
        case noCompatibleAudioFormat
    }

    public init(config: Config) {
        self.config = config
    }

    public func initialize() async throws {
        guard config.enabled else { return }
        guard SpeechTranscriber.isAvailable else {
            throw AudioCaptureError.speechTranscriberNotAvailable
        }

        let supportedLocales = await SpeechTranscriber.supportedLocales
        guard supportedLocales.contains(where: {
            $0.identifier(.bcp47) == config.locale.identifier(.bcp47)
        }) else {
            throw AudioCaptureError.unsupportedLocale(config.locale.identifier)
        }

        for reserved in await AssetInventory.reservedLocales {
            await AssetInventory.release(reservedLocale: reserved)
        }
        try await AssetInventory.reserve(locale: config.locale)

        let transcriber = SpeechTranscriber(
            locale: config.locale,
            transcriptionOptions: config.censor ? [.etiquetteReplacements] : [],
            reportingOptions: [.volatileResults],
            attributeOptions: []
        )
        self.transcriber = transcriber

        let modules: [any SpeechModule] = [transcriber]
        let installedLocales = await SpeechTranscriber.installedLocales
        if !installedLocales.contains(where: {
            $0.identifier(.bcp47) == config.locale.identifier(.bcp47)
        }) {
            logger.info("Downloading speech model for \(config.locale.identifier)...")
            if let request = try await AssetInventory.assetInstallationRequest(supporting: modules) {
                try await request.downloadAndInstall()
            }
        }

        let analyzer = SpeechAnalyzer(modules: modules)
        self.analyzer = analyzer
        self.analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: modules)

        guard analyzerFormat != nil else {
            throw AudioCaptureError.noCompatibleAudioFormat
        }
        logger.info("SpeechTranscriber initialized (locale: \(config.locale.identifier))")
    }

    public func mute() {
        _muted = true
        _muteGeneration &+= 1
    }

    public func unmute() {
        _muteGeneration &+= 1
        _muted = false
    }

    public func requestMicrophonePermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }

    public func start() async throws {
        guard config.enabled, let transcriber = transcriber, let analyzer = analyzer,
              let targetFormat = analyzerFormat else {
            throw AudioCaptureError.notInitialized
        }

        let (inputSequence, continuation) = AsyncStream.makeStream(of: AnalyzerInput.self)
        self.inputContinuation = continuation

        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0 else { throw AudioCaptureError.permissionDenied }

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw AudioCaptureError.noCompatibleAudioFormat
        }
        self.converter = converter

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: nil) { [weak self] buffer, _ in
            self?.handleAudioBuffer(buffer)
        }
        audioEngine.prepare()
        try audioEngine.start()
        try await analyzer.start(inputSequence: inputSequence)

        resultsTask = Task {
            for try await result in transcriber.results {
                let muted = self._muted
                if muted { continue }
                let text = String(result.text.characters)
                let gen = self._muteGeneration
                if result.isFinal {
                    await MainActor.run {
                        guard self._muteGeneration == gen, !self._muted else { return }
                        self.onFinalResult?(text)
                    }
                } else {
                    await MainActor.run {
                        guard !self._muted else { return }
                        self.onVolatileResult?(text)
                    }
                }
            }
        }
        isRunning = true
    }

    public func stop() {
        guard isRunning else { return }
        inputContinuation?.finish()
        inputContinuation = nil
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        resultsTask?.cancel()
        resultsTask = nil
        isRunning = false
    }

    nonisolated private func handleAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let converter = converter,
              let targetFormat = converter.outputFormat as? AVAudioFormat,
              let targetBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: AVAudioFrameCount(
                    Double(buffer.frameLength) * targetFormat.sampleRate / buffer.format.sampleRate
                )
              )
        else { return }

        var error: NSError?
        var consumed = false
        converter.convert(to: targetBuffer, error: &error) { _, outStatus in
            if consumed { outStatus.pointee = .noDataNow; return nil }
            consumed = true
            outStatus.pointee = .haveData
            return buffer
        }

        if error == nil {
            inputContinuation?.yield(.pcmBuffer(targetBuffer))
        }
    }
}

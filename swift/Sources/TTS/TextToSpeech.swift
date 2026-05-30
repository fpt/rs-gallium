import Foundation
import AVFoundation
import Util

public class TextToSpeech: NSObject, @unchecked Sendable {
    private let synthesizer = AVSpeechSynthesizer()
    private let logger = Logger("TTS")
    private var isSpeaking = false
    private var completion: (() -> Void)?
    private let resolvedVoice: AVSpeechSynthesisVoice?

    public struct Config {
        public let enabled: Bool
        public let voice: String?
        public let rate: Float
        public let pitchMultiplier: Float
        public let volume: Float

        public init(
            enabled: Bool = true,
            voice: String? = nil,
            rate: Float = 0.52,
            pitchMultiplier: Float = 1.0,
            volume: Float = 1.0
        ) {
            self.enabled = enabled
            self.voice = voice
            self.rate = rate
            self.pitchMultiplier = pitchMultiplier
            self.volume = volume
        }
    }

    private let config: Config

    public init(config: Config) {
        self.config = config
        if let id = config.voice {
            self.resolvedVoice = AVSpeechSynthesisVoice(identifier: id)
                ?? AVSpeechSynthesisVoice(language: "en-US")
        } else {
            self.resolvedVoice = AVSpeechSynthesisVoice(language: "en-US")
        }
        super.init()
        synthesizer.delegate = self
    }

    public func speakAsync(_ text: String) async {
        guard config.enabled, !text.isEmpty else { return }
        if isSpeaking { stop() }
        await withCheckedContinuation { continuation in
            self.completion = { continuation.resume() }
            self.isSpeaking = true
            let utterance = AVSpeechUtterance(string: text)
            utterance.voice = self.resolvedVoice
            utterance.rate = self.config.rate
            utterance.pitchMultiplier = self.config.pitchMultiplier
            utterance.volume = self.config.volume
            self.synthesizer.speak(utterance)
        }
    }

    public func stop() {
        guard isSpeaking else { return }
        synthesizer.stopSpeaking(at: .immediate)
        isSpeaking = false
        completion?()
        completion = nil
    }

    public var speaking: Bool { isSpeaking }
}

extension TextToSpeech: AVSpeechSynthesizerDelegate {
    public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        isSpeaking = false
        completion?()
        completion = nil
    }

    public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        isSpeaking = false
        completion?()
        completion = nil
    }
}

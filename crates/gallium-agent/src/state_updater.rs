/// Backchannel detector for voice mode.
///
/// Detects when a short user utterance with a pause deserves a quick
/// acknowledgment ("got it", "uh-huh") without a full LLM round-trip.
pub trait BackchannelDetector: Send + Sync {
    /// Returns `Some(backchannel_text)` if the utterance warrants a quick ack.
    fn should_backchannel(&self, utterance: &str, pause_ms: u64) -> Option<String>;
}

/// Rule-based backchannel detector (lightweight, no model required).
pub struct RuleBasedBackchannelDetector;

impl RuleBasedBackchannelDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RuleBasedBackchannelDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl BackchannelDetector for RuleBasedBackchannelDetector {
    fn should_backchannel(&self, utterance: &str, pause_ms: u64) -> Option<String> {
        let word_count = utterance.split_whitespace().count();
        let lower = utterance.to_lowercase();

        // Short status updates deserve acknowledgment
        if word_count < 10 && pause_ms > 500 {
            if lower.contains("done") || lower.contains("finished") || lower.contains("できた") {
                return Some("got it".to_string());
            }

            if lower.contains("mixed")
                || lower.contains("added")
                || lower.contains("溶いた")
                || lower.contains("入れた")
            {
                return Some("mm-hmm".to_string());
            }
        }

        // Conjunction words (user likely continuing)
        if pause_ms > 400 {
            if lower.ends_with("and")
                || lower.ends_with("but")
                || lower.ends_with("so")
                || lower.ends_with("で")
                || lower.ends_with("が")
            {
                return Some("uh-huh".to_string());
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_backchannel() {
        let detector = RuleBasedBackchannelDetector::new();

        // Short status with pause
        assert_eq!(
            detector.should_backchannel("I'm done", 600),
            Some("got it".to_string())
        );

        // Mid-sentence pause
        assert_eq!(
            detector.should_backchannel("I mixed the eggs and", 450),
            Some("uh-huh".to_string())
        );

        // No backchannel for long utterance
        assert_eq!(
            detector.should_backchannel(
                "This is a very long utterance that should not trigger backchannel",
                600
            ),
            None
        );

        // No backchannel for short pause
        assert_eq!(detector.should_backchannel("I'm done", 300), None);
    }
}

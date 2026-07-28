//! Who is allowed to do what, and who gets asked.
//!
//! Before this module the answer was two-valued: prompt on a terminal, or set
//! `GALLIUM_AUTO_APPROVE=1` and allow everything. Headless runs had no third
//! option, so the testsuite exported the variable and the approval path went
//! untested — the cost recorded in issue #14 §4.
//!
//! The replacement sorts actions into [`RiskLevel`] tiers and gives each tier
//! its own rule. The tiers are the ones the issue names, and they line up with
//! the [`ToolAnnotations`](crate::tool::ToolAnnotations) that #31 already
//! attached to every tool:
//!
//! | Tier | What it covers | Default |
//! |---|---|---|
//! | [`ReadOnly`](RiskLevel::ReadOnly) | observing the workspace | allowed, never asked |
//! | [`WorkspaceWrite`](RiskLevel::WorkspaceWrite) | writing inside the workspace root | allowed for the session |
//! | [`ExternalSideEffect`](RiskLevel::ExternalSideEffect) | effects we cannot see or undo from here — the network, a remote API | ask |
//! | [`Destructive`](RiskLevel::Destructive) | may remove something unrecoverably, or escapes the workspace | ask, every time |
//!
//! Two of those defaults deserve their reasons stated.
//!
//! **A workspace write is granted without asking.** It is the tier the agent
//! spends its whole day in, the damage is bounded by a directory the user chose,
//! and the alternative is what we had: a prompt so frequent that everyone
//! disables it wholesale, which also disables the tiers that matter. Writing
//! *outside* that root is a different question, so it is classified
//! [`Destructive`](RiskLevel::Destructive) rather than waved through with it.
//!
//! **Destructive is never granted for the session.** "Yes to all" on `rm -rf`
//! is a promise about commands nobody has read yet. An
//! [`AllowForSession`](ApprovalDecision::AllowForSession) answer to a
//! destructive request is honored once and not remembered.
//!
//! Nothing here silently allows what it cannot ask about: with no terminal and
//! no client to consult, a tier whose rule is [`Ask`](ApprovalRule::Ask) is
//! denied, and the error says which rule to change.

use std::collections::HashSet;
use std::io::{IsTerminal, Write};
use std::sync::{Arc, Mutex};

use crate::AgentError;

/// How much of the world an action can reach, from least to most.
///
/// Derived per call site rather than per tool: the same `write` is a
/// [`WorkspaceWrite`](Self::WorkspaceWrite) inside the workspace and a
/// [`Destructive`](Self::Destructive) outside it, and only the call knows which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RiskLevel {
    /// Observes; changes nothing. Never asked about.
    ReadOnly,
    /// Changes the workspace the user pointed gallium at, recoverably.
    WorkspaceWrite,
    /// Lands somewhere we cannot inspect afterwards: the network, a remote API,
    /// another process. Recoverable or not, it is not ours to undo.
    ExternalSideEffect,
    /// May destroy something, or reaches outside the workspace root. The tier
    /// that is always confirmed.
    Destructive,
}

impl RiskLevel {
    /// The tier of a workspace mutation, given whether its target stayed inside
    /// the workspace root. Escaping the root is what makes an ordinary write
    /// interesting, so it is classified by where it lands, not by which tool
    /// asked.
    pub fn for_write(inside_workspace: bool) -> Self {
        if inside_workspace {
            Self::WorkspaceWrite
        } else {
            Self::Destructive
        }
    }

    /// Whether a grant for this tier may outlive the request that produced it.
    /// False for [`Destructive`](Self::Destructive): see the module docs.
    fn grantable_for_session(self) -> bool {
        self != Self::Destructive
    }

    /// The wording used in prompts and errors.
    pub fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace write",
            Self::ExternalSideEffect => "external side effect",
            Self::Destructive => "destructive",
        }
    }

    /// The config key that governs this tier, for an error that has to tell
    /// someone how to change the answer.
    fn config_key(self) -> &'static str {
        match self {
            Self::ReadOnly => "readOnly",
            Self::WorkspaceWrite => "workspaceWrite",
            Self::ExternalSideEffect => "externalSideEffect",
            Self::Destructive => "destructive",
        }
    }
}

/// What a policy says about one tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalRule {
    /// Proceed without asking anyone.
    Allow,
    /// Consult the client, or the terminal; deny if there is neither.
    Ask,
    /// Refuse without asking. For a surface that must not do this at all.
    Deny,
}

impl ApprovalRule {
    /// Parse a config value. `None` for anything else, so the caller can report
    /// the bad key rather than silently picking a rule for the user.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "allow" => Some(Self::Allow),
            "ask" | "confirm" => Some(Self::Ask),
            "deny" | "never" => Some(Self::Deny),
            _ => None,
        }
    }
}

/// An answer to one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// Approve this one action.
    AllowOnce,
    /// Approve, and stop asking about this tier for the rest of the session.
    /// Downgraded to [`AllowOnce`](Self::AllowOnce) for
    /// [`RiskLevel::Destructive`].
    AllowForSession,
    /// Refuse.
    Deny,
}

/// One question: what is about to happen, to what, and how much it can reach.
#[derive(Debug, Clone, Copy)]
pub struct ApprovalRequest<'a> {
    /// Imperative, lowercase, as shown to whoever answers: `"overwrite file"`,
    /// `"run command"`.
    pub action: &'a str,
    /// What it happens to — a path, a command line, a repository.
    pub target: &'a str,
    pub risk: RiskLevel,
}

impl<'a> ApprovalRequest<'a> {
    pub fn new(action: &'a str, target: &'a str, risk: RiskLevel) -> Self {
        Self {
            action,
            target,
            risk,
        }
    }
}

/// Answers approval questions somewhere other than the local terminal.
///
/// Installed when gallium runs headless under a driving client (the app-server),
/// where the built-in prompt has nothing to prompt.
pub trait ApprovalSink: Send + Sync {
    fn request(&self, request: &ApprovalRequest) -> Result<ApprovalDecision, AgentError>;
}

/// The rule for each tier. [`ReadOnly`](RiskLevel::ReadOnly) has no entry: a
/// tool that only observes is not worth a policy knob, and giving it one invites
/// a configuration that cannot read a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApprovalPolicy {
    pub workspace_write: ApprovalRule,
    pub external_side_effect: ApprovalRule,
    pub destructive: ApprovalRule,
}

impl Default for ApprovalPolicy {
    fn default() -> Self {
        Self {
            workspace_write: ApprovalRule::Allow,
            external_side_effect: ApprovalRule::Ask,
            destructive: ApprovalRule::Ask,
        }
    }
}

impl ApprovalPolicy {
    /// Ask about everything that is not a plain read. What an operator who wants
    /// the old prompt-on-every-write behavior back sets, and what the app-server
    /// uses so every mutation reaches the driving client.
    pub const CAUTIOUS: Self = Self {
        workspace_write: ApprovalRule::Ask,
        external_side_effect: ApprovalRule::Ask,
        destructive: ApprovalRule::Ask,
    };

    /// Ask about nothing — what `GALLIUM_AUTO_APPROVE=1` used to mean, for a
    /// sandbox that is already disposable and a test that is exercising
    /// something else. It has to be chosen deliberately in a config now, rather
    /// than being one environment variable away from any run.
    pub const PERMISSIVE: Self = Self {
        workspace_write: ApprovalRule::Allow,
        external_side_effect: ApprovalRule::Allow,
        destructive: ApprovalRule::Allow,
    };

    /// The tiers in order, for anyone reporting the policy.
    pub fn rules(&self) -> [(RiskLevel, ApprovalRule); 3] {
        [
            (RiskLevel::WorkspaceWrite, self.workspace_write),
            (RiskLevel::ExternalSideEffect, self.external_side_effect),
            (RiskLevel::Destructive, self.destructive),
        ]
    }

    pub fn rule_for(&self, risk: RiskLevel) -> ApprovalRule {
        match risk {
            RiskLevel::ReadOnly => ApprovalRule::Allow,
            RiskLevel::WorkspaceWrite => self.workspace_write,
            RiskLevel::ExternalSideEffect => self.external_side_effect,
            RiskLevel::Destructive => self.destructive,
        }
    }
}

impl std::fmt::Display for ApprovalRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Allow => "allow",
            Self::Ask => "ask",
            Self::Deny => "deny",
        })
    }
}

/// One line naming every tier and its rule, for a startup banner. Worth showing:
/// what gallium will do without asking is not something an operator should have
/// to infer from a config file they may not have written.
impl std::fmt::Display for ApprovalPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts: Vec<String> = self
            .rules()
            .iter()
            .map(|(risk, rule)| format!("{} = {}", risk.label(), rule))
            .collect();
        f.write_str(&parts.join(", "))
    }
}

/// Applies an [`ApprovalPolicy`] to each request, asking whoever is available
/// when the policy says to ask, and remembering the grants that are allowed to
/// be remembered.
///
/// One broker per session, shared by every tool: a "don't ask again" answered in
/// one tool is answered for the tier, which is the promise the prompt makes.
pub struct ApprovalBroker {
    policy: ApprovalPolicy,
    /// Tiers granted for the rest of the session by an
    /// [`AllowForSession`](ApprovalDecision::AllowForSession) answer.
    granted: Mutex<HashSet<RiskLevel>>,
    /// Where questions go when a client is driving. `None` means the terminal.
    sink: Option<Arc<dyn ApprovalSink>>,
}

impl Default for ApprovalBroker {
    fn default() -> Self {
        Self::new(ApprovalPolicy::default())
    }
}

impl ApprovalBroker {
    pub fn new(policy: ApprovalPolicy) -> Self {
        Self {
            policy,
            granted: Mutex::new(HashSet::new()),
            sink: None,
        }
    }

    /// A broker whose questions are answered by `sink` rather than by prompting
    /// on stdin.
    pub fn with_sink(policy: ApprovalPolicy, sink: Arc<dyn ApprovalSink>) -> Self {
        Self {
            policy,
            granted: Mutex::new(HashSet::new()),
            sink: Some(sink),
        }
    }

    pub fn policy(&self) -> ApprovalPolicy {
        self.policy
    }

    /// Decide whether `request` may proceed. `Ok(())` means yes; the error is
    /// the message the model sees, so it says what was refused and why.
    pub fn authorize(&self, request: &ApprovalRequest) -> Result<(), AgentError> {
        if request.risk == RiskLevel::ReadOnly {
            return Ok(());
        }
        if self.is_granted(request.risk) {
            return Ok(());
        }

        match self.policy.rule_for(request.risk) {
            ApprovalRule::Allow => Ok(()),
            ApprovalRule::Deny => Err(self.refused(request, "denied by policy")),
            ApprovalRule::Ask => self.ask(request),
        }
    }

    fn is_granted(&self, risk: RiskLevel) -> bool {
        self.granted
            .lock()
            .map(|g| g.contains(&risk))
            .unwrap_or(false)
    }

    fn grant(&self, risk: RiskLevel) {
        if risk.grantable_for_session() {
            if let Ok(mut g) = self.granted.lock() {
                g.insert(risk);
            }
        }
    }

    fn ask(&self, request: &ApprovalRequest) -> Result<(), AgentError> {
        // A driving client is authoritative when one is attached: it has a user
        // in front of it and we do not.
        let decision = match &self.sink {
            Some(sink) => sink.request(request)?,
            None if std::io::stdin().is_terminal() => self.prompt(request)?,
            // Nobody to ask. Refusing is the only honest answer, and naming the
            // knob is what keeps this from being the dead end the env var was.
            None => {
                return Err(self.refused(
                    request,
                    &format!(
                        "requires approval and there is no interactive terminal \
                         (set [agent.approvals] {} = \"allow\" to permit it non-interactively)",
                        request.risk.config_key()
                    ),
                ))
            }
        };

        match decision {
            ApprovalDecision::AllowOnce => Ok(()),
            ApprovalDecision::AllowForSession => {
                self.grant(request.risk);
                Ok(())
            }
            ApprovalDecision::Deny => Err(self.refused(request, "denied")),
        }
    }

    /// The terminal prompt. "Yes to all" is offered only for a tier that can
    /// actually be granted for the session, so the prompt never makes an offer
    /// the broker will not keep.
    fn prompt(&self, request: &ApprovalRequest) -> Result<ApprovalDecision, AgentError> {
        let all = request.risk.grantable_for_session();
        let mut err = std::io::stderr();
        let _ = write!(
            err,
            "\n\u{26a0}\u{fe0f}  Allow {} '{}'? [{}]\n  1) yes   {}3) no  > ",
            request.action,
            request.target,
            request.risk.label(),
            if all { "2) yes to all   " } else { "" },
        );
        let _ = err.flush();

        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            return Ok(ApprovalDecision::Deny);
        }

        Ok(match line.trim().to_lowercase().as_str() {
            "1" | "y" | "yes" => ApprovalDecision::AllowOnce,
            "2" | "a" | "all" if all => ApprovalDecision::AllowForSession,
            _ => ApprovalDecision::Deny,
        })
    }

    fn refused(&self, request: &ApprovalRequest, why: &str) -> AgentError {
        AgentError::InternalError(format!(
            "{} '{}' refused: {} [{}]",
            request.action,
            request.target,
            why,
            request.risk.label()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records what it was asked and answers with a fixed decision.
    struct ScriptedSink {
        answer: ApprovalDecision,
        seen: Mutex<Vec<(String, RiskLevel)>>,
    }

    impl ScriptedSink {
        fn new(answer: ApprovalDecision) -> Arc<Self> {
            Arc::new(Self {
                answer,
                seen: Mutex::new(Vec::new()),
            })
        }
        fn seen(&self) -> Vec<(String, RiskLevel)> {
            self.seen.lock().unwrap().clone()
        }
    }

    impl ApprovalSink for ScriptedSink {
        fn request(&self, request: &ApprovalRequest) -> Result<ApprovalDecision, AgentError> {
            self.seen
                .lock()
                .unwrap()
                .push((request.target.to_string(), request.risk));
            Ok(self.answer)
        }
    }

    fn req(risk: RiskLevel) -> ApprovalRequest<'static> {
        ApprovalRequest::new("do thing", "the-target", risk)
    }

    /// The acceptance criterion from issue #14 §4: a workspace write goes
    /// through with no terminal and no client, because the tier — not an env
    /// var — says so. This is what let `testsuite/runner.sh` stop exporting
    /// `GALLIUM_AUTO_APPROVE=1`.
    #[test]
    fn a_workspace_write_needs_no_terminal() {
        let broker = ApprovalBroker::default();

        assert!(broker.authorize(&req(RiskLevel::WorkspaceWrite)).is_ok());
    }

    /// The other half of that: the tiers the default policy asks about are
    /// refused rather than waved through when there is nobody to ask.
    #[test]
    fn the_asking_tiers_are_refused_with_nobody_to_ask() {
        let broker = ApprovalBroker::default();

        for risk in [RiskLevel::ExternalSideEffect, RiskLevel::Destructive] {
            let err = broker.authorize(&req(risk)).unwrap_err().to_string();
            assert!(err.contains(risk.label()), "{err}");
            assert!(err.contains(risk.config_key()), "{err}");
        }
    }

    /// Reading is never a question, whatever the policy says about everything
    /// else — there is no knob that can turn a read into a prompt.
    #[test]
    fn reading_is_never_asked_about() {
        let broker = ApprovalBroker::new(ApprovalPolicy {
            workspace_write: ApprovalRule::Deny,
            external_side_effect: ApprovalRule::Deny,
            destructive: ApprovalRule::Deny,
        });

        assert!(broker.authorize(&req(RiskLevel::ReadOnly)).is_ok());
    }

    #[test]
    fn a_client_is_asked_when_one_is_attached() {
        let sink = ScriptedSink::new(ApprovalDecision::AllowOnce);
        let broker = ApprovalBroker::with_sink(ApprovalPolicy::default(), sink.clone());

        assert!(broker.authorize(&req(RiskLevel::Destructive)).is_ok());
        assert_eq!(
            sink.seen(),
            vec![("the-target".to_string(), RiskLevel::Destructive)]
        );
    }

    /// A tier the policy allows outright never reaches the client: asking about
    /// something already decided is latency and noise.
    #[test]
    fn an_allowed_tier_is_not_put_to_the_client() {
        let sink = ScriptedSink::new(ApprovalDecision::Deny);
        let broker = ApprovalBroker::with_sink(ApprovalPolicy::default(), sink.clone());

        assert!(broker.authorize(&req(RiskLevel::WorkspaceWrite)).is_ok());
        assert!(sink.seen().is_empty());
    }

    /// "Yes to all" is answered once and then remembered, which is the whole
    /// point of the session grant.
    #[test]
    fn a_session_grant_stops_the_asking() {
        let sink = ScriptedSink::new(ApprovalDecision::AllowForSession);
        let broker = ApprovalBroker::with_sink(ApprovalPolicy::CAUTIOUS, sink.clone());

        for _ in 0..3 {
            assert!(broker.authorize(&req(RiskLevel::WorkspaceWrite)).is_ok());
        }

        assert_eq!(sink.seen().len(), 1, "asked more than once after a grant");
    }

    /// ...but not for the destructive tier. "Yes to all" there is a promise
    /// about commands nobody has read yet, so it is honored once and forgotten.
    #[test]
    fn destructive_is_confirmed_every_time_even_after_yes_to_all() {
        let sink = ScriptedSink::new(ApprovalDecision::AllowForSession);
        let broker = ApprovalBroker::with_sink(ApprovalPolicy::default(), sink.clone());

        for _ in 0..3 {
            assert!(broker.authorize(&req(RiskLevel::Destructive)).is_ok());
        }

        assert_eq!(sink.seen().len(), 3, "a destructive grant was remembered");
    }

    /// A grant is per tier, so approving writes does not quietly approve `bash`.
    #[test]
    fn a_grant_does_not_leak_across_tiers() {
        let sink = ScriptedSink::new(ApprovalDecision::AllowForSession);
        let broker = ApprovalBroker::with_sink(ApprovalPolicy::CAUTIOUS, sink.clone());

        broker.authorize(&req(RiskLevel::WorkspaceWrite)).unwrap();
        broker
            .authorize(&req(RiskLevel::ExternalSideEffect))
            .unwrap();

        assert_eq!(
            sink.seen().iter().map(|s| s.1).collect::<Vec<_>>(),
            vec![RiskLevel::WorkspaceWrite, RiskLevel::ExternalSideEffect]
        );
    }

    #[test]
    fn a_denied_request_says_what_it_refused() {
        let sink = ScriptedSink::new(ApprovalDecision::Deny);
        let broker = ApprovalBroker::with_sink(ApprovalPolicy::default(), sink);

        let err = broker
            .authorize(&ApprovalRequest::new(
                "run command",
                "rm -rf /",
                RiskLevel::Destructive,
            ))
            .unwrap_err()
            .to_string();

        assert!(err.contains("run command"), "{err}");
        assert!(err.contains("rm -rf /"), "{err}");
    }

    /// A write that escapes the workspace is not the tier its tool belongs to.
    #[test]
    fn leaving_the_workspace_is_a_different_tier() {
        assert_eq!(RiskLevel::for_write(true), RiskLevel::WorkspaceWrite);
        assert_eq!(RiskLevel::for_write(false), RiskLevel::Destructive);
    }

    #[test]
    fn a_deny_rule_refuses_without_asking() {
        let sink = ScriptedSink::new(ApprovalDecision::AllowOnce);
        let broker = ApprovalBroker::with_sink(
            ApprovalPolicy {
                workspace_write: ApprovalRule::Deny,
                ..ApprovalPolicy::default()
            },
            sink.clone(),
        );

        assert!(broker.authorize(&req(RiskLevel::WorkspaceWrite)).is_err());
        assert!(sink.seen().is_empty(), "asked about a decided refusal");
    }

    #[test]
    fn rules_parse_from_the_words_a_config_would_use() {
        assert_eq!(ApprovalRule::parse("allow"), Some(ApprovalRule::Allow));
        assert_eq!(ApprovalRule::parse("Ask"), Some(ApprovalRule::Ask));
        assert_eq!(ApprovalRule::parse("confirm"), Some(ApprovalRule::Ask));
        assert_eq!(ApprovalRule::parse(" deny "), Some(ApprovalRule::Deny));
        assert_eq!(ApprovalRule::parse("never"), Some(ApprovalRule::Deny));
        assert_eq!(ApprovalRule::parse("sometimes"), None);
    }
}

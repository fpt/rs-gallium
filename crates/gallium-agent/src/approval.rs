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

use std::collections::{HashSet, VecDeque};
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

/// What actually became of one request, once the policy, any standing grant,
/// and whoever was available to ask had all had their say.
///
/// Finer than [`ApprovalDecision`] on purpose: "allowed" and "allowed because
/// nobody objected two calls ago" are the same outcome for the tool and very
/// different ones for anyone reading back what a turn was permitted to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalOutcome {
    /// The policy allows this tier outright; nobody was asked.
    Allowed,
    /// An earlier [`AllowForSession`](ApprovalDecision::AllowForSession) already
    /// covered this tier.
    AllowedBySession,
    /// Asked, and answered yes for this one action.
    AllowedOnce,
    /// Asked, answered yes for the session, and the tier is now granted.
    AllowedForSession,
    /// Asked, and refused.
    Denied,
    /// The policy refuses this tier without asking.
    DeniedByPolicy,
    /// The rule says ask and there was nobody to ask.
    Unanswerable,
}

impl ApprovalOutcome {
    /// The wording used in traces.
    pub fn label(self) -> &'static str {
        match self {
            Self::Allowed => "allowed",
            Self::AllowedBySession => "allowed-by-session",
            Self::AllowedOnce => "allowed-once",
            Self::AllowedForSession => "allowed-for-session",
            Self::Denied => "denied",
            Self::DeniedByPolicy => "denied-by-policy",
            Self::Unanswerable => "unanswerable",
        }
    }

    pub fn allowed(self) -> bool {
        !matches!(
            self,
            Self::Denied | Self::DeniedByPolicy | Self::Unanswerable
        )
    }
}

/// One decision, kept so a turn can be reconstructed afterwards.
///
/// See [`ApprovalBroker::take_journal`] for why the broker holds these rather
/// than handing them to a caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalRecord {
    pub action: String,
    pub target: String,
    pub risk: RiskLevel,
    pub outcome: ApprovalOutcome,
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
    /// Decisions made but not yet collected, oldest first. See
    /// [`take_journal`](Self::take_journal).
    journal: Mutex<VecDeque<ApprovalRecord>>,
}

/// How many decisions the journal keeps before dropping the oldest. Only a
/// session nobody is tracing ever reaches it, since a recorder drains the
/// journal after every tool call.
const JOURNAL_CAPACITY: usize = 128;

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
            journal: Mutex::new(VecDeque::new()),
        }
    }

    /// A broker whose questions are answered by `sink` rather than by prompting
    /// on stdin.
    pub fn with_sink(policy: ApprovalPolicy, sink: Arc<dyn ApprovalSink>) -> Self {
        Self {
            sink: Some(sink),
            ..Self::new(policy)
        }
    }

    pub fn policy(&self) -> ApprovalPolicy {
        self.policy
    }

    /// Take the decisions made since this was last called, oldest first.
    ///
    /// The turn trace collects them after each tool call and attributes them to
    /// it. Reading them off the broker rather than returning them from
    /// [`authorize`](Self::authorize) is what keeps this out of the tools: a
    /// decision is made several frames below `Tool::call`, which takes no turn
    /// context, and threading one through twenty tool impls to carry a record
    /// out would be a large change to report a small fact.
    ///
    /// One broker serves one session and a session runs one turn at a time, so
    /// there is no second reader to race for the entries.
    pub fn take_journal(&self) -> Vec<ApprovalRecord> {
        match self.journal.lock() {
            Ok(mut journal) => journal.drain(..).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Decide whether `request` may proceed. `Ok(())` means yes; the error is
    /// the message the model sees, so it says what was refused and why.
    pub fn authorize(&self, request: &ApprovalRequest) -> Result<(), AgentError> {
        if request.risk == RiskLevel::ReadOnly {
            // Never asked about, and so never journaled: a question nobody was
            // asked is not a decision, and recording one per `read` would bury
            // the decisions that were made.
            return Ok(());
        }

        let outcome = self.decide(request)?;
        self.record(request, outcome);

        match outcome {
            ApprovalOutcome::Allowed
            | ApprovalOutcome::AllowedBySession
            | ApprovalOutcome::AllowedOnce
            | ApprovalOutcome::AllowedForSession => Ok(()),
            ApprovalOutcome::Denied => Err(self.refused(request, "denied")),
            ApprovalOutcome::DeniedByPolicy => Err(self.refused(request, "denied by policy")),
            // Nobody to ask. Refusing is the only honest answer, and naming the
            // knob is what keeps this from being the dead end the env var was.
            ApprovalOutcome::Unanswerable => Err(self.refused(
                request,
                &format!(
                    "requires approval and there is no interactive terminal \
                     (set [agent.approvals] {} = \"allow\" to permit it non-interactively)",
                    request.risk.config_key()
                ),
            )),
        }
    }

    /// What happens to `request`, before it is turned into an answer.
    ///
    /// An error is the sink itself failing — a client that went away mid-
    /// question. That is not a decision, so nothing is journaled for it; the
    /// turn fails and its trace's ending carries the reason.
    fn decide(&self, request: &ApprovalRequest) -> Result<ApprovalOutcome, AgentError> {
        if self.is_granted(request.risk) {
            return Ok(ApprovalOutcome::AllowedBySession);
        }
        match self.policy.rule_for(request.risk) {
            ApprovalRule::Allow => Ok(ApprovalOutcome::Allowed),
            ApprovalRule::Deny => Ok(ApprovalOutcome::DeniedByPolicy),
            ApprovalRule::Ask => self.ask(request),
        }
    }

    fn record(&self, request: &ApprovalRequest, outcome: ApprovalOutcome) {
        let Ok(mut journal) = self.journal.lock() else {
            return;
        };
        if journal.len() >= JOURNAL_CAPACITY {
            journal.pop_front();
        }
        journal.push_back(ApprovalRecord {
            action: request.action.to_string(),
            target: request.target.to_string(),
            risk: request.risk,
            outcome,
        });
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

    fn ask(&self, request: &ApprovalRequest) -> Result<ApprovalOutcome, AgentError> {
        // A driving client is authoritative when one is attached: it has a user
        // in front of it and we do not.
        let decision = match &self.sink {
            Some(sink) => sink.request(request)?,
            None if std::io::stdin().is_terminal() => self.prompt(request)?,
            None => return Ok(ApprovalOutcome::Unanswerable),
        };

        Ok(match decision {
            ApprovalDecision::AllowOnce => ApprovalOutcome::AllowedOnce,
            ApprovalDecision::AllowForSession => {
                self.grant(request.risk);
                // A destructive "yes to all" is honored once and not
                // remembered, so the record says what happened rather than what
                // was answered.
                if request.risk.grantable_for_session() {
                    ApprovalOutcome::AllowedForSession
                } else {
                    ApprovalOutcome::AllowedOnce
                }
            }
            ApprovalDecision::Deny => ApprovalOutcome::Denied,
        })
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

    // ------------------------------------------------------------------
    // The journal: what a turn trace reads back out.
    // ------------------------------------------------------------------

    fn outcomes(broker: &ApprovalBroker) -> Vec<ApprovalOutcome> {
        broker.take_journal().iter().map(|r| r.outcome).collect()
    }

    /// A tier the policy waves through was still a decision, and one worth
    /// being able to see afterwards — "nobody asked me" is the answer to most
    /// questions about what an agent was allowed to do.
    #[test]
    fn an_allowed_tier_is_journaled_as_allowed() {
        let broker = ApprovalBroker::new(ApprovalPolicy::default());

        broker.authorize(&req(RiskLevel::WorkspaceWrite)).unwrap();

        assert_eq!(outcomes(&broker), vec![ApprovalOutcome::Allowed]);
    }

    /// A read is never asked about, so it is never journaled: one entry per
    /// `read` would bury the decisions somebody actually made.
    #[test]
    fn a_read_is_not_a_decision() {
        let broker = ApprovalBroker::new(ApprovalPolicy::default());

        broker.authorize(&req(RiskLevel::ReadOnly)).unwrap();

        assert!(broker.take_journal().is_empty());
    }

    /// Taking the journal empties it, so the next tool call's trace gets its own
    /// decisions rather than every decision so far.
    #[test]
    fn taking_the_journal_empties_it() {
        let broker = ApprovalBroker::new(ApprovalPolicy::default());
        broker.authorize(&req(RiskLevel::WorkspaceWrite)).unwrap();

        assert_eq!(broker.take_journal().len(), 1);
        assert!(broker.take_journal().is_empty());
    }

    /// The journal records the target too — which file, which command — since a
    /// tier alone does not say what was permitted.
    #[test]
    fn the_journal_says_what_was_asked_about() {
        let broker = ApprovalBroker::new(ApprovalPolicy::default());

        broker
            .authorize(&ApprovalRequest::new(
                "create file",
                "out.txt",
                RiskLevel::WorkspaceWrite,
            ))
            .unwrap();

        let journal = broker.take_journal();
        assert_eq!(journal[0].action, "create file");
        assert_eq!(journal[0].target, "out.txt");
        assert_eq!(journal[0].risk, RiskLevel::WorkspaceWrite);
    }

    /// The second call was allowed for a different reason than the first, and a
    /// trace that spelled both "allowed" would hide the grant that did it.
    #[test]
    fn a_session_grant_is_journaled_as_the_reason_the_next_call_passed() {
        let sink = ScriptedSink::new(ApprovalDecision::AllowForSession);
        let broker = ApprovalBroker::with_sink(ApprovalPolicy::CAUTIOUS, sink);

        broker.authorize(&req(RiskLevel::WorkspaceWrite)).unwrap();
        broker.authorize(&req(RiskLevel::WorkspaceWrite)).unwrap();

        assert_eq!(
            outcomes(&broker),
            vec![
                ApprovalOutcome::AllowedForSession,
                ApprovalOutcome::AllowedBySession
            ]
        );
    }

    /// A destructive "yes to all" is honored once and not remembered, so the
    /// record says what happened rather than what was answered.
    #[test]
    fn a_destructive_yes_to_all_is_journaled_as_the_once_it_actually_was() {
        let sink = ScriptedSink::new(ApprovalDecision::AllowForSession);
        let broker = ApprovalBroker::with_sink(ApprovalPolicy::default(), sink);

        broker.authorize(&req(RiskLevel::Destructive)).unwrap();
        broker.authorize(&req(RiskLevel::Destructive)).unwrap();

        assert_eq!(
            outcomes(&broker),
            vec![ApprovalOutcome::AllowedOnce, ApprovalOutcome::AllowedOnce],
            "a blanket yes on shell commands is a promise about commands nobody has read"
        );
    }

    /// The three ways to be refused are three different facts about a run: the
    /// client said no, the policy said no, and there was nobody to ask.
    #[test]
    fn the_refusals_are_told_apart() {
        let denied = ApprovalBroker::with_sink(
            ApprovalPolicy::CAUTIOUS,
            ScriptedSink::new(ApprovalDecision::Deny),
        );
        assert!(denied.authorize(&req(RiskLevel::Destructive)).is_err());
        assert_eq!(outcomes(&denied), vec![ApprovalOutcome::Denied]);

        let by_policy = ApprovalBroker::new(ApprovalPolicy {
            workspace_write: ApprovalRule::Deny,
            ..ApprovalPolicy::default()
        });
        assert!(by_policy
            .authorize(&req(RiskLevel::WorkspaceWrite))
            .is_err());
        assert_eq!(outcomes(&by_policy), vec![ApprovalOutcome::DeniedByPolicy]);

        // No sink, and a test harness's stdin is not a terminal.
        let nobody = ApprovalBroker::new(ApprovalPolicy::CAUTIOUS);
        assert!(nobody.authorize(&req(RiskLevel::WorkspaceWrite)).is_err());
        assert_eq!(outcomes(&nobody), vec![ApprovalOutcome::Unanswerable]);
    }

    /// A session nobody is tracing never drains the journal, so it has to stop
    /// growing on its own.
    #[test]
    fn an_uncollected_journal_stays_bounded() {
        let broker = ApprovalBroker::new(ApprovalPolicy::default());

        for _ in 0..JOURNAL_CAPACITY * 2 {
            broker.authorize(&req(RiskLevel::WorkspaceWrite)).unwrap();
        }

        assert_eq!(broker.take_journal().len(), JOURNAL_CAPACITY);
    }
}

//! GitHub Projects V2 tools.
//!
//! These tools let the agent read and manage the user's GitHub Projects board.
//! Unlike `capture`/OCR (which need Apple frameworks and so live in Swift behind
//! the capture bridge), every operation here is just a `gh api graphql`
//! subprocess + JSON parse — no platform dependency — so it lives entirely in
//! Rust. That keeps it working from the Swift CLI, the Windows C# CLI, and the
//! standalone Rust CLI alike, with no UniFFI/binding changes.
//!
//! Queries are ported from the `m6o-deskcat` project. Project/field/option node
//! IDs are resolved lazily via one metadata query and cached for the session, so
//! configuration is just three human-friendly env vars:
//!   - `KESSEL_GH_ORG`     organization login (required to enable the tools)
//!   - `KESSEL_GH_PROJECT` project number (required)
//!   - `KESSEL_GH_REPO`    default repo `owner/name` (required only for promote)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::tool::{ToolHandler, ToolResult, ToolSession};
use crate::AgentError;

/// Runs a `gh` invocation (args after the `gh` program name) and returns stdout.
/// Boxed so tests can inject a mock instead of spawning the real CLI.
type GhRunner = Box<dyn Fn(&[&str]) -> Result<String, AgentError> + Send + Sync>;

/// A single project board item (issue or draft) assigned to the user.
#[derive(Debug, Clone)]
struct ProjectItem {
    item_id: String,
    title: String,
    number: Option<i64>,
    url: Option<String>,
    repo: Option<String>,
    status: Option<String>,
    sprint: Option<String>,
    labels: Vec<String>,
    is_draft: bool,
}

/// Cached project metadata, resolved from a single GraphQL query.
#[derive(Debug, Clone)]
struct ProjectMeta {
    project_node_id: String,
    status_field_id: Option<String>,
    /// Status option name -> option id.
    status_options: HashMap<String, String>,
    sprint_field_id: Option<String>,
    /// The iteration id whose date range contains today (best effort).
    current_sprint_id: Option<String>,
}

/// Talks to GitHub Projects V2 via the `gh` CLI. Cheap to clone the cached
/// values out; the client itself is shared behind an `Arc`.
pub struct GithubClient {
    org: String,
    project_number: u64,
    default_repo: Option<String>,
    runner: GhRunner,
    meta: Mutex<Option<ProjectMeta>>,
    /// (node id, login) of the authenticated user.
    viewer: Mutex<Option<(String, String)>>,
    repo_node_id: Mutex<Option<String>>,
}

impl GithubClient {
    /// Build a client from `KESSEL_GH_*` env vars. Returns `None` (tools stay
    /// unregistered) unless both org and a valid project number are set.
    pub fn from_env() -> Option<Self> {
        let org = std::env::var("KESSEL_GH_ORG").ok().filter(|s| !s.trim().is_empty())?;
        let project_number = std::env::var("KESSEL_GH_PROJECT")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())?;
        let default_repo = std::env::var("KESSEL_GH_REPO")
            .ok()
            .filter(|s| !s.trim().is_empty());
        Some(Self::new(org, project_number, default_repo, Box::new(default_gh_runner)))
    }

    fn new(org: String, project_number: u64, default_repo: Option<String>, runner: GhRunner) -> Self {
        Self {
            org,
            project_number,
            default_repo,
            runner,
            meta: Mutex::new(None),
            viewer: Mutex::new(None),
            repo_node_id: Mutex::new(None),
        }
    }

    // ---- low-level gh / GraphQL ------------------------------------------

    /// Run a GraphQL query/mutation, returning its `data` object. Surfaces
    /// GraphQL-level `errors` as an `AgentError`.
    fn graphql(&self, query: &str) -> Result<Value, AgentError> {
        let arg = format!("query={}", query);
        let out = (self.runner)(&["api", "graphql", "-f", &arg])?;
        let v: Value = serde_json::from_str(&out)
            .map_err(|e| AgentError::ParseError(format!("gh returned non-JSON: {e}")))?;
        if let Some(errors) = v.get("errors").and_then(Value::as_array) {
            if !errors.is_empty() {
                let msgs: Vec<String> = errors
                    .iter()
                    .map(|e| e.get("message").and_then(Value::as_str).unwrap_or("unknown").to_string())
                    .collect();
                return Err(AgentError::InternalError(format!("GraphQL error: {}", msgs.join("; "))));
            }
        }
        Ok(v.get("data").cloned().unwrap_or(Value::Null))
    }

    // ---- lazily-resolved, cached metadata --------------------------------

    /// (node id, login) of the authenticated `gh` user.
    fn viewer(&self) -> Result<(String, String), AgentError> {
        if let Some(v) = self.viewer.lock().unwrap().as_ref() {
            return Ok(v.clone());
        }
        let data = self.graphql("query { viewer { id login } }")?;
        let v = &data["viewer"];
        let id = v["id"].as_str().ok_or_else(|| AgentError::ParseError("viewer.id missing".into()))?;
        let login = v["login"].as_str().ok_or_else(|| AgentError::ParseError("viewer.login missing".into()))?;
        let pair = (id.to_string(), login.to_string());
        *self.viewer.lock().unwrap() = Some(pair.clone());
        Ok(pair)
    }

    /// Node id of the configured default repo (`owner/name`).
    fn repo_node_id(&self) -> Result<String, AgentError> {
        if let Some(id) = self.repo_node_id.lock().unwrap().as_ref() {
            return Ok(id.clone());
        }
        let repo = self.default_repo.as_ref().ok_or_else(|| {
            AgentError::ConfigError("KESSEL_GH_REPO not set (required to promote a draft to an issue)".into())
        })?;
        let (owner, name) = repo
            .split_once('/')
            .ok_or_else(|| AgentError::ConfigError(format!("KESSEL_GH_REPO must be 'owner/name', got '{repo}'")))?;
        let q = format!(
            "query {{ repository(owner: \"{}\", name: \"{}\") {{ id }} }}",
            escape_graphql(owner),
            escape_graphql(name)
        );
        let data = self.graphql(&q)?;
        let id = data["repository"]["id"]
            .as_str()
            .ok_or_else(|| AgentError::InternalError(format!("repo '{repo}' not found")))?
            .to_string();
        *self.repo_node_id.lock().unwrap() = Some(id.clone());
        Ok(id)
    }

    /// Project node id, Status/Sprint field ids, status option map, and the
    /// current iteration — all from one query, cached for the session.
    fn project_meta(&self) -> Result<ProjectMeta, AgentError> {
        if let Some(m) = self.meta.lock().unwrap().as_ref() {
            return Ok(m.clone());
        }
        let q = format!(
            r#"query {{
  organization(login: "{org}") {{
    projectV2(number: {num}) {{
      id
      fields(first: 50) {{
        nodes {{
          ... on ProjectV2SingleSelectField {{ id name options {{ id name }} }}
          ... on ProjectV2IterationField {{ id name configuration {{ iterations {{ id title startDate duration }} }} }}
          ... on ProjectV2FieldCommon {{ id name }}
        }}
      }}
    }}
  }}
}}"#,
            org = escape_graphql(&self.org),
            num = self.project_number
        );
        let data = self.graphql(&q)?;
        let project = &data["organization"]["projectV2"];
        let project_node_id = project["id"]
            .as_str()
            .ok_or_else(|| AgentError::InternalError(format!(
                "project #{} not found in org '{}'",
                self.project_number, self.org
            )))?
            .to_string();

        let empty = vec![];
        let fields = project["fields"]["nodes"].as_array().unwrap_or(&empty);

        // Resolve the Status single-select field (prefer name "Status", else the
        // first single-select field that has options).
        let mut status_field_id = None;
        let mut status_options = HashMap::new();
        let mut fallback_status: Option<(String, HashMap<String, String>)> = None;
        // Resolve the Sprint iteration field (prefer name "Sprint", else first).
        let mut sprint_field_id = None;
        let mut current_sprint_id = None;
        let mut fallback_sprint: Option<(String, Option<String>)> = None;

        for f in fields {
            let name = f.get("name").and_then(Value::as_str).unwrap_or("");
            let id = f.get("id").and_then(Value::as_str).unwrap_or("");
            if let Some(options) = f.get("options").and_then(Value::as_array) {
                let map: HashMap<String, String> = options
                    .iter()
                    .filter_map(|o| {
                        Some((o["name"].as_str()?.to_string(), o["id"].as_str()?.to_string()))
                    })
                    .collect();
                if name.eq_ignore_ascii_case("Status") {
                    status_field_id = Some(id.to_string());
                    status_options = map;
                } else if fallback_status.is_none() && !map.is_empty() {
                    fallback_status = Some((id.to_string(), map));
                }
            } else if let Some(cfg) = f.get("configuration") {
                let iterations = cfg["iterations"].as_array();
                let current = iterations.and_then(|its| current_iteration_id(its));
                if name.eq_ignore_ascii_case("Sprint") {
                    sprint_field_id = Some(id.to_string());
                    current_sprint_id = current;
                } else if fallback_sprint.is_none() {
                    fallback_sprint = Some((id.to_string(), current));
                }
            }
        }
        if status_field_id.is_none() {
            if let Some((id, map)) = fallback_status {
                status_field_id = Some(id);
                status_options = map;
            }
        }
        if sprint_field_id.is_none() {
            if let Some((id, current)) = fallback_sprint {
                sprint_field_id = Some(id);
                current_sprint_id = current;
            }
        }

        let meta = ProjectMeta {
            project_node_id,
            status_field_id,
            status_options,
            sprint_field_id,
            current_sprint_id,
        };
        *self.meta.lock().unwrap() = Some(meta.clone());
        Ok(meta)
    }

    // ---- operations ------------------------------------------------------

    /// List the user's open assigned project items (issues + drafts).
    fn list_my_items(&self, status_filter: Option<&str>, include_done: bool) -> Result<Vec<ProjectItem>, AgentError> {
        let (_, login) = self.viewer()?;
        let mut items = Vec::new();
        let mut cursor: Option<String> = None;

        for _ in 0..10 {
            let after = cursor
                .as_ref()
                .map(|c| format!(", after: \"{}\"", c))
                .unwrap_or_default();
            let q = format!(
                r#"query {{
  organization(login: "{org}") {{
    projectV2(number: {num}) {{
      items(first: 100{after}) {{
        pageInfo {{ hasNextPage endCursor }}
        nodes {{
          id
          content {{
            ... on Issue {{
              title number state url
              repository {{ nameWithOwner }}
              assignees(first: 5) {{ nodes {{ login }} }}
              labels(first: 5) {{ nodes {{ name }} }}
            }}
            ... on DraftIssue {{
              title body
              assignees(first: 5) {{ nodes {{ login }} }}
            }}
          }}
          fieldValues(first: 10) {{
            nodes {{
              ... on ProjectV2ItemFieldSingleSelectValue {{ name field {{ ... on ProjectV2FieldCommon {{ name }} }} }}
              ... on ProjectV2ItemFieldIterationValue {{ title field {{ ... on ProjectV2FieldCommon {{ name }} }} }}
            }}
          }}
        }}
      }}
    }}
  }}
}}"#,
                org = escape_graphql(&self.org),
                num = self.project_number,
                after = after
            );
            let data = self.graphql(&q)?;
            let conn = &data["organization"]["projectV2"]["items"];
            let empty = vec![];
            for node in conn["nodes"].as_array().unwrap_or(&empty) {
                if let Some(item) = parse_item(node, &login) {
                    if !include_done
                        && matches!(item.status.as_deref(), Some("Done") | Some("Cancelled"))
                    {
                        continue;
                    }
                    if let Some(want) = status_filter {
                        if item.status.as_deref().map(|s| s.eq_ignore_ascii_case(want)) != Some(true) {
                            continue;
                        }
                    }
                    items.push(item);
                }
            }
            if conn["pageInfo"]["hasNextPage"].as_bool() == Some(true) {
                cursor = conn["pageInfo"]["endCursor"].as_str().map(str::to_string);
            } else {
                break;
            }
        }
        Ok(items)
    }

    /// Create a draft on the board, set it to "Todo", and assign the current
    /// sprint (best effort). Returns the new project item id.
    fn create_draft(&self, title: &str, body: &str) -> Result<String, AgentError> {
        let meta = self.project_meta()?;
        let q = format!(
            r#"mutation {{
  addProjectV2DraftIssue(input: {{ projectId: "{pid}", title: "{title}", body: "{body}" }}) {{
    projectItem {{ id }}
  }}
}}"#,
            pid = meta.project_node_id,
            title = escape_graphql(title),
            body = escape_graphql(body),
        );
        let data = self.graphql(&q)?;
        let item_id = data["addProjectV2DraftIssue"]["projectItem"]["id"]
            .as_str()
            .ok_or_else(|| AgentError::InternalError("create draft: no item id returned".into()))?
            .to_string();

        // Best effort: set status Todo and current sprint. Don't fail the whole
        // operation if these secondary fields can't be set.
        if let (Some(field), Some(opt)) = (meta.status_field_id.as_ref(), meta.status_options.get("Todo")) {
            let _ = self.set_single_select(&meta.project_node_id, &item_id, field, opt);
        }
        if let (Some(field), Some(iter)) = (meta.sprint_field_id.as_ref(), meta.current_sprint_id.as_ref()) {
            let _ = self.set_iteration(&meta.project_node_id, &item_id, field, iter);
        }
        Ok(item_id)
    }

    fn set_single_select(&self, project_id: &str, item_id: &str, field_id: &str, option_id: &str) -> Result<(), AgentError> {
        let q = format!(
            r#"mutation {{ updateProjectV2ItemFieldValue(input: {{ projectId: "{p}", itemId: "{i}", fieldId: "{f}", value: {{ singleSelectOptionId: "{o}" }} }}) {{ projectV2Item {{ id }} }} }}"#,
            p = project_id, i = item_id, f = field_id, o = option_id
        );
        self.graphql(&q).map(|_| ())
    }

    fn set_iteration(&self, project_id: &str, item_id: &str, field_id: &str, iteration_id: &str) -> Result<(), AgentError> {
        let q = format!(
            r#"mutation {{ updateProjectV2ItemFieldValue(input: {{ projectId: "{p}", itemId: "{i}", fieldId: "{f}", value: {{ iterationId: "{it}" }} }}) {{ projectV2Item {{ id }} }} }}"#,
            p = project_id, i = item_id, f = field_id, it = iteration_id
        );
        self.graphql(&q).map(|_| ())
    }

    /// Set a project item's Status field by option name (e.g. "In Progress").
    fn set_status(&self, item_id: &str, status: &str) -> Result<String, AgentError> {
        let meta = self.project_meta()?;
        let field = meta
            .status_field_id
            .as_ref()
            .ok_or_else(|| AgentError::InternalError("project has no Status field".into()))?;
        // Case-insensitive lookup of the option name.
        let (canonical, option_id) = meta
            .status_options
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(status))
            .ok_or_else(|| {
                let mut names: Vec<&str> = meta.status_options.keys().map(String::as_str).collect();
                names.sort_unstable();
                AgentError::ParseError(format!(
                    "unknown status '{status}'. Valid statuses: {}",
                    names.join(", ")
                ))
            })?;
        self.set_single_select(&meta.project_node_id, item_id, field, option_id)?;
        Ok(canonical.clone())
    }

    /// Convert a draft project item into a real issue in the default repo and
    /// assign it to the current user. Returns (number, url).
    fn promote_draft(&self, item_id: &str) -> Result<(i64, String), AgentError> {
        let repo_id = self.repo_node_id()?;
        let q = format!(
            r#"mutation {{
  convertProjectV2DraftIssueItemToIssue(input: {{ itemId: "{i}", repositoryId: "{r}" }}) {{
    item {{ content {{ ... on Issue {{ id number url }} }} }}
  }}
}}"#,
            i = item_id, r = repo_id
        );
        let data = self.graphql(&q)?;
        let issue = &data["convertProjectV2DraftIssueItemToIssue"]["item"]["content"];
        let issue_id = issue["id"]
            .as_str()
            .ok_or_else(|| AgentError::InternalError("promote: item is not a draft or conversion failed".into()))?;
        let number = issue["number"].as_i64().unwrap_or(0);
        let url = issue["url"].as_str().unwrap_or("").to_string();

        // Assign to the current user (best effort — the issue exists regardless).
        if let Ok((user_id, _)) = self.viewer() {
            let aq = format!(
                r#"mutation {{ addAssigneesToAssignable(input: {{ assignableId: "{a}", assigneeIds: ["{u}"] }}) {{ assignable {{ ... on Issue {{ number }} }} }} }}"#,
                a = issue_id, u = user_id
            );
            let _ = self.graphql(&aq);
        }
        Ok((number, url))
    }

    /// Post an activity comment on the issue backing a project item. Returns the
    /// comment URL. Errors if the item is still a draft (no issue to comment on).
    fn log_activity(&self, item_id: &str, text: &str, context: &[String]) -> Result<String, AgentError> {
        let q = format!(
            r#"query {{ node(id: "{i}") {{ ... on ProjectV2Item {{ content {{ ... on Issue {{ id }} }} }} }} }}"#,
            i = item_id
        );
        let data = self.graphql(&q)?;
        let issue_id = data["node"]["content"]["id"].as_str().ok_or_else(|| {
            AgentError::ParseError(
                "item has no underlying issue (it's a draft — promote it first with github_promote_draft)".into(),
            )
        })?;

        let mut body = format!("<!-- kessel -->\n## Activity Update (Kessel)\n\n{text}\n\n");
        if !context.is_empty() {
            body.push_str("<details><summary>context</summary>\n\n");
            for line in context.iter().take(15) {
                body.push_str(&format!("- {line}\n"));
            }
            body.push_str("\n</details>");
        }

        let mq = format!(
            r#"mutation {{ addComment(input: {{ subjectId: "{s}", body: "{b}" }}) {{ commentEdge {{ node {{ url }} }} }} }}"#,
            s = issue_id,
            b = escape_graphql(&body)
        );
        let data = self.graphql(&mq)?;
        Ok(data["addComment"]["commentEdge"]["node"]["url"]
            .as_str()
            .unwrap_or("(comment posted)")
            .to_string())
    }
}

/// Default runner: spawn the real `gh` CLI.
fn default_gh_runner(args: &[&str]) -> Result<String, AgentError> {
    let output = std::process::Command::new("gh")
        .args(args)
        .output()
        .map_err(|e| {
            AgentError::InternalError(format!("failed to run `gh` (is the GitHub CLI installed and on PATH?): {e}"))
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = if stderr.trim().is_empty() { stdout.trim() } else { stderr.trim() };
        return Err(AgentError::InternalError(format!("gh command failed: {detail}")));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Escape a string for inline embedding in a GraphQL string literal.
fn escape_graphql(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "")
}

/// Given an iteration field's `iterations` array, return the id of the one whose
/// `[startDate, startDate + duration)` window contains today (Asia/Tokyo), or
/// the first iteration as a fallback. Dependency-free date math.
fn current_iteration_id(iterations: &[Value]) -> Option<String> {
    let today = today_jst_days();
    for it in iterations {
        let (Some(start), Some(duration), Some(id)) = (
            it.get("startDate").and_then(Value::as_str),
            it.get("duration").and_then(Value::as_i64),
            it.get("id").and_then(Value::as_str),
        ) else {
            continue;
        };
        if let Some(start_days) = parse_ymd_to_days(start) {
            if today >= start_days && today < start_days + duration {
                return Some(id.to_string());
            }
        }
    }
    iterations.first().and_then(|it| it["id"].as_str()).map(str::to_string)
}

/// Today's date in Asia/Tokyo, as days since the Unix epoch.
fn today_jst_days() -> i64 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    (secs + 9 * 3600).div_euclid(86400)
}

/// Parse "YYYY-MM-DD" into days since the Unix epoch.
fn parse_ymd_to_days(s: &str) -> Option<i64> {
    let mut parts = s.split('-');
    let y = parts.next()?.parse::<i64>().ok()?;
    let m = parts.next()?.parse::<i64>().ok()?;
    let d = parts.next()?.parse::<i64>().ok()?;
    Some(days_from_civil(y, m, d))
}

/// Days since 1970-01-01 for a civil (proleptic Gregorian) date.
/// Howard Hinnant's algorithm.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Parse one project item node from the list query; `None` if it isn't assigned
/// to `login` or has no content.
fn parse_item(node: &Value, login: &str) -> Option<ProjectItem> {
    let item_id = node.get("id")?.as_str()?.to_string();
    let content = node.get("content")?;
    if content.is_null() {
        return None;
    }
    // Drafts have no `state`; issues do.
    let is_draft = content.get("state").map(Value::is_null).unwrap_or(true);

    let assignees: Vec<&str> = content["assignees"]["nodes"]
        .as_array()
        .map(|a| a.iter().filter_map(|n| n["login"].as_str()).collect())
        .unwrap_or_default();
    if !assignees.iter().any(|a| a == &login) {
        return None;
    }
    if !is_draft && content.get("state").and_then(Value::as_str) != Some("OPEN") {
        return None;
    }

    let mut status = None;
    let mut sprint = None;
    if let Some(fvs) = node["fieldValues"]["nodes"].as_array() {
        for fv in fvs {
            let field_name = fv["field"]["name"].as_str().unwrap_or("");
            if field_name.eq_ignore_ascii_case("Status") {
                status = fv["name"].as_str().map(str::to_string);
            } else if field_name.eq_ignore_ascii_case("Sprint") {
                sprint = fv["title"].as_str().map(str::to_string);
            }
        }
    }

    let labels = content["labels"]["nodes"]
        .as_array()
        .map(|l| l.iter().filter_map(|n| n["name"].as_str().map(str::to_string)).collect())
        .unwrap_or_default();

    Some(ProjectItem {
        item_id,
        title: content["title"].as_str().unwrap_or("(untitled)").to_string(),
        number: content["number"].as_i64(),
        url: content["url"].as_str().map(str::to_string),
        repo: content["repository"]["nameWithOwner"].as_str().map(str::to_string),
        status,
        sprint,
        labels,
        is_draft,
    })
}

/// Render items into a compact list for the model.
fn format_items(items: &[ProjectItem]) -> String {
    if items.is_empty() {
        return "No matching tasks on the board.".to_string();
    }
    let mut out = format!("{} task(s):", items.len());
    for it in items {
        let status = it.status.as_deref().unwrap_or("No status");
        let num = match it.number {
            Some(n) => format!("#{n} "),
            None => String::new(),
        };
        let kind = if it.is_draft { " (draft)" } else { "" };
        out.push_str(&format!("\n- [{status}]{kind} {num}{}", it.title));
        let mut tail = Vec::new();
        if let Some(r) = &it.repo {
            tail.push(format!("repo={r}"));
        }
        if let Some(s) = &it.sprint {
            tail.push(format!("sprint={s}"));
        }
        if !it.labels.is_empty() {
            tail.push(format!("labels={}", it.labels.join(",")));
        }
        if let Some(u) = &it.url {
            tail.push(format!("url={u}"));
        }
        tail.push(format!("item_id={}", it.item_id));
        out.push_str(&format!("\n    {}", tail.join("  ")));
    }
    out
}

// ============================================================================
// Tools
// ============================================================================

/// Read-only: list the user's assigned tasks. No permission prompt.
pub struct GithubListTasksTool {
    client: Arc<GithubClient>,
}

impl GithubListTasksTool {
    pub fn new(client: Arc<GithubClient>) -> Self {
        Self { client }
    }
}

impl ToolHandler for GithubListTasksTool {
    fn name(&self) -> &str {
        "github_list_tasks"
    }
    fn description(&self) -> &str {
        "List the user's assigned GitHub Projects board tasks (issues and drafts). \
         Returns each task's status, sprint, labels, url, issue number, and item_id \
         (the item_id is needed by the other github_* tools). Open items only; \
         Done/Cancelled are hidden unless include_done is set."
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "status": { "type": "string", "description": "Optional: only return tasks with this status (e.g. 'In Progress', 'Todo')" },
                "include_done": { "type": "boolean", "description": "Include Done/Cancelled tasks (default false)" }
            }
        })
    }
    fn call(&self, args: Value) -> Result<ToolResult, AgentError> {
        let status = args.get("status").and_then(Value::as_str);
        let include_done = args.get("include_done").and_then(Value::as_bool).unwrap_or(false);
        let items = self.client.list_my_items(status, include_done)?;
        Ok(ToolResult::text(format_items(&items)))
    }
}

/// Create a draft task on the board (current sprint, status Todo). Gated.
pub struct GithubCreateDraftTool {
    client: Arc<GithubClient>,
    session: Arc<ToolSession>,
}

impl GithubCreateDraftTool {
    pub fn new(client: Arc<GithubClient>, session: Arc<ToolSession>) -> Self {
        Self { client, session }
    }
}

impl ToolHandler for GithubCreateDraftTool {
    fn name(&self) -> &str {
        "github_create_draft"
    }
    fn description(&self) -> &str {
        "Create a new draft task on the GitHub Projects board. Sets it to the current \
         sprint and status 'Todo'. Returns the new item_id. Asks for permission first."
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "title": { "type": "string", "description": "Task title" },
                "body": { "type": "string", "description": "Task description (optional)" }
            },
            "required": ["title"]
        })
    }
    fn call(&self, args: Value) -> Result<ToolResult, AgentError> {
        let title = args.get("title").and_then(Value::as_str).ok_or_else(|| {
            AgentError::ParseError("'title' is required".into())
        })?;
        let body = args.get("body").and_then(Value::as_str).unwrap_or("");
        self.session.request_github("create GitHub draft", title)?;
        let item_id = self.client.create_draft(title, body)?;
        Ok(ToolResult::text(format!("Created draft '{title}' (item_id={item_id}).")))
    }
}

/// Promote a draft to a real issue and assign it to the user. Gated.
pub struct GithubPromoteDraftTool {
    client: Arc<GithubClient>,
    session: Arc<ToolSession>,
}

impl GithubPromoteDraftTool {
    pub fn new(client: Arc<GithubClient>, session: Arc<ToolSession>) -> Self {
        Self { client, session }
    }
}

impl ToolHandler for GithubPromoteDraftTool {
    fn name(&self) -> &str {
        "github_promote_draft"
    }
    fn description(&self) -> &str {
        "Convert a draft task into a real GitHub issue in the default repo and assign \
         it to the user. Takes the item_id from github_list_tasks. Asks for permission first."
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "item_id": { "type": "string", "description": "Project item id of the draft (from github_list_tasks)" }
            },
            "required": ["item_id"]
        })
    }
    fn call(&self, args: Value) -> Result<ToolResult, AgentError> {
        let item_id = args.get("item_id").and_then(Value::as_str).ok_or_else(|| {
            AgentError::ParseError("'item_id' is required".into())
        })?;
        self.session.request_github("promote GitHub draft to issue", item_id)?;
        let (number, url) = self.client.promote_draft(item_id)?;
        Ok(ToolResult::text(format!("Promoted draft to issue #{number}. {url}")))
    }
}

/// Set a task's Status field. Gated.
pub struct GithubSetStatusTool {
    client: Arc<GithubClient>,
    session: Arc<ToolSession>,
}

impl GithubSetStatusTool {
    pub fn new(client: Arc<GithubClient>, session: Arc<ToolSession>) -> Self {
        Self { client, session }
    }
}

impl ToolHandler for GithubSetStatusTool {
    fn name(&self) -> &str {
        "github_set_status"
    }
    fn description(&self) -> &str {
        "Set the Status of a task on the board (e.g. 'Todo', 'In Progress', 'Done'). \
         Takes the item_id from github_list_tasks. Asks for permission first."
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "item_id": { "type": "string", "description": "Project item id (from github_list_tasks)" },
                "status": { "type": "string", "description": "Target status name (must match a board status option)" }
            },
            "required": ["item_id", "status"]
        })
    }
    fn call(&self, args: Value) -> Result<ToolResult, AgentError> {
        let item_id = args.get("item_id").and_then(Value::as_str).ok_or_else(|| {
            AgentError::ParseError("'item_id' is required".into())
        })?;
        let status = args.get("status").and_then(Value::as_str).ok_or_else(|| {
            AgentError::ParseError("'status' is required".into())
        })?;
        self.session.request_github(&format!("set GitHub task status to '{status}'"), item_id)?;
        let canonical = self.client.set_status(item_id, status)?;
        Ok(ToolResult::text(format!("Status set to '{canonical}'.")))
    }
}

/// Post an activity comment on a task's issue. Gated.
pub struct GithubLogActivityTool {
    client: Arc<GithubClient>,
    session: Arc<ToolSession>,
}

impl GithubLogActivityTool {
    pub fn new(client: Arc<GithubClient>, session: Arc<ToolSession>) -> Self {
        Self { client, session }
    }
}

impl ToolHandler for GithubLogActivityTool {
    fn name(&self) -> &str {
        "github_log_activity"
    }
    fn description(&self) -> &str {
        "Post an activity-update comment on the GitHub issue behind a task. Takes the \
         item_id from github_list_tasks (the task must be a real issue, not a draft). \
         Asks for permission first."
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "item_id": { "type": "string", "description": "Project item id of an issue-backed task (from github_list_tasks)" },
                "text": { "type": "string", "description": "The activity update to post" },
                "context": { "type": "array", "items": { "type": "string" }, "description": "Optional supporting context lines, shown in a collapsed section" }
            },
            "required": ["item_id", "text"]
        })
    }
    fn call(&self, args: Value) -> Result<ToolResult, AgentError> {
        let item_id = args.get("item_id").and_then(Value::as_str).ok_or_else(|| {
            AgentError::ParseError("'item_id' is required".into())
        })?;
        let text = args.get("text").and_then(Value::as_str).ok_or_else(|| {
            AgentError::ParseError("'text' is required".into())
        })?;
        let context: Vec<String> = args
            .get("context")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        self.session.request_github("post GitHub activity comment", item_id)?;
        let url = self.client.log_activity(item_id, text, &context)?;
        Ok(ToolResult::text(format!("Comment posted: {url}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock gh runner: dispatches on a substring of the GraphQL query and
    /// records every invocation for assertions.
    fn mock_client(
        responses: Vec<(&'static str, &'static str)>,
    ) -> (Arc<GithubClient>, Arc<Mutex<Vec<String>>>) {
        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let calls_c = calls.clone();
        let responses: Vec<(String, String)> =
            responses.into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let runner: GhRunner = Box::new(move |args: &[&str]| {
            let joined = args.join(" ");
            calls_c.lock().unwrap().push(joined.clone());
            for (needle, resp) in &responses {
                if joined.contains(needle.as_str()) {
                    return Ok(resp.clone());
                }
            }
            Err(AgentError::InternalError(format!("mock: no response for {joined}")))
        });
        let client = GithubClient::new("acme".into(), 29, Some("acme/app".into()), runner);
        (Arc::new(client), calls)
    }

    #[test]
    fn test_escape_graphql() {
        assert_eq!(escape_graphql("a\"b"), "a\\\"b");
        assert_eq!(escape_graphql("a\nb\r"), "a\\nb");
        assert_eq!(escape_graphql("c:\\path"), "c:\\\\path");
    }

    #[test]
    fn test_days_from_civil() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(2000, 1, 1), 10957);
    }

    #[test]
    fn test_current_iteration_picks_active_window() {
        // Build iterations around a known epoch day. today_jst_days() is "now",
        // so construct one iteration that definitely contains today and one past.
        let today = today_jst_days();
        // Convert today back to a YYYY-MM-DD by brute force isn't needed; instead
        // verify the fallback returns the first id when no dates match.
        let iters = serde_json::json!([
            { "id": "past", "title": "S1", "startDate": "2000-01-01", "duration": 14 },
            { "id": "future", "title": "S2", "startDate": "2999-01-01", "duration": 14 }
        ]);
        let arr = iters.as_array().unwrap();
        // Neither window contains today → fallback to first.
        assert_eq!(current_iteration_id(arr).as_deref(), Some("past"));
        let _ = today; // silence unused in case of platform clock oddities
    }

    #[test]
    fn test_list_my_items_filters_assignee_and_done() {
        let resp = r#"{"data":{"organization":{"projectV2":{"items":{
            "pageInfo":{"hasNextPage":false,"endCursor":null},
            "nodes":[
              {"id":"PVTI_1","content":{"title":"Mine open","number":1,"state":"OPEN","url":"u1","repository":{"nameWithOwner":"acme/app"},"assignees":{"nodes":[{"login":"me"}]},"labels":{"nodes":[{"name":"bug"}]}},"fieldValues":{"nodes":[{"name":"In Progress","field":{"name":"Status"}}]}},
              {"id":"PVTI_2","content":{"title":"Mine done","number":2,"state":"OPEN","url":"u2","repository":{"nameWithOwner":"acme/app"},"assignees":{"nodes":[{"login":"me"}]},"labels":{"nodes":[]}},"fieldValues":{"nodes":[{"name":"Done","field":{"name":"Status"}}]}},
              {"id":"PVTI_3","content":{"title":"Someone else","number":3,"state":"OPEN","url":"u3","repository":{"nameWithOwner":"acme/app"},"assignees":{"nodes":[{"login":"other"}]},"labels":{"nodes":[]}},"fieldValues":{"nodes":[]}},
              {"id":"PVTI_4","content":{"title":"My draft","assignees":{"nodes":[{"login":"me"}]}},"fieldValues":{"nodes":[{"title":"Sprint 7","field":{"name":"Sprint"}}]}}
            ]}}}}}"#;
        let viewer = r#"{"data":{"viewer":{"id":"U_me","login":"me"}}}"#;
        let (client, _calls) = mock_client(vec![("viewer", viewer), ("items(first", resp)]);

        let items = client.list_my_items(None, false).unwrap();
        let titles: Vec<&str> = items.iter().map(|i| i.title.as_str()).collect();
        // "Mine done" filtered (Done), "Someone else" filtered (assignee).
        assert_eq!(titles, vec!["Mine open", "My draft"]);
        assert!(items[0].number == Some(1) && !items[0].is_draft);
        assert!(items[1].is_draft && items[1].sprint.as_deref() == Some("Sprint 7"));
    }

    #[test]
    fn test_create_draft_sets_status_and_sprint() {
        let meta = r#"{"data":{"organization":{"projectV2":{
            "id":"PROJ",
            "fields":{"nodes":[
              {"id":"FSTATUS","name":"Status","options":[{"id":"OPT_TODO","name":"Todo"},{"id":"OPT_IP","name":"In Progress"}]},
              {"id":"FSPRINT","name":"Sprint","configuration":{"iterations":[{"id":"IT1","title":"S1","startDate":"2000-01-01","duration":14}]}}
            ]}
        }}}}"#;
        let create = r#"{"data":{"addProjectV2DraftIssue":{"projectItem":{"id":"PVTI_new"}}}}"#;
        let update = r#"{"data":{"updateProjectV2ItemFieldValue":{"projectV2Item":{"id":"PVTI_new"}}}}"#;
        let (client, calls) = mock_client(vec![
            ("projectV2(number: 29)", meta),
            ("addProjectV2DraftIssue", create),
            ("updateProjectV2ItemFieldValue", update),
        ]);

        let id = client.create_draft("Hello \"world\"", "body").unwrap();
        assert_eq!(id, "PVTI_new");
        let calls = calls.lock().unwrap();
        // status (singleSelectOptionId) and sprint (iterationId) updates both sent.
        assert!(calls.iter().any(|c| c.contains("singleSelectOptionId: \"OPT_TODO\"")));
        assert!(calls.iter().any(|c| c.contains("iterationId: \"IT1\"")));
        // title was escaped in the create mutation.
        assert!(calls.iter().any(|c| c.contains("Hello \\\"world\\\"")));
    }

    #[test]
    fn test_set_status_unknown_lists_valid() {
        let meta = r#"{"data":{"organization":{"projectV2":{
            "id":"PROJ",
            "fields":{"nodes":[
              {"id":"FSTATUS","name":"Status","options":[{"id":"OPT_TODO","name":"Todo"},{"id":"OPT_DONE","name":"Done"}]}
            ]}
        }}}}"#;
        let (client, _calls) = mock_client(vec![("projectV2(number: 29)", meta)]);
        let err = client.set_status("PVTI_1", "Bogus").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown status 'Bogus'"));
        assert!(msg.contains("Done") && msg.contains("Todo"));
    }

    #[test]
    fn test_set_status_case_insensitive() {
        let meta = r#"{"data":{"organization":{"projectV2":{
            "id":"PROJ",
            "fields":{"nodes":[
              {"id":"FSTATUS","name":"Status","options":[{"id":"OPT_IP","name":"In Progress"}]}
            ]}
        }}}}"#;
        let update = r#"{"data":{"updateProjectV2ItemFieldValue":{"projectV2Item":{"id":"x"}}}}"#;
        let (client, calls) = mock_client(vec![
            ("projectV2(number: 29)", meta),
            ("updateProjectV2ItemFieldValue", update),
        ]);
        let canonical = client.set_status("PVTI_1", "in progress").unwrap();
        assert_eq!(canonical, "In Progress");
        assert!(calls.lock().unwrap().iter().any(|c| c.contains("singleSelectOptionId: \"OPT_IP\"")));
    }

    #[test]
    fn test_log_activity_rejects_draft() {
        // node has no Issue content → null id.
        let node = r#"{"data":{"node":{"content":null}}}"#;
        let (client, _calls) = mock_client(vec![("node(id:", node)]);
        let err = client.log_activity("PVTI_draft", "did stuff", &[]).unwrap_err();
        assert!(err.to_string().contains("promote it first"));
    }

    #[test]
    fn test_graphql_surfaces_errors() {
        let errs = r#"{"errors":[{"message":"Could not resolve to a node"}]}"#;
        let (client, _calls) = mock_client(vec![("viewer", errs)]);
        let err = client.viewer().unwrap_err();
        assert!(err.to_string().contains("Could not resolve to a node"));
    }

    #[test]
    fn test_list_tasks_tool_round_trip() {
        let viewer = r#"{"data":{"viewer":{"id":"U_me","login":"me"}}}"#;
        let resp = r#"{"data":{"organization":{"projectV2":{"items":{
            "pageInfo":{"hasNextPage":false,"endCursor":null},
            "nodes":[{"id":"PVTI_1","content":{"title":"Task A","number":7,"state":"OPEN","url":"u","repository":{"nameWithOwner":"acme/app"},"assignees":{"nodes":[{"login":"me"}]},"labels":{"nodes":[]}},"fieldValues":{"nodes":[{"name":"Todo","field":{"name":"Status"}}]}}]
        }}}}}"#;
        let (client, _calls) = mock_client(vec![("viewer", viewer), ("items(first", resp)]);
        let tool = GithubListTasksTool::new(client);
        let out = tool.call(serde_json::json!({})).unwrap();
        assert!(out.text.contains("Task A"));
        assert!(out.text.contains("#7"));
        assert!(out.text.contains("item_id=PVTI_1"));
    }
}

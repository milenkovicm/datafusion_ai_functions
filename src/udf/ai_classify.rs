//! `ai_classify(content, labels [, options])` async scalar UDF.
//!
//! The model is instructed to answer with only the label name(s). The answer is
//! matched against the labels. By default the result is just the label, e.g.
//! `urgent` (NULL if the answer matches no label). With `responseFormat = 'json'`,
//! or when `version` or `multilabel = 'true'` is set without `responseFormat`,
//! the result is JSON:
//!
//! ```json
//! {"response":[{"value":"urgent"}],"metadata":{"version":"2.1"},"error_message":null}
//! ```

use async_trait::async_trait;
use datafusion::arrow::array::Array;
use datafusion::common::{Result, exec_err};
use datafusion::logical_expr::async_udf::AsyncScalarUDFImpl;
use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs};
use serde_json::{Value, json};

use super::{
    Arg, OptionRows, ai_udf, is_string_or_null, map_rows, memo_rows, non_blank, parse_bool,
    resolve_model, utf8_arg,
};
use crate::config::AiQueryConfig;
use crate::openai::{ChatRequest, chat};

pub(crate) const NAME: &str = "ai_classify";
const MIN_LABELS: usize = 2;
const MAX_LABELS: usize = 500;
const MAX_LABEL_LEN: usize = 100;
const MAX_DESCRIPTION_LEN: usize = 1000;
const MAX_INSTRUCTIONS_LEN: usize = 20_000;

ai_udf!(
    AiClassify,
    name: NAME,
    args: [
        Arg::string("content"),
        Arg::new("labels", "a JSON string", is_string_or_null),
        Arg::options(),
    ],
    required: 2,
);

#[async_trait]
impl AsyncScalarUDFImpl for AiClassify {
    async fn invoke_async_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let cfg = AiQueryConfig::from_options(&args.config_options);
        let contents = utf8_arg(&args, 0)?;
        let labels = utf8_arg(&args, 1)?;
        let options = OptionRows::new(NAME, &args, 2)?;
        // Rows with NULL content or NULL labels send no request.
        let specs = memo_rows(
            contents.len(),
            |i| {
                (contents.is_valid(i) && labels.is_valid(i))
                    .then(|| (labels.value(i), options.row(i)))
            },
            |(raw_labels, raw_options)| RowSpec::parse(&cfg, raw_labels, raw_options),
        )?;
        let client = self.clients.get(&cfg)?;

        let out = map_rows(cfg.max_concurrency, contents.len(), |i| {
            let (client, cfg, contents, specs) = (&client, &cfg, &contents, &specs);
            async move {
                let Some(spec) = &specs[i] else {
                    return Ok(None);
                };
                let req = ChatRequest {
                    model: &spec.model,
                    system: Some(&spec.system_prompt),
                    user: contents.value(i),
                    temperature: Some(cfg.temperature.unwrap_or(0.0)),
                    ..Default::default()
                };
                // An error goes into `error_message` (JSON) or becomes NULL (text).
                let result = match chat(client, cfg, req).await {
                    Ok(answer) => parse_answer(&answer, &spec.labels, spec.options.multilabel)
                        .inspect_err(|msg| log::warn!("{NAME}: {msg}")),
                    Err(e) if cfg.fail_on_error => return Err(e),
                    Err(e) => {
                        log::warn!("{NAME} request failed: {e}");
                        Err(e.to_string())
                    }
                };
                if spec.json {
                    return Ok(Some(render_response(spec.options.version, result)));
                }
                // `parse_answer` returns at least one label on success.
                Ok(result.ok().map(|mut names| names.swap_remove(0)))
            }
        })
        .await?;
        Ok(ColumnarValue::Array(out))
    }
}

/// Output format version (the `version` option).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Version {
    V2_0,
    #[default]
    V2_1,
}

impl Version {
    fn as_str(self) -> &'static str {
        match self {
            Version::V2_0 => "2.0",
            Version::V2_1 => "2.1",
        }
    }
}

/// Output format (the `responseFormat` option).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    /// Just the label.
    Text,
    /// JSON envelope.
    Json,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ClassifyOptions {
    version: Version,
    /// Whether `version` was given explicitly.
    version_set: bool,
    response_format: Option<OutputFormat>,
    multilabel: bool,
    instructions: Option<String>,
    model: Option<String>,
}

impl ClassifyOptions {
    /// JSON when requested, or when `version` / `multilabel` is set without
    /// an explicit `responseFormat`.
    fn json_output(&self) -> bool {
        match self.response_format {
            Some(f) => f == OutputFormat::Json,
            None => self.version_set || self.multilabel,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Label {
    name: String,
    description: Option<String>,
}

/// Everything needed to classify one row, derived from `labels` and `options`.
#[derive(Debug)]
struct RowSpec {
    model: String,
    system_prompt: String,
    labels: Vec<Label>,
    options: ClassifyOptions,
    /// Render theJSON envelope instead of the plain label.
    json: bool,
}

impl RowSpec {
    fn parse(
        cfg: &AiQueryConfig,
        raw_labels: &str,
        raw_options: &[(String, Option<String>)],
    ) -> Result<Self> {
        let labels = parse_labels(raw_labels)?;
        let options = parse_options(raw_options)?;
        Ok(Self {
            model: resolve_model(cfg, NAME, options.model.as_deref())?,
            system_prompt: build_system_prompt(&labels, &options),
            labels,
            json: options.json_output(),
            options,
        })
    }
}

/// Parse `["a", "b"]` or `{"a": "description", "b": "description"}`.
fn parse_labels(raw: &str) -> Result<Vec<Label>> {
    let value: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(e) => return exec_err!("{NAME}: labels must be a JSON array or object: {e}"),
    };
    let labels: Vec<Label> = match value {
        Value::Array(items) => items
            .into_iter()
            .map(|v| match v {
                Value::String(name) => Ok(Label {
                    name,
                    description: None,
                }),
                other => exec_err!("{NAME}: label names must be strings, got {other}"),
            })
            .collect::<Result<_>>()?,
        Value::Object(map) => map
            .into_iter()
            .map(|(name, v)| match v {
                Value::String(d) if d.is_empty() => Ok(Label {
                    name,
                    description: None,
                }),
                Value::String(d) => Ok(Label {
                    name,
                    description: Some(d),
                }),
                Value::Null => Ok(Label {
                    name,
                    description: None,
                }),
                other => exec_err!("{NAME}: label descriptions must be strings, got {other}"),
            })
            .collect::<Result<_>>()?,
        _ => return exec_err!("{NAME}: labels must be a JSON array or object"),
    };

    if !(MIN_LABELS..=MAX_LABELS).contains(&labels.len()) {
        return exec_err!(
            "{NAME}: expected {MIN_LABELS} to {MAX_LABELS} labels, got {}",
            labels.len()
        );
    }
    for (i, l) in labels.iter().enumerate() {
        let len = l.name.chars().count();
        if len == 0 || len > MAX_LABEL_LEN || l.name.trim().is_empty() {
            return exec_err!(
                "{NAME}: label names must be 1 to {MAX_LABEL_LEN} characters, got '{}'",
                l.name
            );
        }
        if l.description
            .as_ref()
            .is_some_and(|d| d.chars().count() > MAX_DESCRIPTION_LEN)
        {
            return exec_err!(
                "{NAME}: description of label '{}' is longer than {MAX_DESCRIPTION_LEN} characters",
                l.name
            );
        }
        if labels[..i].iter().any(|o| o.name == l.name) {
            return exec_err!("{NAME}: duplicate label '{}'", l.name);
        }
    }
    Ok(labels)
}

fn parse_options(entries: &[(String, Option<String>)]) -> Result<ClassifyOptions> {
    let mut opts = ClassifyOptions::default();
    for (key, value) in entries {
        let Some(value) = value else {
            continue;
        };
        match key.to_ascii_lowercase().as_str() {
            "version" => {
                opts.version = match value.trim() {
                    "2.0" => Version::V2_0,
                    "2.1" => Version::V2_1,
                    v => {
                        return exec_err!(
                            "{NAME}: unsupported version '{v}', expected '2.0' or '2.1'"
                        );
                    }
                };
                opts.version_set = true;
            }
            "responseformat" => {
                opts.response_format = match value.trim().to_ascii_lowercase().as_str() {
                    "" => None,
                    "text" => Some(OutputFormat::Text),
                    "json" => Some(OutputFormat::Json),
                    _ => {
                        return exec_err!(
                            "{NAME}: option '{key}' must be 'text' or 'json', got '{value}'"
                        );
                    }
                }
            }
            "multilabel" => opts.multilabel = parse_bool(NAME, key, value)?,
            "instructions" => {
                if value.chars().count() >= MAX_INSTRUCTIONS_LEN {
                    return exec_err!(
                        "{NAME}: instructions must be shorter than {MAX_INSTRUCTIONS_LEN} characters"
                    );
                }
                opts.instructions = non_blank(value);
            }
            "model" => opts.model = non_blank(value),
            "enableconfidencescores" | "enablerationales" => {
                if parse_bool(NAME, key, value)? {
                    return exec_err!("{NAME}: option '{key}' is not supported yet");
                }
            }
            _ => return exec_err!("{NAME}: unknown option '{key}'"),
        }
    }
    if opts.multilabel && opts.response_format == Some(OutputFormat::Text) {
        return exec_err!("{NAME}: multilabel requires responseFormat 'json'");
    }
    Ok(opts)
}

fn build_system_prompt(labels: &[Label], opts: &ClassifyOptions) -> String {
    let (how_many, answer) = if opts.multilabel {
        (
            "one or more",
            "the names of all applicable labels, one per line",
        )
    } else {
        ("exactly one", "the label name")
    };
    let mut prompt = format!(
        "You are a text classifier. Classify the text provided by the user into {how_many} of the following labels.\n"
    );
    if let Some(instructions) = &opts.instructions {
        prompt.push_str(&format!("Additional instructions: {instructions}\n"));
    }
    prompt.push_str("Labels:\n");
    for l in labels {
        match &l.description {
            Some(d) => prompt.push_str(&format!("- {}: {d}\n", l.name)),
            None => prompt.push_str(&format!("- {}\n", l.name)),
        }
    }
    prompt.push_str(&format!(
        "Respond with only {answer}, exactly as written above. \
         Do not output explanations, punctuation, quotes or any other text."
    ));
    prompt
}

/// Strip formatting a model commonly adds around a label.
fn clean_token(mut s: &str) -> &str {
    loop {
        let next = s
            .trim()
            .trim_start_matches(['-', '*', '•'])
            .trim_matches(['"', '\'', '`', '[', ']'])
            .trim_end_matches('.');
        if next == s {
            return s;
        }
        s = next;
    }
}

fn find_label<'a>(token: &str, labels: &'a [Label]) -> Option<&'a Label> {
    let token = clean_token(token);
    if token.is_empty() {
        return None;
    }
    labels.iter().find(|l| l.name == token).or_else(|| {
        let token = token.to_lowercase();
        labels.iter().find(|l| l.name.to_lowercase() == token)
    })
}

/// Match the model's answer against the labels. Returns label names, or an
/// error message when nothing matches.
fn parse_answer(answer: &str, labels: &[Label], multilabel: bool) -> Result<Vec<String>, String> {
    let mut found: Vec<&Label> = Vec::new();
    if let Some(l) = find_label(answer, labels) {
        found.push(l);
    } else {
        for line in answer.lines() {
            match find_label(line, labels) {
                Some(l) => found.push(l),
                None if multilabel => {
                    found.extend(line.split(',').filter_map(|t| find_label(t, labels)))
                }
                None => {}
            }
            if !multilabel && !found.is_empty() {
                break;
            }
        }
    }
    if found.is_empty() {
        return Err(format!(
            "model returned '{}' which is not one of the labels",
            answer.trim()
        ));
    }
    // Keep label order and drop duplicates.
    Ok(labels
        .iter()
        .filter(|l| found.iter().any(|f| std::ptr::eq(*f, *l)))
        .map(|l| l.name.clone())
        .collect())
}

fn render_response(version: Version, result: Result<Vec<String>, String>) -> String {
    let (response, error) = match result {
        Ok(names) => {
            let response = match version {
                Version::V2_0 => json!(names),
                Version::V2_1 => names.into_iter().map(|v| json!({ "value": v })).collect(),
            };
            (response, Value::Null)
        }
        Err(msg) => (Value::Null, Value::String(msg)),
    };
    json!({
        "response": response,
        "metadata": { "version": version.as_str() },
        "error_message": error,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(names: &[&str]) -> Vec<Label> {
        names
            .iter()
            .map(|n| Label {
                name: n.to_string(),
                description: None,
            })
            .collect()
    }

    fn opts(entries: &[(&str, &str)]) -> Result<ClassifyOptions> {
        let e: Vec<_> = entries
            .iter()
            .map(|(k, v)| (k.to_string(), Some(v.to_string())))
            .collect();
        parse_options(&e)
    }

    #[test]
    fn parses_array_and_object_labels() {
        assert_eq!(parse_labels(r#"["a", "b"]"#).unwrap(), labels(&["a", "b"]));
        let l = parse_labels(r#"{"z": "last letter", "a": ""}"#).unwrap();
        assert_eq!(l[0].name, "z"); // object order is kept
        assert_eq!(l[0].description.as_deref(), Some("last letter"));
        assert_eq!(l[1].description, None);
    }

    #[test]
    fn rejects_invalid_labels() {
        for bad in [
            r#"["only"]"#,
            r#"["a", "a"]"#,
            r#"["a", ""]"#,
            r#"["a", 1]"#,
            "a,b",
            "42",
        ] {
            assert!(parse_labels(bad).is_err(), "{bad}");
        }
        let long = format!(r#"["a", "{}"]"#, "x".repeat(101));
        assert!(parse_labels(&long).is_err());
    }

    #[test]
    fn parses_options() {
        assert_eq!(opts(&[]).unwrap(), ClassifyOptions::default());
        let o = opts(&[
            ("version", "2.0"),
            ("multilabel", "TRUE"),
            ("instructions", "be strict"),
        ])
        .unwrap();
        assert_eq!(o.version, Version::V2_0);
        assert!(o.version_set);
        assert!(o.multilabel);
        assert_eq!(o.instructions.as_deref(), Some("be strict"));
        assert_eq!(
            opts(&[("responseFormat", "JSON")]).unwrap().response_format,
            Some(OutputFormat::Json)
        );
        assert_eq!(
            opts(&[("responseformat", "text")]).unwrap().response_format,
            Some(OutputFormat::Text)
        );
        assert_eq!(
            opts(&[("responseFormat", " ")]).unwrap().response_format,
            None
        );
        assert!(opts(&[("responseFormat", "xml")]).is_err());
        assert!(opts(&[("responseFormat", "text"), ("multilabel", "true")]).is_err());
        assert!(opts(&[("responseFormat", "text"), ("multilabel", "false")]).is_ok());
        assert!(opts(&[("enableConfidenceScores", "false")]).is_ok());
        assert!(opts(&[("enableConfidenceScores", "true")]).is_err());
        assert!(opts(&[("version", "1.0")]).is_err());
        assert!(opts(&[("multilabel", "yes")]).is_err());
        assert!(opts(&[("nope", "x")]).is_err());
    }

    #[test]
    fn picks_output_format() {
        let json = |e: &[(&str, &str)]| opts(e).unwrap().json_output();
        assert!(!json(&[]));
        assert!(!json(&[("responseFormat", "text")]));
        assert!(!json(&[("responseFormat", "text"), ("version", "2.1")]));
        assert!(!json(&[("multilabel", "false")]));
        assert!(json(&[("responseFormat", "json")]));
        assert!(json(&[("version", "2.1")]));
        assert!(json(&[("multilabel", "true")]));
    }

    #[test]
    fn prompt_lists_labels_and_instructions() {
        let l = parse_labels(r#"{"billing": "money things", "other": ""}"#).unwrap();
        let o = opts(&[("instructions", "support tickets")]).unwrap();
        let p = build_system_prompt(&l, &o);
        assert!(p.contains("exactly one"));
        assert!(p.contains("- billing: money things\n- other\n"));
        assert!(p.contains("Additional instructions: support tickets"));
        let p = build_system_prompt(&l, &opts(&[("multilabel", "true")]).unwrap());
        assert!(p.contains("one or more") && p.contains("one per line"));
    }

    #[test]
    fn matches_answers() {
        let l = labels(&["urgent", "not_urgent"]);
        assert_eq!(parse_answer("urgent", &l, false).unwrap(), ["urgent"]);
        assert_eq!(
            parse_answer(" \"Not_Urgent\".\n", &l, false).unwrap(),
            ["not_urgent"]
        );
        assert_eq!(parse_answer("`urgent`", &l, false).unwrap(), ["urgent"]);
        assert!(parse_answer("maybe", &l, false).is_err());

        let l = labels(&["a", "b", "c"]);
        assert_eq!(parse_answer("c\n- a\nc", &l, true).unwrap(), ["a", "c"]);
        assert_eq!(parse_answer("b, c", &l, true).unwrap(), ["b", "c"]);
        assert_eq!(parse_answer("b\nc", &l, false).unwrap(), ["b"]);
    }

    #[test]
    fn renders_versions() {
        assert_eq!(
            render_response(Version::V2_1, Ok(vec!["urgent".into()])),
            r#"{"response":[{"value":"urgent"}],"metadata":{"version":"2.1"},"error_message":null}"#
        );
        assert_eq!(
            render_response(Version::V2_0, Ok(vec!["a".into(), "b".into()])),
            r#"{"response":["a","b"],"metadata":{"version":"2.0"},"error_message":null}"#
        );
        assert_eq!(
            render_response(Version::V2_1, Err("boom".into())),
            r#"{"response":null,"metadata":{"version":"2.1"},"error_message":"boom"}"#
        );
    }
}

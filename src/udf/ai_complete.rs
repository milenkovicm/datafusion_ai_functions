//! `ai_complete(prompt [, options])` async scalar UDF.
//!
//! `AI_COMPLETE(model, prompt [, model_parameters, response_format,
//! show_details])` with everything except the prompt moved into an optional
//! `MAP<STRING, STRING>`: `model`, `temperature`, `top_p`, `max_tokens`,
//! `response_format`, `show_details` and `fail_on_error`.
//!
//! With `show_details = 'true'` the result is JSON:
//!
//! ```json
//! {"choices":[{"messages":"..."}],"created":1708536426,"model":"m",
//!  "usage":{"completion_tokens":10,"prompt_tokens":22,"total_tokens":32}}
//! ```

use async_openai::types::chat::{ResponseFormat, ResponseFormatJsonSchema};
use async_trait::async_trait;
use datafusion::arrow::array::Array;
use datafusion::common::{Result, exec_err};
use datafusion::logical_expr::async_udf::AsyncScalarUDFImpl;
use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs};
use serde_json::{Value, json};

use super::{
    Arg, OptionRows, ai_udf, map_rows, memo_rows, non_blank, null_on_error, parse_bool,
    parse_f64_min, parse_f64_range, parse_response_format, parse_u32, resolve_model, utf8_arg,
};
use crate::config::AiQueryConfig;
use crate::openai::{ChatRequest, ChatResponse, chat_full};

pub(crate) const NAME: &str = "ai_complete";
/// `json_schema.name` used `{"type": "json", "schema": ...}`.
const SCHEMA_NAME: &str = "response";

ai_udf!(
    AiComplete,
    name: NAME,
    args: [Arg::string("prompt"), Arg::options()],
    required: 1,
);

#[async_trait]
impl AsyncScalarUDFImpl for AiComplete {
    async fn invoke_async_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let cfg = AiQueryConfig::from_options(&args.config_options);
        let prompts = utf8_arg(&args, 0)?;
        let options = OptionRows::new(NAME, &args, 1)?;
        let specs = memo_rows(
            prompts.len(),
            |i| prompts.is_valid(i).then(|| options.row(i)),
            |raw| parse_options(&cfg, raw),
        )?;
        let client = self.clients.get(&cfg)?;

        let out = map_rows(cfg.max_concurrency, prompts.len(), |i| {
            let (client, cfg, prompts, specs) = (&client, &cfg, &prompts, &specs);
            async move {
                let Some(spec) = &specs[i] else {
                    return Ok(None);
                };
                let req = ChatRequest {
                    model: &spec.model,
                    system: Some(cfg.system_prompt.as_str()),
                    user: prompts.value(i),
                    temperature: spec.temperature,
                    max_tokens: spec.max_tokens,
                    top_p: spec.top_p,
                    response_format: spec.response_format.as_ref(),
                };
                let result = chat_full(client, cfg, req).await.map(|resp| {
                    if spec.show_details {
                        render_details(&resp, spec.response_format.is_some())
                    } else {
                        resp.content
                    }
                });
                null_on_error(NAME, spec.fail_on_error, result)
            }
        })
        .await?;
        Ok(ColumnarValue::Array(out))
    }
}

/// Everything needed to send one row's request, resolved from the
/// `options` map and the session config.
#[derive(Debug, Clone, PartialEq)]
struct RowSpec {
    model: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    response_format: Option<ResponseFormat>,
    show_details: bool,
    fail_on_error: bool,
}

fn parse_options(cfg: &AiQueryConfig, entries: &[(String, Option<String>)]) -> Result<RowSpec> {
    let (mut model, mut temperature, mut top_p, mut max_tokens) = (None, None, None, None);
    let (mut response_format, mut show_details, mut fail_on_error) = (None, false, None);
    for (key, value) in entries {
        let Some(value) = value else {
            continue;
        };
        match key.to_ascii_lowercase().as_str() {
            "model" => model = non_blank(value),
            "temperature" => temperature = Some(parse_f64_min(NAME, key, value, 0.0)?),
            "top_p" => top_p = Some(parse_f64_range(NAME, key, value, 0.0, 1.0)?),
            "max_tokens" => max_tokens = Some(parse_u32(NAME, key, value)?),
            "response_format" => response_format = parse_format(key, value)?,
            "show_details" => show_details = parse_bool(NAME, key, value)?,
            "fail_on_error" => fail_on_error = Some(parse_bool(NAME, key, value)?),
            _ => return exec_err!("{NAME}: unknown option '{key}'"),
        }
    }
    Ok(RowSpec {
        model: resolve_model(cfg, NAME, model.as_deref())?,
        temperature,
        top_p,
        max_tokens,
        response_format,
        show_details,
        fail_on_error: fail_on_error.unwrap_or(cfg.fail_on_error),
    })
}

/// `{"type": "json", "schema": {...}}` becomes an OpenAI
/// `json_schema` response format. Anything else is handled like `ai_query`'s
/// `responseFormat` (`text`, `json_object` or an OpenAI `response_format`).
fn parse_format(key: &str, value: &str) -> Result<Option<ResponseFormat>> {
    if let Ok(Value::Object(mut obj)) = serde_json::from_str::<Value>(value.trim())
        && obj
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|t| t.eq_ignore_ascii_case("json"))
    {
        return match obj.remove("schema") {
            Some(schema @ Value::Object(_)) => Ok(Some(ResponseFormat::JsonSchema {
                json_schema: ResponseFormatJsonSchema {
                    description: None,
                    name: SCHEMA_NAME.to_string(),
                    schema,
                    strict: None,
                },
            })),
            _ => exec_err!("{NAME}: option '{key}' with type 'json' needs a 'schema' object"),
        };
    }
    parse_response_format(NAME, key, value)
}

/// `show_details` output. With a response format the content goes
/// into `structured_output` (parsed as JSON when possible), otherwise into
/// `choices`.
fn render_details(resp: &ChatResponse, structured: bool) -> String {
    let usage = match &resp.usage {
        Some(u) => json!({
            "completion_tokens": u.completion_tokens,
            "prompt_tokens": u.prompt_tokens,
            "total_tokens": u.total_tokens,
        }),
        None => json!({}),
    };
    let mut out = serde_json::Map::new();
    if structured {
        let raw = serde_json::from_str::<Value>(&resp.content)
            .unwrap_or_else(|_| Value::String(resp.content.clone()));
        out.insert(
            "structured_output".into(),
            json!([{ "raw_message": raw, "type": "json" }]),
        );
    } else {
        out.insert("choices".into(), json!([{ "messages": resp.content }]));
    }
    out.insert("created".into(), json!(resp.created));
    out.insert("model".into(), json!(resp.model));
    out.insert("usage".into(), usage);
    Value::Object(out).to_string()
}

#[cfg(test)]
mod tests {
    use async_openai::types::chat::CompletionUsage;

    use super::*;

    fn cfg() -> AiQueryConfig {
        AiQueryConfig {
            default_model: "default".into(),
            ..Default::default()
        }
    }

    fn opts(entries: &[(&str, &str)]) -> Result<RowSpec> {
        let e: Vec<_> = entries
            .iter()
            .map(|(k, v)| (k.to_string(), Some(v.to_string())))
            .collect();
        parse_options(&cfg(), &e)
    }

    #[test]
    fn parses_options() {
        let defaults = opts(&[]).unwrap();
        assert_eq!(defaults.model, "default");
        assert!(defaults.fail_on_error && !defaults.show_details);
        let o = opts(&[
            ("Model", "m"),
            ("max_tokens", "64"),
            ("TEMPERATURE", "0.5"),
            ("top_p", "0.9"),
            ("show_details", "TRUE"),
            ("fail_on_error", "false"),
        ])
        .unwrap();
        assert_eq!(o.model, "m");
        assert_eq!(o.max_tokens, Some(64));
        assert_eq!(o.temperature, Some(0.5));
        assert_eq!(o.top_p, Some(0.9));
        assert!(o.show_details);
        assert!(!o.fail_on_error);
        let nulls = parse_options(&cfg(), &[("model".into(), None)]).unwrap();
        assert_eq!(nulls, defaults);
    }

    #[test]
    fn rejects_bad_options() {
        for bad in [
            ("max_tokens", "-1"),
            ("temperature", "-0.1"),
            ("top_p", "1.5"),
            ("top_p", "x"),
            ("show_details", "yes"),
            ("response_format", "{not json"),
            ("response_format", r#"{"type": "json"}"#),
            ("response_format", r#"{"type": "json", "schema": "x"}"#),
            ("guardrails", "true"),
            ("responseFormat", "text"),
        ] {
            assert!(opts(&[bad]).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn parses_response_format() {
        let f = |v| opts(&[("response_format", v)]).unwrap().response_format;
        let Some(ResponseFormat::JsonSchema { json_schema }) = f(
            r#"{"type": "JSON", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}"#,
        ) else {
            panic!("expected json_schema");
        };
        assert_eq!(json_schema.name, SCHEMA_NAME);
        assert_eq!(json_schema.strict, None);
        assert_eq!(json_schema.schema["properties"]["a"]["type"], "string");
        // OpenAI forms still work.
        assert_eq!(f("json_object"), Some(ResponseFormat::JsonObject));
        assert_eq!(f(r#"{"type": "text"}"#), Some(ResponseFormat::Text));
        assert_eq!(f(" "), None);
    }

    fn response(content: &str, usage: bool) -> ChatResponse {
        ChatResponse {
            content: content.into(),
            created: 1708536426,
            model: "m".into(),
            usage: usage.then_some(CompletionUsage {
                prompt_tokens: 22,
                completion_tokens: 10,
                total_tokens: 32,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
        }
    }

    #[test]
    fn renders_details() {
        assert_eq!(
            render_details(&response("hi", true), false),
            r#"{"choices":[{"messages":"hi"}],"created":1708536426,"model":"m","usage":{"completion_tokens":10,"prompt_tokens":22,"total_tokens":32}}"#
        );
        assert_eq!(
            render_details(&response(r#"{"a": 1}"#, false), true),
            r#"{"structured_output":[{"raw_message":{"a":1},"type":"json"}],"created":1708536426,"model":"m","usage":{}}"#
        );
        assert_eq!(
            render_details(&response("oops", false), true),
            r#"{"structured_output":[{"raw_message":"oops","type":"json"}],"created":1708536426,"model":"m","usage":{}}"#
        );
    }
}

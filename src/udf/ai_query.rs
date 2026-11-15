//! `ai_query(prompt [, options])` async scalar UDF.
//!
//! `options` is an optional `MAP<STRING, STRING>` with per call settings:
//! `model`, `max_tokens`, `temperature`, `responseFormat` and `failOnError`.
//! Without a `model` option, `datafusion_ai_functions.default_model` is used.

use async_openai::types::chat::ResponseFormat;
use async_trait::async_trait;
use datafusion::arrow::array::Array;
use datafusion::common::{Result, exec_err};
use datafusion::logical_expr::async_udf::AsyncScalarUDFImpl;
use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs};

use super::{
    Arg, OptionRows, ai_udf, map_rows, memo_rows, non_blank, null_on_error, parse_bool,
    parse_f64_min, parse_response_format, parse_u32, resolve_model, utf8_arg,
};
use crate::config::AiQueryConfig;
use crate::openai::{ChatRequest, chat};

pub(crate) const NAME: &str = "ai_query";

ai_udf!(
    AiQuery,
    name: NAME,
    args: [Arg::string("prompt"), Arg::options()],
    required: 1,
);

#[async_trait]
impl AsyncScalarUDFImpl for AiQuery {
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
                    top_p: None,
                    response_format: spec.response_format.as_ref(),
                };
                null_on_error(NAME, spec.fail_on_error, chat(client, cfg, req).await)
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
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    response_format: Option<ResponseFormat>,
    fail_on_error: bool,
}

fn parse_options(cfg: &AiQueryConfig, entries: &[(String, Option<String>)]) -> Result<RowSpec> {
    let (mut model, mut max_tokens, mut temperature, mut response_format, mut fail_on_error) =
        (None, None, None, None, None);
    for (key, value) in entries {
        let Some(value) = value else {
            continue;
        };
        match key.to_ascii_lowercase().as_str() {
            "model" => model = non_blank(value),
            "max_tokens" => max_tokens = Some(parse_u32(NAME, key, value)?),
            "temperature" => temperature = Some(parse_f64_min(NAME, key, value, 0.0)?),
            "responseformat" => response_format = parse_response_format(NAME, key, value)?,
            "failonerror" => fail_on_error = Some(parse_bool(NAME, key, value)?),
            _ => return exec_err!("{NAME}: unknown option '{key}'"),
        }
    }
    Ok(RowSpec {
        model: resolve_model(cfg, NAME, model.as_deref())?,
        max_tokens,
        temperature,
        response_format,
        fail_on_error: fail_on_error.unwrap_or(cfg.fail_on_error),
    })
}

#[cfg(test)]
mod tests {
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
        assert!(defaults.fail_on_error);
        let o = opts(&[
            ("Model", "m"),
            ("max_tokens", "64"),
            ("TEMPERATURE", "0.5"),
            ("failOnError", "False"),
        ])
        .unwrap();
        assert_eq!(o.model, "m");
        assert_eq!(o.max_tokens, Some(64));
        assert_eq!(o.temperature, Some(0.5));
        assert!(!o.fail_on_error);
        assert_eq!(opts(&[("model", "  ")]).unwrap().model, "default");
        let nulls = parse_options(&cfg(), &[("model".into(), None)]).unwrap();
        assert_eq!(nulls, defaults);
    }

    #[test]
    fn rejects_bad_options() {
        for bad in [
            ("max_tokens", "-1"),
            ("max_tokens", "ten"),
            ("temperature", "hot"),
            ("temperature", "-0.1"),
            ("failOnError", "maybe"),
            ("responseFormat", "{not json"),
            ("responseFormat", r#"{"type": "xml"}"#),
            ("nope", "x"),
        ] {
            assert!(opts(&[bad]).is_err(), "{bad:?}");
        }
        assert!(parse_options(&AiQueryConfig::default(), &[]).is_err());
    }

    #[test]
    fn parses_response_format() {
        let f = |v| opts(&[("responseFormat", v)]).unwrap().response_format;
        assert_eq!(f("text"), Some(ResponseFormat::Text));
        assert_eq!(f("JSON_OBJECT"), Some(ResponseFormat::JsonObject));
        assert_eq!(
            f(r#"{"type": "json_object"}"#),
            Some(ResponseFormat::JsonObject)
        );
        let Some(ResponseFormat::JsonSchema { json_schema }) = f(
            r#"{"type": "json_schema", "json_schema": {"name": "p", "strict": true,
                "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}}"#,
        ) else {
            panic!("expected json_schema");
        };
        assert_eq!(json_schema.name, "p");
        assert_eq!(json_schema.strict, Some(true));
        assert_eq!(json_schema.schema["properties"]["a"]["type"], "string");
    }
}

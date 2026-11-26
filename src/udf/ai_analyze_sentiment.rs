//! `ai_analyze_sentiment(content)` async scalar UDF.
//!
//! Returns one of `positive`, `negative`, `neutral` or `mixed`, or NULL when
//! the sentiment cannot be detected.

use async_trait::async_trait;
use datafusion::arrow::array::Array;
use datafusion::common::Result;
use datafusion::logical_expr::async_udf::AsyncScalarUDFImpl;
use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs};

use super::{Arg, ai_udf, map_unique_rows, null_on_error, resolve_model, string_array, utf8_arg};
use crate::config::AiQueryConfig;
use crate::openai::{ChatRequest, chat};

pub(crate) const NAME: &str = "ai_analyze_sentiment";
const SENTIMENTS: [&str; 4] = ["positive", "negative", "neutral", "mixed"];

const SYSTEM_PROMPT: &str = "You are a sentiment analysis assistant. \
    Determine the overall sentiment of the text provided by the user. \
    Respond with exactly one of these words: positive, negative, neutral, mixed. \
    Use mixed when the text clearly expresses both positive and negative sentiment. \
    If the sentiment cannot be determined, respond with unknown. \
    Do not output explanations, punctuation, quotes or any other text.";

ai_udf!(
    AiAnalyzeSentiment,
    name: NAME,
    args: [Arg::string("content")],
    required: 1,
);

#[async_trait]
impl AsyncScalarUDFImpl for AiAnalyzeSentiment {
    async fn invoke_async_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let cfg = AiQueryConfig::from_options(&args.config_options);
        let contents = utf8_arg(&args, 0)?;
        let model = resolve_model(&cfg, NAME, None)?;
        let client = self.clients.get(&cfg)?;

        // NULL content sends no request; identical contents are asked once.
        let key = |i: usize| contents.is_valid(i).then(|| contents.value(i));
        let out = map_unique_rows(cfg.max_concurrency, contents.len(), key, |i| {
            let (client, cfg, contents, model) = (&client, &cfg, &contents, &model);
            async move {
                let req = ChatRequest {
                    model,
                    system: Some(SYSTEM_PROMPT),
                    user: contents.value(i),
                    temperature: Some(cfg.temperature.unwrap_or(0.0)),
                    ..Default::default()
                };
                let answer = null_on_error(NAME, cfg.fail_on_error, chat(client, cfg, req).await)?;
                let sentiment = answer.as_deref().and_then(|a| {
                    let s = parse_sentiment(a);
                    if s.is_none() {
                        log::debug!("{NAME}: no sentiment in model answer '{}'", a.trim());
                    }
                    s
                });
                Ok(sentiment.map(str::to_string))
            }
        })
        .await?;
        Ok(ColumnarValue::Array(string_array(out)))
    }
}

/// Match the model's answer against the allowed sentiments. Anything else
/// (including `unknown`) means the sentiment couldn't be detected.
fn parse_sentiment(answer: &str) -> Option<&'static str> {
    let token = answer
        .trim()
        .trim_start_matches(['-', '*', '•'])
        .trim_matches(|c: char| c.is_whitespace() || "\"'`[].!".contains(c))
        .to_lowercase();
    SENTIMENTS.into_iter().find(|s| *s == token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sentiments() {
        assert_eq!(parse_sentiment("positive"), Some("positive"));
        assert_eq!(parse_sentiment(" \"Negative\".\n"), Some("negative"));
        assert_eq!(parse_sentiment("`MIXED`"), Some("mixed"));
        assert_eq!(parse_sentiment("- neutral"), Some("neutral"));
        assert_eq!(parse_sentiment("unknown"), None);
        assert_eq!(parse_sentiment("very positive"), None);
        assert_eq!(parse_sentiment(""), None);
    }
}

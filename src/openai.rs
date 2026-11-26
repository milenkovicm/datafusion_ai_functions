//! Thin wrapper around `async-openai` chat completions.

use std::sync::Mutex;
use std::time::Duration;

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::middleware::ReqwestService;
use async_openai::middleware::retry::OpenAIRetryLayer;
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CompletionUsage, CreateChatCompletionRequestArgs,
    FinishReason, ResponseFormat,
};
use datafusion::common::{DataFusionError, Result, exec_err};
use tower::Layer;

use crate::config::AiQueryConfig;

pub type OpenAiClient = Client<OpenAIConfig>;

/// Settings that require a new HTTP client when they change.
#[derive(Clone, PartialEq, Eq)]
struct ClientKey {
    endpoint: String,
    secret: String,
    timeout_ms: u64,
    max_retries: usize,
}

impl From<&AiQueryConfig> for ClientKey {
    fn from(cfg: &AiQueryConfig) -> Self {
        Self {
            endpoint: cfg.endpoint.clone(),
            secret: cfg.secret.clone(),
            timeout_ms: cfg.timeout_ms,
            max_retries: cfg.max_retries,
        }
    }
}

/// Number of distinct client settings kept, so a few sessions with
/// different endpoints don't keep rebuilding each other's client.
const MAX_CACHED_CLIENTS: usize = 4;

/// Caches clients so connection pools are reused across batches, queries and
/// functions. A client is rebuilt only when relevant config changes (e.g.
/// after `SET`). Least recently used entries are evicted first.
#[derive(Default)]
pub struct ClientCache {
    /// Most recently used last.
    inner: Mutex<Vec<(ClientKey, OpenAiClient)>>,
}

/// Doesn't print the cached settings, since they include the API token.
impl std::fmt::Debug for ClientCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientCache").finish_non_exhaustive()
    }
}

impl ClientCache {
    pub fn get(&self, cfg: &AiQueryConfig) -> Result<OpenAiClient> {
        let key = ClientKey::from(cfg);
        let mut entries = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = entries.iter().position(|(k, _)| *k == key) {
            let entry = entries.remove(pos);
            let client = entry.1.clone();
            entries.push(entry);
            return Ok(client);
        }
        let client = build_client(cfg)?;
        if entries.len() >= MAX_CACHED_CLIENTS {
            entries.remove(0);
        }
        entries.push((key, client.clone()));
        Ok(client)
    }
}

fn build_client(cfg: &AiQueryConfig) -> Result<OpenAiClient> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_millis(cfg.timeout_ms))
        .build()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    let mut oa_cfg = OpenAIConfig::new().with_api_base(cfg.endpoint.trim_end_matches('/'));
    if !cfg.secret.is_empty() {
        oa_cfg = oa_cfg.with_api_key(cfg.secret.clone());
    }

    let service = OpenAIRetryLayer::new(cfg.max_retries).layer(ReqwestService::new(http.clone()));
    Ok(Client::with_config(oa_cfg)
        .with_http_client(http)
        .with_http_service(service))
}

/// A single chat completion request.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatRequest<'a> {
    pub model: &'a str,
    /// Optional system message
    pub system: Option<&'a str>,
    /// User message
    pub user: &'a str,
    /// Overrides `cfg.temperature` when set
    pub temperature: Option<f64>,
    /// Overrides `cfg.max_tokens` when set
    pub max_tokens: Option<u32>,
    /// Optional nucleus sampling `top_p`
    pub top_p: Option<f64>,
    /// Optional `response_format` (e.g. JSON schema structured output)
    pub response_format: Option<&'a ResponseFormat>,
}

/// First choice's content plus response metadata.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    /// Unix timestamp (seconds)
    pub created: u32,
    /// Model name reported by the endpoint
    pub model: String,
    pub usage: Option<CompletionUsage>,
}

/// Send a single chat completion request and return the first choice's content.
pub async fn chat(
    client: &OpenAiClient,
    cfg: &AiQueryConfig,
    req: ChatRequest<'_>,
) -> Result<String> {
    Ok(chat_full(client, cfg, req).await?.content)
}

/// Send a single chat completion request and return the first choice's
/// content together with the response metadata.
pub async fn chat_full(
    client: &OpenAiClient,
    cfg: &AiQueryConfig,
    req: ChatRequest<'_>,
) -> Result<ChatResponse> {
    let ext = |e: async_openai::error::OpenAIError| DataFusionError::External(Box::new(e));

    let mut messages: Vec<ChatCompletionRequestMessage> = Vec::with_capacity(2);
    if let Some(system) = req.system.filter(|s| !s.is_empty()) {
        messages.push(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system)
                .build()
                .map_err(ext)?
                .into(),
        );
    }
    messages.push(
        ChatCompletionRequestUserMessageArgs::default()
            .content(req.user)
            .build()
            .map_err(ext)?
            .into(),
    );

    let mut builder = CreateChatCompletionRequestArgs::default();
    builder.model(req.model).messages(messages);
    if let Some(t) = req.max_tokens.or(cfg.max_tokens) {
        builder.max_completion_tokens(t);
    }
    if let Some(t) = req.temperature.or(cfg.temperature) {
        builder.temperature(t as f32);
    }
    if let Some(p) = req.top_p {
        builder.top_p(p as f32);
    }
    if let Some(f) = req.response_format {
        builder.response_format(f.clone());
    }
    let request = builder.build().map_err(ext)?;

    let resp = client.chat().create(request).await.map_err(ext)?;
    let (content, finish_reason) = match resp.choices.into_iter().next() {
        Some(c) => (c.message.content, c.finish_reason),
        None => (None, None),
    };
    let content = check_content(req.model, content, finish_reason)?;
    Ok(ChatResponse {
        content,
        created: resp.created,
        model: resp.model,
        usage: resp.usage,
    })
}

/// The answer, or an error when there is none. An answer that hit the token
/// limit before any text (e.g. a reasoning model that spent the limit on
/// thinking) is an error; one cut off part way is kept with a warning.
fn check_content(
    model: &str,
    content: Option<String>,
    finish_reason: Option<FinishReason>,
) -> Result<String> {
    let truncated = finish_reason == Some(FinishReason::Length);
    match content {
        Some(c) if !c.is_empty() => {
            if truncated {
                log::warn!("model '{model}' hit the token limit, the answer is cut off");
            }
            Ok(c)
        }
        _ if truncated => exec_err!(
            "model '{model}' hit the token limit before answering; raise max_tokens \
             (reasoning models spend tokens on thinking first)"
        ),
        Some(c) => Ok(c),
        None => exec_err!("model '{model}' returned no content"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(endpoint: &str) -> AiQueryConfig {
        AiQueryConfig {
            endpoint: endpoint.into(),
            secret: "sk-secret".into(),
            ..Default::default()
        }
    }

    #[test]
    fn checks_content() {
        let check = |c: Option<&str>, f| check_content("m", c.map(String::from), f);
        let length = Some(FinishReason::Length);
        assert_eq!(check(Some("hi"), Some(FinishReason::Stop)).unwrap(), "hi");
        assert_eq!(check(Some("cut"), length).unwrap(), "cut");
        assert_eq!(check(Some(""), None).unwrap(), "");
        for content in [Some(""), None] {
            let err = check(content, length).unwrap_err().to_string();
            assert!(
                err.contains("hit the token limit before answering"),
                "{err}"
            );
        }
        let err = check(None, None).unwrap_err().to_string();
        assert!(err.contains("returned no content"), "{err}");
    }

    #[test]
    fn debug_hides_token() {
        let cache = ClientCache::default();
        cache.get(&cfg("http://a/v1")).unwrap();
        let dbg = format!("{cache:?}");
        assert!(!dbg.contains("sk-secret"), "{dbg}");
    }

    #[test]
    fn keeps_recent_clients() {
        let cache = ClientCache::default();
        let len = || cache.inner.lock().unwrap().len();
        cache.get(&cfg("http://a/v1")).unwrap();
        cache.get(&cfg("http://a/v1")).unwrap();
        assert_eq!(len(), 1);
        for e in ["b", "c", "d", "e"] {
            cache.get(&cfg(&format!("http://{e}/v1"))).unwrap();
        }
        assert_eq!(len(), MAX_CACHED_CLIENTS);
        let endpoints: Vec<_> = cache
            .inner
            .lock()
            .unwrap()
            .iter()
            .map(|(k, _)| k.endpoint.clone())
            .collect();
        assert!(
            !endpoints.contains(&"http://a/v1".to_string()),
            "{endpoints:?}"
        );
    }
}

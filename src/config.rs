//! DataFusion configuration extension for `ai_query`.
//!
//! All options live under the `datafusion_ai_functions.` prefix and can be set either
//! programmatically or from SQL, e.g. `SET datafusion_ai_functions.endpoint = 'http://localhost:8000/v1'`.

use datafusion::common::config::{ConfigExtension, ConfigOptions};
use datafusion::common::extensions_options;

extensions_options! {
    /// Options controlling the `ai_query` function
    pub struct AiQueryConfig {
        /// OpenAI compatible API base url (must include the version path, e.g. `/v1`)
        pub endpoint: String, default = "https://api.openai.com/v1".to_string()
        /// API token sent as `Authorization: Bearer <token>`; empty means no token
        pub api_token: String, default = String::new()
        /// Maximum number of requests in flight at once for one function call
        pub max_concurrency: usize, default = 8
        /// Per request timeout in milliseconds
        pub timeout_ms: u64, default = 60_000
        /// Number of retries on 429 / 5xx / connect errors (exponential backoff)
        pub max_retries: usize, default = 3
        /// If true, a failed request fails the query; otherwise the row becomes NULL
        pub fail_on_error: bool, default = true
        /// Optional system prompt prepended to every request; empty means none
        pub system_prompt: String, default = String::new()
        /// Optional `max_completion_tokens` passed to the model
        pub max_tokens: Option<u32>, default = None
        /// Optional sampling temperature passed to the model
        pub temperature: Option<f64>, default = None
        /// Model used by all functions unless a call sets the `model` option
        pub default_model: String, default = String::new()
        /// If true (and the `AiSinglePartition` physical optimizer rule is
        /// installed), AI functions are evaluated on a single partition
        pub single_partition: bool, default = true
    }
}

impl ConfigExtension for AiQueryConfig {
    const PREFIX: &'static str = "datafusion_ai_functions";
}

impl AiQueryConfig {
    /// Defaults, with `endpoint` and `api_token` taken from
    /// `OPENAI_BASE_URL` and `OPENAI_API_KEY` when set.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = std::env::var("OPENAI_BASE_URL") {
            cfg.endpoint = v;
        }
        if let Ok(v) = std::env::var("OPENAI_API_KEY") {
            cfg.api_token = v;
        }
        cfg
    }

    /// Resolve config from session options, falling back to defaults.
    pub fn from_options(options: &ConfigOptions) -> Self {
        options
            .extensions
            .get::<AiQueryConfig>()
            .cloned()
            .unwrap_or_default()
    }
}

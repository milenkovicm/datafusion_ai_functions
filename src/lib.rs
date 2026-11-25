//! AI functions for DataFusion, backed by any
//! OpenAI compatible chat completions endpoint:
//!
//! - `ai_query(prompt [, options])`
//! - `ai_complete(prompt [, options])`
//! - `ai_classify(content, labels [, options])`
//! - `ai_summarize(content [, max_words])`
//! - `ai_analyze_sentiment(content)`
//!
//! and by a decision model endpoint (OpenAI `/decisions` or Ollama
//! `/systemone`, see `datafusion_ai_functions.decision_api`):
//!
//! - `ai_probability(input, condition [, options])`
//! - `ai_if(input, condition [, options])`
//! - `ai_score(input, instructions, levels [, options])`
//! - `ai_decide_classify(input, labels [, options])`
//! - `ai_decide(input, questions [, options])`
//!
//! ```sql
//! SET datafusion_ai_functions.endpoint = 'http://localhost:8000/v1';
//! SET datafusion_ai_functions.api_token = 'sk-...';
//! SET datafusion_ai_functions.default_model = 'qwen2.5:14b-instruct';
//! SELECT text, ai_query('Summarize: ' || text) AS summary FROM docs;
//! SELECT ai_query('Say hi', MAP('model', 'gpt-oss-120b', 'temperature', '0'));
//! SELECT ai_complete('Say hi', MAP('model', 'gpt-oss-120b', 'show_details', 'true'));
//! SELECT ai_classify(text, '["urgent", "not_urgent"]') AS urgency FROM docs;
//! SELECT ai_summarize(text, 20) AS summary FROM docs;
//! SELECT ai_analyze_sentiment(text) AS sentiment FROM docs;
//!
//! SET datafusion_ai_functions.decision_api = 'systemone';
//! SET datafusion_ai_functions.decision_endpoint = 'http://localhost:11434/v1';
//! SET datafusion_ai_functions.decision_model = 'nimble';
//! SELECT id FROM tickets WHERE ai_if(body, 'The customer asks for a refund');
//! SELECT ai_decide_classify(body, ['billing', 'technical'], MAP('on_error', 'null')) FROM tickets;
//! ```
//!
//! AI functions are endpoint bound, so evaluating them on many partitions
//! mostly multiplies requests in flight. [`AiSinglePartition`] merges their
//! input into one partition; physical optimizer rules can only be set while
//! building the session state, so use [`with_ai_optimizer_rules`] or
//! [`ai_session_context`]:
//!
//! ```no_run
//! use datafusion::execution::SessionStateBuilder;
//! use datafusion::prelude::{SessionConfig, SessionContext};
//! use datafusion_ai_functions::{register_ai_functions, with_ai_optimizer_rules};
//!
//! let builder = SessionStateBuilder::new()
//!     .with_config(SessionConfig::new())
//!     .with_default_features();
//! let ctx = SessionContext::new_with_state(with_ai_optimizer_rules(builder).build());
//! register_ai_functions(&ctx);
//! ```

mod config;
mod decision;
mod openai;
mod optimizer;
mod udf;

use std::sync::Arc;

use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::ScalarUDF;
use datafusion::logical_expr::async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::prelude::{SessionConfig, SessionContext};

pub use config::AiQueryConfig;
use openai::ClientCache;
pub use optimizer::AiSinglePartition;
pub use udf::{
    AiAnalyzeSentiment, AiClassify, AiComplete, AiDecide, AiDecideClassify, AiIf, AiProbability,
    AiQuery, AiScore, AiSummarize,
};

fn into_udf<T: AsyncScalarUDFImpl + 'static>(udf: T) -> ScalarUDF {
    AsyncScalarUDF::new(Arc::new(udf)).into_scalar_udf()
}

/// Create the `ai_query` scalar UDF.
pub fn ai_query_udf() -> ScalarUDF {
    into_udf(AiQuery::new())
}

/// Create the `ai_complete` scalar UDF.
pub fn ai_complete_udf() -> ScalarUDF {
    into_udf(AiComplete::new())
}

/// Create the `ai_classify` scalar UDF.
pub fn ai_classify_udf() -> ScalarUDF {
    into_udf(AiClassify::new())
}

/// Create the `ai_summarize` scalar UDF.
pub fn ai_summarize_udf() -> ScalarUDF {
    into_udf(AiSummarize::new())
}

/// Create the `ai_analyze_sentiment` scalar UDF.
pub fn ai_analyze_sentiment_udf() -> ScalarUDF {
    into_udf(AiAnalyzeSentiment::new())
}

/// Create the `ai_probability` scalar UDF.
pub fn ai_probability_udf() -> ScalarUDF {
    into_udf(AiProbability::new())
}

/// Create the `ai_if` scalar UDF.
pub fn ai_if_udf() -> ScalarUDF {
    into_udf(AiIf::new())
}

/// Create the `ai_score` scalar UDF.
pub fn ai_score_udf() -> ScalarUDF {
    into_udf(AiScore::new())
}

/// Create the `ai_decide_classify` scalar UDF.
pub fn ai_decide_classify_udf() -> ScalarUDF {
    into_udf(AiDecideClassify::new())
}

/// Create the `ai_decide` scalar UDF.
pub fn ai_decide_udf() -> ScalarUDF {
    into_udf(AiDecide::new())
}

/// Session config with the given `ai_query` options registered.
pub fn session_config_with_ai_query(cfg: AiQueryConfig) -> SessionConfig {
    SessionConfig::new().with_option_extension(cfg)
}

/// Register `ai_query` on the context. Also registers default `datafusion_ai_functions.*`
/// options if absent, so `SET datafusion_ai_functions.xxx = ...` works from SQL.
pub fn register_ai_query(ctx: &SessionContext) {
    register_config(ctx);
    ctx.register_udf(ai_query_udf());
}

/// Register all AI functions (`ai_query`, `ai_complete`, `ai_classify`,
/// `ai_summarize`, `ai_analyze_sentiment`, `ai_probability`, `ai_if`,
/// `ai_score`, `ai_decide_classify`, `ai_decide`) and the default
/// `datafusion_ai_functions.*` options (if absent) on the context. The functions share
/// one HTTP client (connection pool).
pub fn register_ai_functions(ctx: &SessionContext) {
    register_config(ctx);
    let clients = Arc::new(ClientCache::default());
    ctx.register_udf(into_udf(AiQuery::with_clients(Arc::clone(&clients))));
    ctx.register_udf(into_udf(AiComplete::with_clients(Arc::clone(&clients))));
    ctx.register_udf(into_udf(AiClassify::with_clients(Arc::clone(&clients))));
    ctx.register_udf(into_udf(AiSummarize::with_clients(Arc::clone(&clients))));
    ctx.register_udf(into_udf(AiAnalyzeSentiment::with_clients(Arc::clone(
        &clients,
    ))));
    ctx.register_udf(into_udf(AiProbability::with_clients(Arc::clone(&clients))));
    ctx.register_udf(into_udf(AiIf::with_clients(Arc::clone(&clients))));
    ctx.register_udf(into_udf(AiScore::with_clients(Arc::clone(&clients))));
    ctx.register_udf(into_udf(AiDecideClassify::with_clients(Arc::clone(
        &clients,
    ))));
    ctx.register_udf(into_udf(AiDecide::with_clients(clients)));
}

/// Add the [`AiSinglePartition`] rule to the builder's physical optimizer
/// rules (the defaults if none are set), just before `SanityCheckPlan` so the
/// rewritten plan is still validated.
pub fn with_ai_optimizer_rules(mut builder: SessionStateBuilder) -> SessionStateBuilder {
    let mut rules = builder
        .physical_optimizers()
        .take()
        .unwrap_or_default()
        .rules;
    let rule: Arc<dyn PhysicalOptimizerRule + Send + Sync> = Arc::new(AiSinglePartition::new());
    match rules.iter().position(|r| r.name() == "SanityCheckPlan") {
        Some(i) => rules.insert(i, rule),
        None => rules.push(rule),
    }
    builder.with_physical_optimizer_rules(rules)
}

/// Context with default features, the [`AiSinglePartition`] rule and all AI
/// functions registered.
pub fn ai_session_context(config: SessionConfig) -> SessionContext {
    let builder = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features();
    let ctx = SessionContext::new_with_state(with_ai_optimizer_rules(builder).build());
    register_ai_functions(&ctx);
    ctx
}

fn register_config(ctx: &SessionContext) {
    let state = ctx.state_ref();
    let mut state = state.write();
    let options = state.config_mut().options_mut();
    if options.extensions.get::<AiQueryConfig>().is_none() {
        options.extensions.insert(AiQueryConfig::default());
    }
}

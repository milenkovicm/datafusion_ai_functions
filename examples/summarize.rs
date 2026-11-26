//! `ai_summarize` examples against any OpenAI compatible endpoint, e.g. a local ollama:
//!
//! ollama pull qwen2.5:14b-instruct
//! cargo run --example summarize
//!
//! OPENAI_BASE_URL=http://localhost:11434/v1 AI_MODEL=qwen2.5:14b-instruct cargo run --example summarize

use datafusion::prelude::SessionContext;
use datafusion_ai_functions::register_ai_functions;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();
    register_ai_functions(&ctx);

    let endpoint =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let model = std::env::var("AI_MODEL").unwrap_or_else(|_| "qwen2.5:14b-instruct".into());

    ctx.sql(&format!(
        "SET datafusion_ai_functions.endpoint = '{endpoint}'"
    ))
    .await?;
    ctx.sql(&format!("SET datafusion_ai_functions.secret = '{token}'"))
        .await?;
    // ai_summarize has no model argument, it uses the session default model.
    ctx.sql(&format!(
        "SET datafusion_ai_functions.default_model = '{model}'"
    ))
    .await?;
    ctx.sql("SET datafusion_ai_functions.max_concurrency = 4")
        .await?;

    println!("\n== example (max_words = 20)");
    ctx.sql(
        "SELECT ai_summarize(
           'Apache Spark is a unified analytics engine for large-scale data processing. ' ||
           'It provides high-level APIs in Java, Scala, Python and R, and an optimized ' ||
           'engine that supports general execution graphs. It also supports a rich set ' ||
           'of higher-level tools including Spark SQL for SQL and structured data ' ||
           'processing, pandas API on Spark for pandas workloads, MLlib for machine ' ||
           'learning, GraphX for graph processing, and Structured Streaming for incremental ' ||
           'computation and stream processing.',
           20
         ) AS summary",
    )
    .await?
    .show()
    .await?;

    ctx.sql(
        "CREATE TABLE articles(id INT, max_words INT, text VARCHAR) AS VALUES
         (1, 10, 'Apache DataFusion is an extensible query engine written in Rust that uses Apache Arrow as its in-memory format. It offers SQL and DataFrame APIs, excellent performance, built-in support for CSV, Parquet, JSON and Avro, and extensive customization points.'),
         (2, 25, 'The Rust borrow checker enforces memory safety at compile time without a garbage collector. Every value has a single owner, references must not outlive the data they point to, and mutable references are exclusive, which rules out data races in safe code.'),
         (3, 0, 'Apache Arrow defines a language-independent columnar memory format for flat and hierarchical data, organized for efficient analytic operations on modern hardware like CPUs and GPUs.'),
         (4, 20, NULL)",
    )
    .await?;

    println!("\n== Per-row max_words from a column (0 = no limit)");
    ctx.sql(
        "SELECT id, max_words, ai_summarize(text, max_words) AS summary
         FROM articles ORDER BY id",
    )
    .await?
    .show()
    .await?;

    println!("\n== Default max_words (50)");
    ctx.sql("SELECT id, ai_summarize(text) AS summary FROM articles ORDER BY id")
        .await?
        .show()
        .await?;

    Ok(())
}

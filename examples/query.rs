//! Run against any OpenAI compatible endpoint, e.g. a local ollama:
//!
//! OPENAI_BASE_URL=http://localhost:11434/v1 AI_MODEL=qwen2.5:1.5b cargo run --example query

use datafusion::prelude::SessionContext;
use datafusion_ai_functions::register_ai_query;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();
    register_ai_query(&ctx);

    let endpoint =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let model = std::env::var("AI_MODEL").unwrap_or_else(|_| "qwen2.5:1.5b-instruct".into());

    ctx.sql(&format!(
        "SET datafusion_ai_functions.endpoint = '{endpoint}'"
    ))
    .await?;
    ctx.sql(&format!(
        "SET datafusion_ai_functions.api_token = '{token}'"
    ))
    .await?;
    ctx.sql(&format!(
        "SET datafusion_ai_functions.default_model = '{model}'"
    ))
    .await?;
    ctx.sql("SET datafusion_ai_functions.max_concurrency = 4")
        .await?;

    ctx.sql(
        "CREATE TABLE docs(text VARCHAR) AS VALUES
         ('Apache DataFusion is an extensible query engine written in Rust that uses Apache Arrow as its in-memory format.'),
         ('The Rust borrow checker enforces memory safety at compile time without a garbage collector.'),
         ('ai_query is a general-purpose AI Function that lets you query any supported AI model directly from SQL or Python. Unlike task-specific AI Functions, which are purpose-built and optimized for a single task, ai_query gives you full control over the model, prompt, and parameters.'),
         (NULL)",
    )
    .await?;

    // Plain text answer from `datafusion_ai_functions.default_model`.
    ctx.sql(
        "SELECT text, ai_query(
            'Summarize the given text in one short sentence: ' || text
         ) AS summary FROM docs",
    )
    .await?
    .show()
    .await?;

    // Structured output with per-call options.
    ctx.sql(
        r#"SELECT ai_query(
            'Extract the main topic and the programming language of the text: ' || text,
            MAP(
                'temperature', '0',
                'max_tokens', '100',
                'failOnError', 'false',
                'responseFormat', '{
                    "type": "json_schema",
                    "json_schema": {
                        "name": "topic",
                        "strict": true,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "language": {"type": ["string", "null"]}
                            },
                            "required": ["topic", "language"],
                            "additionalProperties": false
                        }
                    }
                }'
            )
         ) AS extracted FROM docs"#,
    )
    .await?
    .show()
    .await?;

    Ok(())
}

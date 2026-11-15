//! `ai_complete` examples against any OpenAI compatible endpoint, e.g. a local ollama:
//!
//! OPENAI_BASE_URL=http://localhost:11434/v1 AI_MODEL=qwen2.5:1.5b cargo run --example complete

use datafusion::prelude::SessionContext;
use datafusion_ai_functions::register_ai_functions;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();
    register_ai_functions(&ctx);

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

    ctx.sql(
        "CREATE TABLE notes(text VARCHAR) AS VALUES
         ('I ordered 3 items at 10, 20 and 5 dollars, ship to 1 Main St. Member since January 2024.'),
         ('Two books for 12.50 each, deliver to 42 Elm Road. I joined in March 2023.'),
         (NULL)",
    )
    .await?;

    // Plain text answer with model parameters.
    ctx.sql(
        "SELECT ai_complete(
            'How does a snowflake get its unique pattern? Answer in one sentence.',
            MAP('temperature', '0.7', 'max_tokens', '60')
         ) AS answer",
    )
    .await?
    .show()
    .await?;

    // Answer plus metadata (model, created, token usage).
    ctx.sql(
        "SELECT ai_complete(
            'Say hi',
            MAP('max_tokens', '10', 'show_details', 'true')
         ) AS details",
    )
    .await?
    .show()
    .await?;

    // Structured output with a JSON schema.
    ctx.sql(
        r#"SELECT ai_complete(
            'Extract the order details from this customer note: ' || text,
            MAP(
                'temperature', '0',
                'response_format', '{
                    "type": "json",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "items_count": {"type": "number"},
                            "price": {"type": "array", "items": {"type": "string"}},
                            "address": {"type": "string"},
                            "member_date": {"type": "string"}
                        },
                        "required": ["items_count", "price", "address", "member_date"]
                    }
                }'
            )
         ) AS extracted FROM notes"#,
    )
    .await?
    .show()
    .await?;

    Ok(())
}

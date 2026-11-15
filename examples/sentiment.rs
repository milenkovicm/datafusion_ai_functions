//! `ai_analyze_sentiment` examples against any OpenAI compatible endpoint, e.g. a local ollama:
//!
//! ollama pull qwen2.5:14b-instruct
//! cargo run --example sentiment
//!
//! OPENAI_BASE_URL=http://localhost:11434/v1 AI_MODEL=qwen2.5:14b-instruct cargo run --example sentiment

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
    ctx.sql(&format!(
        "SET datafusion_ai_functions.api_token = '{token}'"
    ))
    .await?;
    // ai_analyze_sentiment has no model argument, it uses the session default model.
    ctx.sql(&format!(
        "SET datafusion_ai_functions.default_model = '{model}'"
    ))
    .await?;
    ctx.sql("SET datafusion_ai_functions.max_concurrency = 4")
        .await?;

    println!("\n== ai_analyze_sentiment");
    ctx.sql("SELECT ai_analyze_sentiment('I am happy') AS sentiment")
        .await?
        .show()
        .await?;
    ctx.sql("SELECT ai_analyze_sentiment('I am sad') AS sentiment")
        .await?
        .show()
        .await?;

    ctx.sql(
        "CREATE TABLE reviews(id INT, review VARCHAR) AS VALUES
         (1, 'Absolutely love this product, it exceeded all my expectations!'),
         (2, 'Terrible support. I waited two weeks and nobody answered my ticket.'),
         (3, 'The package was delivered on Tuesday.'),
         (4, 'Great camera, but the battery barely lasts half a day.'),
         (5, NULL)",
    )
    .await?;

    println!("\n== Sentiment per row (NULL content gives NULL)");
    ctx.sql(
        "SELECT id, review, ai_analyze_sentiment(review) AS sentiment
         FROM reviews ORDER BY id",
    )
    .await?
    .show()
    .await?;

    println!("\n== Count reviews by sentiment");
    ctx.sql(
        // Async UDFs can't be evaluated inside an aggregate, so compute the
        // sentiment in a subquery and group on its output column.
        "SELECT sentiment, count(*) AS reviews
         FROM (SELECT ai_analyze_sentiment(review) AS sentiment FROM reviews)
         GROUP BY sentiment ORDER BY reviews DESC, sentiment",
    )
    .await?
    .show()
    .await?;

    Ok(())
}

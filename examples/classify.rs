//! `ai_classify` examples against any OpenAI compatible endpoint, e.g. a local ollama:
//!
//! ollama pull qwen2.5:14b-instruct
//! cargo run --example classify
//!
//! OPENAI_BASE_URL=http://localhost:11434/v1 AI_MODEL=qwen2.5:14b-instruct cargo run --example classify
//!
//! `AI_CLASSIFY_ALT_MODEL` (default `qwen2.5:1.5b`) is used by the query that
//! picks its model per call with `MAP('model', ...)`.

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
    let alt_model =
        std::env::var("AI_CLASSIFY_ALT_MODEL").unwrap_or_else(|_| "qwen2.5:1.5b".into());

    ctx.sql(&format!(
        "SET datafusion_ai_functions.endpoint = '{endpoint}'"
    ))
    .await?;
    ctx.sql(&format!("SET datafusion_ai_functions.secret = '{token}'"))
        .await?;
    // ai_classify has no model argument, it uses the session default model.
    ctx.sql(&format!(
        "SET datafusion_ai_functions.default_model = '{model}'"
    ))
    .await?;
    ctx.sql("SET datafusion_ai_functions.max_concurrency = 4")
        .await?;

    ctx.sql(
        "CREATE TABLE tickets(id INT, text VARCHAR) AS VALUES
         (1, 'My password is leaked and someone logged into my account.'),
         (2, 'I was charged twice for my last order, please refund one payment.'),
         (3, 'I want a refund, the blender arrived with a cracked jar.'),
         (4, 'It would be great if the app had a dark mode.'),
         (5, NULL)",
    )
    .await?;

    let queries = [
        (
            "example (version 2.1)",
            r#"SELECT ai_classify(
                 'My password is leaked.',
                 '["urgent", "not_urgent"]',
                 MAP('version', '2.1')
               ) AS result"#,
        ),
        (
            "JSON output with responseFormat",
            r#"SELECT id, ai_classify(
                 text,
                 '["urgent", "not_urgent"]',
                 MAP('responseFormat', 'json')
               ) AS urgency
               FROM tickets ORDER BY id"#,
        ),
        (
            "Single label over a table (plain label)",
            r#"SELECT id, text, ai_classify(text, '["urgent", "not_urgent"]') AS urgency
               FROM tickets ORDER BY id"#,
        ),
        (
            "Labels with descriptions and instructions",
            r#"SELECT id, ai_classify(
                 text,
                 '{"billing_issue": "payments, charges and refunds",
                   "product_defect": "damaged or malfunctioning products",
                   "account_security": "passwords, logins and compromised accounts",
                   "feature_request": "suggestions for new functionality"}',
                 MAP('instructions', 'Classify customer support tickets of an online shop.')
               ) AS category
               FROM tickets ORDER BY id"#,
        ),
        (
            "Multi-label",
            r#"SELECT id, ai_classify(
                 text,
                 '{"billing_issue": "payments, charges and refunds",
                   "product_defect": "damaged or malfunctioning products",
                   "shipping_issue": "late or lost deliveries"}',
                 MAP('multilabel', 'true')
               ) AS categories
               FROM tickets ORDER BY id"#,
        ),
        (
            "Version 2.0 output (plain label array)",
            r#"SELECT id, ai_classify(
                 text,
                 '["urgent", "not_urgent"]',
                 MAP('version', '2.0')
               ) AS urgency
               FROM tickets ORDER BY id"#,
        ),
    ];

    for (title, sql) in queries {
        println!("\n== {title}");
        ctx.sql(sql).await?.show().await?;
    }

    // `model` in the options overrides `datafusion_ai_functions.default_model` for this call.
    println!("\n== Per-call model ({alt_model}) with version 2.0");
    ctx.sql(&format!(
        r#"SELECT id, ai_classify(
             text,
             '["urgent", "not_urgent"]',
             MAP('model', '{alt_model}', 'version', '2.0')
           ) AS urgency
           FROM tickets ORDER BY id"#
    ))
    .await?
    .show()
    .await?;

    Ok(())
}

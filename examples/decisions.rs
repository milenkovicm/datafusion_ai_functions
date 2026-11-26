//! Decision function examples (`ai_probability`, `ai_if`, `ai_score`,
//! `ai_decide_classify`, `ai_decide`), by default against a local Ollama
//! serving the `nimble` decision model on its `/v1/systemone` endpoint:
//!
//! cargo run --example decisions
//!
//! Against OpenAI's `/v1/decisions`:
//!
//! AI_DECISION_API=openai AI_DECISION_ENDPOINT=https://api.openai.com/v1 \
//!   OPENAI_API_KEY=sk-... AI_DECISION_MODEL=gpt-6-luna cargo run --example decisions

use datafusion::prelude::SessionContext;
use datafusion_ai_functions::register_ai_functions;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();
    register_ai_functions(&ctx);

    let env = |k: &str, default: &str| std::env::var(k).unwrap_or_else(|_| default.into());
    let endpoint = env("AI_DECISION_ENDPOINT", "http://localhost:11434/v1");
    let api = env("AI_DECISION_API", "systemone");
    let model = env("AI_DECISION_MODEL", "nimble");
    let token = env("OPENAI_API_KEY", "");

    for (key, value) in [
        ("decision_endpoint", endpoint.as_str()),
        ("decision_api", api.as_str()),
        ("decision_model", model.as_str()),
        ("decision_secret", token.as_str()),
    ] {
        ctx.sql(&format!("SET datafusion_ai_functions.{key} = '{value}'"))
            .await?;
    }
    ctx.sql("SET datafusion_ai_functions.max_concurrency = 4")
        .await?;

    ctx.sql(
        "CREATE TABLE tickets(id INT, status VARCHAR, body VARCHAR) AS VALUES
         (1, 'open', 'You charged me twice this month, I want my money back now!'),
         (2, 'open', 'The app crashes every time I open the settings page.'),
         (3, 'closed', 'Thanks, the new dashboard looks great.'),
         (4, 'open', 'This is the third time I am asking. Cancel my subscription or I will dispute the charge.'),
         (5, 'open', NULL)",
    )
    .await?;

    println!("\n== ai_if: open tickets asking for a refund");
    ctx.sql(
        "SELECT id, body FROM tickets
         WHERE status = 'open' AND ai_if(body, 'The customer is asking for a refund')
         ORDER BY id",
    )
    .await?
    .show()
    .await?;

    println!("\n== ai_probability: churn risk");
    ctx.sql(
        "SELECT id, ai_probability(body, 'The customer threatens to cancel') AS churn_risk
         FROM tickets ORDER BY churn_risk DESC NULLS LAST",
    )
    .await?
    .show()
    .await?;

    println!("\n== ai_decide_classify: route to a team");
    ctx.sql(
        "SELECT id, ai_decide_classify(body, ['billing', 'technical', 'other'], MAP('instructions', 'Which team should handle this ticket?', 'on_error', 'null')) AS team
         FROM tickets ORDER BY id",
    )
    .await?
    .show()
    .await?;

    println!("\n== ai_score: frustration (0 = calm, 2 = furious)");
    ctx.sql(
        "SELECT id, ai_score(body, 'How frustrated is the customer?', ['calm', 'annoyed', 'furious']) AS frustration
         FROM tickets ORDER BY frustration DESC NULLS LAST",
    )
    .await?
    .show()
    .await?;

    println!("\n== ai_decide: several questions in one request");
    ctx.sql(
        r#"SELECT id,
                  d['team']['choice'] AS team,
                  round(d['churn']['probability'], 3) AS p_churn,
                  round(d['tone']['score'], 2) AS tone
           FROM (SELECT id, ai_decide(body, '{
                    "churn": {"type": "noul", "instructions": "The customer is likely to cancel"},
                    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                             "criteria": {"billing": "Payments, charges and refunds", "technical": "Bugs and crashes", "other": null}},
                    "tone": {"type": "score", "instructions": "How frustrated is the customer?",
                             "criteria": ["calm", "annoyed", "furious"]}
                 }') AS d FROM tickets)
           ORDER BY id"#,
    )
    .await?
    .show()
    .await?;

    Ok(())
}

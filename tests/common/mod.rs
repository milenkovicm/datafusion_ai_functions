#![allow(dead_code)]

use datafusion::arrow::array::{Array, AsArray};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_ai_functions::{ai_session_context, register_ai_functions};
use serde_json::{Value, json};
use wiremock::{MockServer, Request, Respond, ResponseTemplate};

fn completion(model: &str, content: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "x", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    }))
}

/// Echoes `"<model>: <last user message>"` back as a chat completion.
pub struct Echo;
impl Respond for Echo {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&req.body).unwrap();
        let model = body["model"].as_str().unwrap();
        let prompt = body["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_str()
            .unwrap();
        completion(model, &format!("{model}: {prompt}"))
    }
}

/// Answers with the last user message, unchanged.
pub struct Parrot;
impl Respond for Parrot {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&req.body).unwrap();
        let prompt = body["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_str()
            .unwrap();
        completion("m", prompt)
    }
}

/// Create table `r(id, txt)` = (1,'x'),(2,'y'),(3,'x'),(4,NULL),(5,'x'):
/// four non-NULL values, two of them distinct.
pub async fn create_repeated(ctx: &SessionContext) {
    ctx.sql(
        "CREATE TABLE r(id INT, txt VARCHAR) AS VALUES (1,'x'),(2,'y'),(3,'x'),(4,NULL),(5,'x')",
    )
    .await
    .unwrap();
}

/// Always answers with the given content.
pub struct Fixed(pub &'static str);
impl Respond for Fixed {
    fn respond(&self, _req: &Request) -> ResponseTemplate {
        completion("m", self.0)
    }
}

/// Context with all AI functions pointed at `server` and a table
/// `t(id, txt)` = (1,'a'),(2,NULL),(3,'c'),(4,'d'),(5,'e').
pub async fn ctx_for(server: &MockServer) -> SessionContext {
    let ctx = SessionContext::new();
    register_ai_functions(&ctx);
    setup(ctx, server).await
}

/// Like [`ctx_for`], but with 4 target partitions and the
/// `AiSinglePartition` rule installed.
pub async fn ctx_with_rule_for(server: &MockServer) -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(4);
    setup(ai_session_context(config), server).await
}

async fn setup(ctx: SessionContext, server: &MockServer) -> SessionContext {
    ctx.sql(&format!(
        "SET datafusion_ai_functions.endpoint = '{}/v1'",
        server.uri()
    ))
    .await
    .unwrap();
    ctx.sql("SET datafusion_ai_functions.secret = 'secret'")
        .await
        .unwrap();
    ctx.sql("SET datafusion_ai_functions.max_retries = 0")
        .await
        .unwrap();
    ctx.sql(
        "CREATE TABLE t(id INT, txt VARCHAR) AS VALUES (1,'a'),(2,NULL),(3,'c'),(4,'d'),(5,'e')",
    )
    .await
    .unwrap();
    ctx
}

/// Run `sql` and return its first column as strings.
pub async fn query_col(ctx: &SessionContext, sql: &str) -> Vec<Option<String>> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches
        .iter()
        .flat_map(|b| {
            let a = b.column(0).as_string::<i32>().clone();
            (0..a.len())
                .map(|i| a.is_valid(i).then(|| a.value(i).to_string()))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// JSON bodies of all requests the server received.
pub async fn request_bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|r| serde_json::from_slice(&r.body).unwrap())
        .collect()
}

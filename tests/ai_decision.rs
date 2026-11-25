mod common;

use common::{ctx_for, ctx_with_rule_for, request_bodies};
use datafusion::arrow::array::{Array, AsArray};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::DataType;
use datafusion::prelude::SessionContext;
use serde_json::{Value, json};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Noul probability: 0.9 if the input mentions "refund", else 0.1.
fn probability(input: &str) -> f64 {
    if input.contains("refund") { 0.9 } else { 0.1 }
}

/// Answers `/v1/systemone` requests: noul by [`probability`], choice with
/// the last option, score with the top level.
struct SystemOne;
impl Respond for SystemOne {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&req.body).unwrap();
        let state = body["state"].as_str().unwrap();
        if state.contains("refuse") {
            return ResponseTemplate::new(200).set_body_json(json!({"answers": {}}));
        }
        let mut answers = serde_json::Map::new();
        for (name, q) in body["questions"].as_object().unwrap() {
            let answer = match q["type"].as_str().unwrap() {
                "noul" => json!({"type": "noul", "noul": probability(state)}),
                "choice" => {
                    let options: Vec<&String> = q["criteria"].as_object().unwrap().keys().collect();
                    let last = options.last().unwrap();
                    let probs: serde_json::Map<String, Value> = options
                        .iter()
                        .map(|o| ((*o).clone(), json!(if o == last { 0.75 } else { 0.25 })))
                        .collect();
                    json!({"type": "choice", "choice": last, "probabilities": probs, "confidence": 0.5})
                }
                "score" => {
                    let n = q["criteria"].as_array().unwrap().len();
                    let probs: serde_json::Map<String, Value> = (0..n)
                        .map(|i| (i.to_string(), json!(if i == n - 1 { 1.0 } else { 0.0 })))
                        .collect();
                    json!({"type": "score", "score": (n - 1) as f64, "probabilities": probs, "confidence": 1.0})
                }
                t => panic!("unexpected type {t}"),
            };
            answers.insert(name.clone(), answer);
        }
        ResponseTemplate::new(200).set_body_json(json!({"model": body["model"], "answers": answers,
            "usage": {"input_tokens": 1, "output_tokens": 0}}))
    }
}

/// Answers OpenAI `/v1/decisions` requests like [`SystemOne`].
struct OpenAi;
impl Respond for OpenAi {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&req.body).unwrap();
        let input = body["input"].as_str().unwrap();
        let answers: Vec<Value> = body["questions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| {
                let name = &q["name"];
                if input.contains("refuse") {
                    return json!({"type": "refusal", "name": name});
                }
                match q["type"].as_str().unwrap() {
                    "predicate" => {
                        json!({"type": "predicate", "name": name, "probability": probability(input)})
                    }
                    "choice" => {
                        let choices = q["choices"].as_array().unwrap();
                        let last = &choices.last().unwrap()["value"];
                        let probs: Vec<Value> = choices
                            .iter()
                            .map(|c| json!({"value": c["value"], "probability": if &c["value"] == last { 0.75 } else { 0.25 }}))
                            .collect();
                        json!({"type": "choice", "name": name, "choice": last, "probabilities": probs, "confidence": 0.5})
                    }
                    "score" => {
                        let levels = q["levels"].as_array().unwrap();
                        let n = levels.len();
                        let probs: Vec<Value> = levels
                            .iter()
                            .enumerate()
                            .map(|(i, l)| json!({"value": i, "label": l["label"], "probability": if i == n - 1 { 1.0 } else { 0.0 }}))
                            .collect();
                        json!({"type": "score", "name": name, "score": (n - 1) as f64, "probabilities": probs, "confidence": 1.0})
                    }
                    t => panic!("unexpected type {t}"),
                }
            })
            .collect();
        // No `usage`: endpoints may leave it out.
        ResponseTemplate::new(200)
            .set_body_json(json!({"model": body["model"], "answers": answers}))
    }
}

async fn server(api: &str) -> (MockServer, SessionContext) {
    let server = MockServer::start().await;
    match api {
        "systemone" => {
            Mock::given(method("POST"))
                .and(path("/v1/systemone"))
                .respond_with(SystemOne)
                .mount(&server)
                .await
        }
        _ => {
            Mock::given(method("POST"))
                .and(path("/v1/decisions"))
                .respond_with(OpenAi)
                .mount(&server)
                .await
        }
    }
    // The single partition rule keeps the rows in one batch, so duplicate
    // inputs are deduplicated.
    let ctx = ctx_with_rule_for(&server).await;
    set(&ctx, "decision_api", api).await;
    set(&ctx, "decision_model", "nimble").await;
    ctx.sql("CREATE TABLE tickets(id INT, body VARCHAR) AS VALUES (1, 'I want a refund'), (2, 'app crashes'), (3, NULL), (4, 'I want a refund')")
        .await
        .unwrap();
    (server, ctx)
}

async fn set(ctx: &SessionContext, key: &str, value: &str) {
    ctx.sql(&format!("SET datafusion_ai_functions.{key} = '{value}'"))
        .await
        .unwrap();
}

async fn query_err(ctx: &SessionContext, sql: &str) -> String {
    match ctx.sql(sql).await {
        Err(e) => e.to_string(),
        Ok(df) => df.collect().await.unwrap_err().to_string(),
    }
}

/// Run `sql` and return its first column cast to strings.
async fn query_col(ctx: &SessionContext, sql: &str) -> Vec<Option<String>> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches
        .iter()
        .flat_map(|b| {
            let a = cast(b.column(0), &DataType::Utf8).unwrap();
            let a = a.as_string::<i32>();
            (0..a.len())
                .map(|i| a.is_valid(i).then(|| a.value(i).to_string()))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn strs(v: &[Option<&str>]) -> Vec<Option<String>> {
    v.iter().map(|s| s.map(String::from)).collect()
}

async fn check_all_functions(api: &str) {
    let (server, ctx) = server(api).await;
    let col = |sql: &'static str| {
        let ctx = ctx.clone();
        async move { query_col(&ctx, sql).await }
    };

    assert_eq!(
        col("SELECT CAST(ai_probability(body, 'asks for a refund') AS VARCHAR) FROM tickets ORDER BY id").await,
        strs(&[Some("0.9"), Some("0.1"), None, Some("0.9")])
    );
    assert_eq!(
        col("SELECT CAST(ai_if(body, 'asks for a refund') AS VARCHAR) FROM tickets ORDER BY id")
            .await,
        strs(&[Some("true"), Some("false"), None, Some("true")])
    );
    assert_eq!(
        col("SELECT CAST(id AS VARCHAR) AS id_text FROM tickets WHERE ai_if(body, 'asks for a refund') ORDER BY id").await,
        strs(&[Some("1"), Some("4")])
    );
    assert_eq!(
        col("SELECT CAST(ai_score(body, 'How angry?', ['calm', 'annoyed', 'furious']) AS VARCHAR) FROM tickets ORDER BY id").await,
        strs(&[Some("2.0"), Some("2.0"), None, Some("2.0")])
    );
    assert_eq!(
        col("SELECT ai_decide_classify(body, ['billing', 'technical']) FROM tickets ORDER BY id")
            .await,
        strs(&[
            Some("technical"),
            Some("technical"),
            None,
            Some("technical")
        ])
    );
    assert_eq!(
        col(r#"SELECT ai_decide_classify(body, '{"billing": "Payments", "other": null}', MAP('instructions', 'Which team?')) FROM tickets WHERE id = 1"#).await,
        strs(&[Some("other")])
    );
    assert_eq!(
        col(r#"SELECT CAST(d['refund']['probability'] AS VARCHAR) || ' ' || d['team']['choice'] || ' ' ||
                      CAST(d['tone']['score'] AS VARCHAR) || ' ' || CAST(d['team']['probabilities']['billing'] AS VARCHAR)
               FROM (SELECT id, ai_decide(body, '{
                   "refund": {"type": "noul", "instructions": "asks for a refund"},
                   "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": null, "technical": "Bugs"}},
                   "tone": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "furious"]}
               }') AS d FROM tickets) ORDER BY id"#).await,
        strs(&[Some("0.9 technical 1.0 0.25"), Some("0.1 technical 1.0 0.25"), None, Some("0.9 technical 1.0 0.25")])
    );

    // NULL inputs send nothing and duplicates are asked once: 2 distinct
    // inputs per call, 1 for the single row call. The `WHERE ai_if` call
    // sees the same batch as the `SELECT ai_if` call just before, so it
    // reuses its answers.
    let bodies = request_bodies(&server).await;
    assert_eq!(bodies.len(), 5 * 2 + 1, "{bodies:#?}");
    assert!(bodies.iter().all(|b| b["model"] == "nimble"));
}

#[tokio::test]
async fn systemone_functions() {
    check_all_functions("systemone").await;
}

#[tokio::test]
async fn openai_functions() {
    check_all_functions("openai").await;
}

#[tokio::test]
async fn systemone_request_shape() {
    let (server, ctx) = server("systemone").await;
    query_col(
        &ctx,
        r#"SELECT ai_decide('I want a refund', '{
            "r": {"type": "predicate", "instructions": "asks for a refund"},
            "t": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": null}},
            "s": {"type": "score", "instructions": "How angry?", "criteria": {"calm": null, "furious": "very angry"}}
        }')['t']['choice']"#,
    )
    .await;
    let body = &request_bodies(&server).await[0];
    assert_eq!(
        body,
        &json!({
            "model": "nimble",
            "state": "I want a refund",
            "questions": {
                "r": {"type": "noul", "instructions": "asks for a refund"},
                "t": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": null}},
                "s": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "furious: very angry"]}
            }
        })
    );
}

#[tokio::test]
async fn openai_request_shape() {
    let (server, ctx) = server("openai").await;
    query_col(
        &ctx,
        "SELECT ai_decide_classify('hi', ['billing', 'technical'], MAP('instructions', 'Which team?', 'model', 'gpt-6-luna'))",
    )
    .await;
    let body = &request_bodies(&server).await[0];
    assert_eq!(
        body,
        &json!({
            "model": "gpt-6-luna",
            "input": "hi",
            "questions": [{"type": "choice", "name": "label", "instructions": "Which team?",
                           "choices": [{"value": "billing"}, {"value": "technical"}]}]
        })
    );
}

#[tokio::test]
async fn options_map() {
    let (server, ctx) = server("systemone").await;
    let out = query_col(
        &ctx,
        "SELECT ai_decide_classify(body, ['a', 'b'], MAP('Model', 'other')) FROM tickets WHERE id = 1",
    )
    .await;
    assert_eq!(out, strs(&[Some("b")]));
    // Any subset of keys; a NULL value means the default model.
    let out = query_col(
        &ctx,
        "SELECT CAST(ai_if(body, 'refund', MAP('on_error', 'null', 'model', NULL)) AS VARCHAR) FROM tickets WHERE id = 1",
    )
    .await;
    assert_eq!(out, strs(&[Some("true")]));
    let models: Vec<Value> = request_bodies(&server)
        .await
        .iter()
        .map(|b| b["model"].clone())
        .collect();
    assert_eq!(models, [json!("other"), json!("nimble")]);
}

#[tokio::test]
async fn nested_input_is_sent_as_json() {
    let (server, ctx) = server("systemone").await;
    query_col(
        &ctx,
        "SELECT CAST(ai_if(named_struct('id', id, 'body', body), 'refund') AS VARCHAR) FROM tickets WHERE id = 1",
    )
    .await;
    assert_eq!(
        request_bodies(&server).await[0]["state"],
        r#"{"id":1,"body":"I want a refund"}"#
    );
}

#[tokio::test]
async fn refusals() {
    for api in ["systemone", "openai"] {
        let (_server, ctx) = server(api).await;
        let err = query_err(&ctx, "SELECT ai_if('please refuse', 'x')").await;
        assert!(
            err.contains("declined to answer question 'condition'"),
            "{api}: {err}"
        );
        let out = query_col(
            &ctx,
            "SELECT CAST(ai_probability('please refuse', 'x', MAP('on_error', 'null')) AS VARCHAR)",
        )
        .await;
        assert_eq!(out, vec![None], "{api}");
        // In ai_decide only the refused question is NULL.
        let out = query_col(
            &ctx,
            r#"SELECT CAST(d IS NULL AS VARCHAR) || ' ' || CAST(d['a'] IS NULL AS VARCHAR)
               FROM (SELECT ai_decide('please refuse', '{"a": {"type": "noul"}}', MAP('on_error', 'null')) AS d)"#,
        )
        .await;
        assert_eq!(out, strs(&[Some("false true")]), "{api}");
    }
}

#[tokio::test]
async fn request_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    set(&ctx, "decision_api", "systemone").await;
    set(&ctx, "decision_model", "nimble").await;
    assert!(!query_err(&ctx, "SELECT ai_if('x', 'y')").await.is_empty());
    let out = query_col(
        &ctx,
        "SELECT CAST(ai_if(txt, 'y', MAP('on_error', 'null')) AS VARCHAR) FROM t",
    )
    .await;
    assert_eq!(out, vec![None; 5]);
    // The session wide default applies when on_error isn't given.
    set(&ctx, "fail_on_error", "false").await;
    assert_eq!(
        query_col(&ctx, "SELECT CAST(ai_if('x', 'y') AS VARCHAR)").await,
        vec![None]
    );
}

#[tokio::test]
async fn decision_endpoint_and_token_override() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(header("authorization", "Bearer decision-secret"))
        .respond_with(SystemOne)
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    set(&ctx, "endpoint", "http://127.0.0.1:1/v1").await;
    set(&ctx, "decision_endpoint", &format!("{}/v1", server.uri())).await;
    set(&ctx, "decision_api_token", "decision-secret").await;
    set(&ctx, "decision_api", "systemone").await;
    set(&ctx, "decision_model", "nimble").await;
    let out = query_col(&ctx, "SELECT CAST(ai_if('refund', 'x') AS VARCHAR)").await;
    assert_eq!(out, strs(&[Some("true")]));
}

#[tokio::test]
async fn invalid_calls_fail_before_requests() {
    let (server, ctx) = server("systemone").await;
    for (sql, expected) in [
        (
            "SELECT ai_decide_classify(body, ['a']) FROM tickets",
            "labels must have 2 to 255 entries",
        ),
        (
            "SELECT ai_decide_classify(body, ['a', 'a']) FROM tickets",
            "duplicate 'a' in labels",
        ),
        (
            "SELECT ai_decide_classify(body, body) FROM tickets",
            "labels must be a constant",
        ),
        (
            "SELECT ai_if(body, body) FROM tickets",
            "condition must be a constant",
        ),
        (
            "SELECT ai_score(body, 'x', ['0','1','2','3','4','5','6','7','8','9','10']) FROM tickets",
            "levels must have 2 to 10 entries",
        ),
        (
            "SELECT ai_score(body, 'x', 3) FROM tickets",
            "levels must be a list of strings or a JSON string",
        ),
        (
            "SELECT ai_decide(body, 'nope') FROM tickets",
            "questions must be a JSON object",
        ),
        (
            "SELECT ai_decide(body, body) FROM tickets",
            "questions must be a constant",
        ),
        (
            "SELECT ai_if(body, 'x', MAP('on_error', 'skip')) FROM tickets",
            "on_error must be 'fail' or 'null'",
        ),
        (
            "SELECT ai_if(body) FROM tickets",
            "ai_if expects 2 or 3 arguments",
        ),
        (
            "SELECT ai_if(body, 'x', MAP('nope', 'y')) FROM tickets",
            "unknown option 'nope', expected one of: model, on_error",
        ),
        (
            "SELECT ai_if(body, 'x', MAP('instructions', 'y')) FROM tickets",
            "unknown option 'instructions'",
        ),
        (
            "SELECT ai_if(body, 'x', 'nimble') FROM tickets",
            "options must be a MAP<STRING, STRING>",
        ),
        (
            "SELECT ai_if(CAST('x' AS BYTEA), 'x') FROM tickets",
            "input must be text",
        ),
        (
            "SELECT ai_if(body, NULL) FROM tickets",
            "condition must not be NULL or empty",
        ),
    ] {
        let err = query_err(&ctx, sql).await;
        assert!(err.contains(expected), "{sql}: {err}");
    }
    set(&ctx, "decision_model", "").await;
    let err = query_err(&ctx, "SELECT ai_if(body, 'x') FROM tickets").await;
    assert!(err.contains("no decision model configured"), "{err}");
    set(&ctx, "decision_api", "nope").await;
    let err = query_err(
        &ctx,
        "SELECT ai_if(body, 'x', MAP('model', 'm')) FROM tickets",
    )
    .await;
    assert!(
        err.contains("decision_api must be 'openai' or 'systemone'"),
        "{err}"
    );
    assert!(request_bodies(&server).await.is_empty());
}

#[tokio::test]
async fn ai_decide_field_accesses_share_requests() {
    let (server, ctx) = server("systemone").await;
    // DataFusion copies `ai_decide` once per `d[...]`; the copies must not
    // send the requests again.
    let out = query_col(
        &ctx,
        r#"SELECT CAST(d['a']['probability'] AS VARCHAR) || ' ' || CAST(d['b']['probability'] AS VARCHAR)
           FROM (SELECT id, ai_decide(body, '{"a": {"type": "noul"}, "b": {"type": "noul"}}') AS d FROM tickets)
           ORDER BY id"#,
    )
    .await;
    assert_eq!(
        out,
        strs(&[Some("0.9 0.9"), Some("0.1 0.1"), None, Some("0.9 0.9")])
    );
    assert_eq!(request_bodies(&server).await.len(), 2);
}

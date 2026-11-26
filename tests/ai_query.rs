mod common;

use common::{Echo, ctx_for, query_col, request_bodies};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::SessionContext;
use datafusion_ai_functions::{AiQueryConfig, ai_query_udf, session_config_with_ai_query};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn echo_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Echo)
        .mount(&server)
        .await;
    server
}

async fn failing_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;
    server
}

async fn query_err(ctx: &SessionContext, sql: &str) -> String {
    let result = match ctx.sql(sql).await {
        Ok(df) => df.collect().await.map(|_| ()),
        Err(e) => Err(e),
    };
    result.expect_err(sql).to_string()
}

#[tokio::test]
async fn returns_rows_in_order_and_skips_nulls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer secret"))
        .respond_with(Echo)
        .expect(4) // the NULL row must not hit the server
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm1'")
        .await
        .unwrap();
    ctx.sql("SET datafusion_ai_functions.max_concurrency = 1")
        .await
        .unwrap();

    let out = query_col(&ctx, "SELECT ai_query('p:' || txt) FROM t ORDER BY id").await;
    assert_eq!(
        out,
        vec![
            Some("m1: p:a".into()),
            None,
            Some("m1: p:c".into()),
            Some("m1: p:d".into()),
            Some("m1: p:e".into())
        ]
    );
}

#[tokio::test]
async fn model_option_overrides_default() {
    let server = echo_server().await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm1'")
        .await
        .unwrap();

    let out = query_col(&ctx, "SELECT ai_query('hi', MAP('model', 'm2'))").await;
    assert_eq!(out, vec![Some("m2: hi".into())]);

    // Blank or NULL model falls back to the default.
    let out = query_col(&ctx, "SELECT ai_query('hi', MAP('model', ''))").await;
    assert_eq!(out, vec![Some("m1: hi".into())]);
    let out = query_col(&ctx, "SELECT ai_query('hi', NULL)").await;
    assert_eq!(out, vec![Some("m1: hi".into())]);
}

#[tokio::test]
async fn per_row_model_from_column() {
    let server = echo_server().await;
    let ctx = ctx_for(&server).await;

    let out = query_col(
        &ctx,
        "SELECT ai_query(txt, MAP('model', CASE WHEN id < 3 THEN 'x' ELSE 'y' END))
         FROM t ORDER BY id",
    )
    .await;
    assert_eq!(
        out,
        vec![
            Some("x: a".into()),
            None,
            Some("y: c".into()),
            Some("y: d".into()),
            Some("y: e".into())
        ]
    );
}

#[tokio::test]
async fn model_parameters_and_response_format_are_sent() {
    let server = echo_server().await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();
    ctx.sql("SET datafusion_ai_functions.max_tokens = 10")
        .await
        .unwrap();

    query_col(
        &ctx,
        r#"SELECT ai_query('hi', MAP(
            'max_tokens', '64',
            'temperature', '0.25',
            'responseFormat', '{"type": "json_schema", "json_schema": {"name": "p", "strict": true, "schema": {"type": "object"}}}'
        ))"#,
    )
    .await;
    query_col(
        &ctx,
        "SELECT ai_query('hi', MAP('responseFormat', 'json_object'))",
    )
    .await;
    query_col(&ctx, "SELECT ai_query('hi')").await;

    let bodies = request_bodies(&server).await;
    assert_eq!(bodies.len(), 3);
    assert_eq!(bodies[0]["max_completion_tokens"], 64);
    assert_eq!(bodies[0]["temperature"], 0.25);
    assert_eq!(bodies[0]["response_format"]["type"], "json_schema");
    assert_eq!(bodies[0]["response_format"]["json_schema"]["name"], "p");
    assert_eq!(bodies[0]["response_format"]["json_schema"]["strict"], true);

    assert_eq!(bodies[1]["response_format"]["type"], "json_object");
    assert_eq!(bodies[1]["max_completion_tokens"], 10); // config default

    assert!(bodies[2].get("response_format").is_none());
    assert!(bodies[2].get("temperature").is_none());
}

#[tokio::test]
async fn fail_on_error_toggle() {
    let server = failing_server().await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();

    query_err(&ctx, "SELECT ai_query(txt) FROM t").await;

    ctx.sql("SET datafusion_ai_functions.fail_on_error = false")
        .await
        .unwrap();
    let out = query_col(&ctx, "SELECT ai_query(txt) FROM t").await;
    assert_eq!(out.len(), 5);
    assert!(out.iter().all(Option::is_none));
}

#[tokio::test]
async fn fail_on_error_option_overrides_config() {
    let server = failing_server().await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();

    // config: fail, option: don't fail
    let out = query_col(
        &ctx,
        "SELECT ai_query(txt, MAP('failOnError', 'false')) FROM t",
    )
    .await;
    assert_eq!(out.len(), 5);
    assert!(out.iter().all(Option::is_none));

    // config: don't fail, option: fail
    ctx.sql("SET datafusion_ai_functions.fail_on_error = false")
        .await
        .unwrap();
    query_err(
        &ctx,
        "SELECT ai_query(txt, MAP('failonerror', 'true')) FROM t",
    )
    .await;
}

#[tokio::test]
async fn first_error_stops_sending_requests() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .expect(1)
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();
    ctx.sql("SET datafusion_ai_functions.max_concurrency = 1")
        .await
        .unwrap();

    query_err(&ctx, "SELECT ai_query(txt) FROM t").await;
}

#[tokio::test]
async fn literal_options_apply_to_every_row() {
    let server = echo_server().await;
    let ctx = ctx_for(&server).await;

    let out = query_col(
        &ctx,
        "SELECT ai_query(txt, MAP('model', 'lit')) FROM t ORDER BY id",
    )
    .await;
    assert_eq!(
        out,
        vec![
            Some("lit: a".into()),
            None,
            Some("lit: c".into()),
            Some("lit: d".into()),
            Some("lit: e".into())
        ]
    );
}

#[tokio::test]
async fn invalid_calls_fail() {
    let server = echo_server().await;
    let ctx = ctx_for(&server).await;

    let err = query_err(&ctx, "SELECT ai_query('hi')").await;
    assert!(err.contains("no model configured"), "{err}");

    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();
    let err = query_err(&ctx, "SELECT ai_query('hi', MAP('nope', 'x'))").await;
    assert!(err.contains("unknown option 'nope'"), "{err}");
    let err = query_err(&ctx, "SELECT ai_query('hi', MAP('max_tokens', 'lots'))").await;
    assert!(err.contains("max_tokens"), "{err}");

    // The old `ai_query(model, prompt)` form is rejected at planning.
    let err = query_err(&ctx, "SELECT ai_query('m', 'hi')").await;
    assert!(err.contains("options must be a MAP"), "{err}");
    let err = query_err(&ctx, "SELECT ai_query('a', MAP('model', 'm'), 'b')").await;
    assert!(err.contains("ai_query"), "{err}"); // rejected by the signature

    assert!(request_bodies(&server).await.is_empty());
}

#[tokio::test]
async fn programmatic_config_and_system_prompt() {
    let server = echo_server().await;
    let mut cfg = AiQueryConfig::default();
    cfg.endpoint = format!("{}/v1", server.uri());
    cfg.system_prompt = "be brief".into();
    cfg.default_model = "m".into();
    let ctx = SessionContext::new_with_config(session_config_with_ai_query(cfg));
    ctx.register_udf(ai_query_udf());

    let out = query_col(&ctx, "SELECT ai_query('hi')").await;
    assert_eq!(out, vec![Some("m: hi".into())]);

    let body = &request_bodies(&server).await[0];
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "be brief");
}

#[tokio::test]
async fn plan_uses_async_exec() {
    let server = MockServer::start().await;
    let ctx = ctx_for(&server).await;
    let batches = ctx
        .sql("EXPLAIN SELECT ai_query(txt) FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let plan = pretty_format_batches(&batches).unwrap().to_string();
    assert!(plan.contains("AsyncFuncExec"), "{plan}");
}

fn length_response(content: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "length",
            "message": {"role": "assistant", "content": content}}]
    }))
}

#[tokio::test]
async fn empty_answer_at_token_limit_is_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(length_response(""))
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();

    let err = query_err(&ctx, "SELECT ai_query('hi')").await;
    assert!(
        err.contains("hit the token limit before answering"),
        "{err}"
    );
    let out = query_col(&ctx, "SELECT ai_query('hi', MAP('failOnError', 'false'))").await;
    assert_eq!(out, vec![None]);
}

#[tokio::test]
async fn cut_off_answer_is_kept() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(length_response("Once upon a"))
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();
    let out = query_col(&ctx, "SELECT ai_query('hi')").await;
    assert_eq!(out, vec![Some("Once upon a".to_string())]);
}

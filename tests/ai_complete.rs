mod common;

use common::{Echo, Fixed, ctx_for, query_col, request_bodies};
use datafusion::prelude::SessionContext;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Respond, ResponseTemplate};

async fn server_with(responder: impl Respond + 'static) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(responder)
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
        .respond_with(Echo)
        .expect(5) // 4 rows (the NULL row must not hit the server) + 1
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm1'")
        .await
        .unwrap();

    let out = query_col(&ctx, "SELECT ai_complete(txt) FROM t ORDER BY id").await;
    assert_eq!(
        out,
        vec![
            Some("m1: a".into()),
            None,
            Some("m1: c".into()),
            Some("m1: d".into()),
            Some("m1: e".into())
        ]
    );

    let out = query_col(&ctx, "SELECT ai_complete('hi', MAP('model', 'm2'))").await;
    assert_eq!(out, vec![Some("m2: hi".into())]);
}

#[tokio::test]
async fn model_parameters_and_response_format_are_sent() {
    let server = server_with(Echo).await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();

    query_col(
        &ctx,
        r#"SELECT ai_complete('hi', MAP(
            'temperature', '0.7',
            'top_p', '0.5',
            'max_tokens', '10',
            'response_format', '{"type": "json", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}'
        ))"#,
    )
    .await;
    query_col(&ctx, "SELECT ai_complete('hi')").await;

    let bodies = request_bodies(&server).await;
    assert_eq!(bodies[0]["temperature"], 0.7);
    assert_eq!(bodies[0]["top_p"], 0.5);
    assert_eq!(bodies[0]["max_completion_tokens"], 10);
    assert_eq!(bodies[0]["response_format"]["type"], "json_schema");
    assert_eq!(
        bodies[0]["response_format"]["json_schema"]["name"],
        "response"
    );
    assert_eq!(
        bodies[0]["response_format"]["json_schema"]["schema"]["properties"]["a"]["type"],
        "string"
    );

    for key in [
        "temperature",
        "top_p",
        "max_completion_tokens",
        "response_format",
    ] {
        assert!(bodies[1].get(key).is_none(), "{key}");
    }
}

#[tokio::test]
async fn show_details() {
    let server = server_with(Fixed(r#"{"a": "x"}"#)).await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();

    let parse = |s: &Option<String>| serde_json::from_str::<Value>(s.as_ref().unwrap()).unwrap();
    let usage = json!({"completion_tokens": 2, "prompt_tokens": 3, "total_tokens": 5});

    let out = query_col(
        &ctx,
        "SELECT ai_complete('hi', MAP('show_details', 'true'))",
    )
    .await;
    assert_eq!(
        parse(&out[0]),
        json!({"choices": [{"messages": r#"{"a": "x"}"#}], "created": 0, "model": "m", "usage": usage})
    );

    let out = query_col(
        &ctx,
        r#"SELECT ai_complete('hi', MAP('show_details', 'true',
            'response_format', '{"type": "json", "schema": {"type": "object"}}'))"#,
    )
    .await;
    assert_eq!(
        parse(&out[0]),
        json!({"structured_output": [{"raw_message": {"a": "x"}, "type": "json"}],
               "created": 0, "model": "m", "usage": usage})
    );

    // Without show_details the content is returned as is.
    let out = query_col(
        &ctx,
        "SELECT ai_complete('hi', MAP('show_details', 'false'))",
    )
    .await;
    assert_eq!(out, vec![Some(r#"{"a": "x"}"#.into())]);
}

#[tokio::test]
async fn fail_on_error_option() {
    let server = server_with(ResponseTemplate::new(500).set_body_string("boom")).await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();

    query_err(&ctx, "SELECT ai_complete(txt) FROM t").await;
    let out = query_col(
        &ctx,
        "SELECT ai_complete(txt, MAP('fail_on_error', 'false')) FROM t",
    )
    .await;
    assert_eq!(out.len(), 5);
    assert!(out.iter().all(Option::is_none));
}

#[tokio::test]
async fn invalid_calls_fail() {
    let server = server_with(Echo).await;
    let ctx = ctx_for(&server).await;

    let err = query_err(&ctx, "SELECT ai_complete('hi')").await;
    assert!(err.contains("no model configured"), "{err}");

    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();
    let err = query_err(&ctx, "SELECT ai_complete('hi', MAP('guardrails', 'true'))").await;
    assert!(err.contains("unknown option 'guardrails'"), "{err}");
    let err = query_err(&ctx, "SELECT ai_complete('hi', MAP('top_p', '2'))").await;
    assert!(err.contains("top_p"), "{err}");

    // positional `(model, prompt)` form is rejected at planning.
    let err = query_err(&ctx, "SELECT ai_complete('m', 'hi')").await;
    assert!(err.contains("options must be a MAP"), "{err}");

    assert!(request_bodies(&server).await.is_empty());
}

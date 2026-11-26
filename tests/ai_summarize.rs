mod common;

use common::{Echo, Fixed, ctx_for, query_col, request_bodies};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::SessionContext;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn summarize_server(answer: &'static str) -> (MockServer, SessionContext) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Fixed(answer))
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'qwen2.5:14b-instruct'")
        .await
        .unwrap();
    (server, ctx)
}

#[tokio::test]
async fn summarizes_with_max_words() {
    let (server, ctx) = summarize_server("  \"Spark is a fast analytics engine.\"\n").await;
    let out = query_col(
        &ctx,
        "SELECT ai_summarize('Apache Spark is a unified analytics engine ' || 'for large-scale data processing.', 20)",
    )
    .await;
    assert_eq!(out, vec![Some("Spark is a fast analytics engine.".into())]);

    let body = &request_bodies(&server).await[0];
    assert_eq!(body["model"], "qwen2.5:14b-instruct");
    assert_eq!(body["messages"][0]["role"], "system");
    let system = body["messages"][0]["content"].as_str().unwrap();
    assert!(system.contains("at most 20 words"), "{system}");
    assert_eq!(
        body["messages"][1]["content"],
        "Apache Spark is a unified analytics engine for large-scale data processing."
    );
}

#[tokio::test]
async fn default_and_unlimited_max_words() {
    let (server, ctx) = summarize_server("s").await;
    query_col(&ctx, "SELECT ai_summarize('x')").await;
    query_col(&ctx, "SELECT ai_summarize('x', 0)").await;
    query_col(&ctx, "SELECT ai_summarize('x', CAST(NULL AS INT))").await;
    let prompts: Vec<String> = request_bodies(&server)
        .await
        .iter()
        .map(|b| b["messages"][0]["content"].as_str().unwrap().to_string())
        .collect();
    assert!(prompts[0].contains("at most 50 words"), "{}", prompts[0]);
    assert!(!prompts[1].contains("at most"), "{}", prompts[1]);
    assert!(prompts[2].contains("at most 50 words"), "{}", prompts[2]);
}

#[tokio::test]
async fn null_content_sends_no_request_and_keeps_order() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(Echo)
        .expect(4)
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();

    let out = query_col(&ctx, "SELECT ai_summarize(txt, id) FROM t ORDER BY id").await;
    assert_eq!(
        out,
        vec![
            Some("m: a".into()),
            None,
            Some("m: c".into()),
            Some("m: d".into()),
            Some("m: e".into())
        ]
    );
}

#[tokio::test]
async fn invalid_arguments_fail() {
    let (_server, ctx) = summarize_server("s").await;
    let res = ctx
        .sql("SELECT ai_summarize('x', -1)")
        .await
        .unwrap()
        .collect()
        .await;
    assert!(res.is_err());
    assert!(ctx.sql("SELECT ai_summarize('x', 'ten')").await.is_err());
    assert!(ctx.sql("SELECT ai_summarize()").await.is_err());

    ctx.sql("SET datafusion_ai_functions.default_model = ''")
        .await
        .unwrap();
    let err = ctx
        .sql("SELECT ai_summarize('x')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("datafusion_ai_functions.default_model"),
        "{err}"
    );
}

#[tokio::test]
async fn request_error_respects_fail_on_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();

    assert!(
        ctx.sql("SELECT ai_summarize('x')")
            .await
            .unwrap()
            .collect()
            .await
            .is_err()
    );
    ctx.sql("SET datafusion_ai_functions.fail_on_error = false")
        .await
        .unwrap();
    assert_eq!(
        query_col(&ctx, "SELECT ai_summarize('x')").await,
        vec![None]
    );
}

#[tokio::test]
async fn plan_uses_async_exec() {
    let (_server, ctx) = summarize_server("s").await;
    let batches = ctx
        .sql("EXPLAIN SELECT ai_summarize(txt) FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let plan = pretty_format_batches(&batches).unwrap().to_string();
    assert!(plan.contains("AsyncFuncExec"), "{plan}");
}

#[tokio::test]
async fn identical_contents_are_asked_once() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(common::Parrot)
        .expect(3)
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();
    common::create_repeated(&ctx).await;

    // Rows 1 and 3 share content and limit; row 5 has the same content but
    // another limit, so it's asked separately.
    let out = query_col(
        &ctx,
        "SELECT ai_summarize(txt, CASE WHEN id = 5 THEN 10 ELSE 50 END) FROM r ORDER BY id",
    )
    .await;
    let (x, y) = (Some("x".to_string()), Some("y".to_string()));
    assert_eq!(out, vec![x.clone(), y, x.clone(), None, x]);
    let limits: Vec<bool> = request_bodies(&server)
        .await
        .iter()
        .map(|b| {
            b["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("at most 10 words")
        })
        .collect();
    assert_eq!(limits.iter().filter(|l| **l).count(), 1, "{limits:?}");
}

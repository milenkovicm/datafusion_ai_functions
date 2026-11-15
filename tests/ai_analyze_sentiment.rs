mod common;

use common::{Fixed, ctx_for, query_col, request_bodies};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::SessionContext;
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn sentiment_server(answer: &'static str) -> (MockServer, SessionContext) {
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
async fn examples() {
    let server = MockServer::start().await;
    for (text, answer) in [("I am happy", "positive"), ("I am sad", " Negative.\n")] {
        Mock::given(method("POST"))
            .and(body_string_contains(text))
            .respond_with(Fixed(answer))
            .mount(&server)
            .await;
    }
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();

    let happy = query_col(&ctx, "SELECT ai_analyze_sentiment('I am happy')").await;
    assert_eq!(happy, vec![Some("positive".into())]);
    let sad = query_col(&ctx, "SELECT ai_analyze_sentiment('I am sad')").await;
    assert_eq!(sad, vec![Some("negative".into())]);

    let body = &request_bodies(&server).await[0];
    assert_eq!(body["model"], "m");
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["messages"][0]["role"], "system");
    let system = body["messages"][0]["content"].as_str().unwrap();
    assert!(
        system.contains("positive, negative, neutral, mixed"),
        "{system}"
    );
    assert_eq!(body["messages"][1]["content"], "I am happy");
}

#[tokio::test]
async fn undetectable_sentiment_is_null() {
    for answer in ["unknown", "I am not sure", ""] {
        let (_server, ctx) = sentiment_server(answer).await;
        let out = query_col(&ctx, "SELECT ai_analyze_sentiment('asdf qwer')").await;
        assert_eq!(out, vec![None], "{answer}");
    }
}

#[tokio::test]
async fn null_content_sends_no_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(Fixed("mixed"))
        .expect(4)
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();

    let out = query_col(&ctx, "SELECT ai_analyze_sentiment(txt) FROM t ORDER BY id").await;
    let mixed = Some("mixed".to_string());
    assert_eq!(
        out,
        vec![mixed.clone(), None, mixed.clone(), mixed.clone(), mixed]
    );
    assert_eq!(
        query_col(&ctx, "SELECT ai_analyze_sentiment(NULL)").await,
        vec![None]
    );
}

#[tokio::test]
async fn invalid_arguments_fail() {
    let (_server, ctx) = sentiment_server("positive").await;
    assert!(ctx.sql("SELECT ai_analyze_sentiment(1)").await.is_err());
    assert!(ctx.sql("SELECT ai_analyze_sentiment()").await.is_err());
    assert!(
        ctx.sql("SELECT ai_analyze_sentiment('a', 'b')")
            .await
            .is_err()
    );

    ctx.sql("SET datafusion_ai_functions.default_model = ''")
        .await
        .unwrap();
    let err = ctx
        .sql("SELECT ai_analyze_sentiment('x')")
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
        ctx.sql("SELECT ai_analyze_sentiment('x')")
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
        query_col(&ctx, "SELECT ai_analyze_sentiment('x')").await,
        vec![None]
    );
}

#[tokio::test]
async fn plan_uses_async_exec() {
    let (_server, ctx) = sentiment_server("neutral").await;
    let batches = ctx
        .sql("EXPLAIN SELECT ai_analyze_sentiment(txt) FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let plan = pretty_format_batches(&batches).unwrap().to_string();
    assert!(plan.contains("AsyncFuncExec"), "{plan}");
}

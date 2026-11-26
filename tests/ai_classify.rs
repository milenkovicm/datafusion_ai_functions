mod common;

use common::{Fixed, ctx_for, query_col, request_bodies};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::SessionContext;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn classify_server(answer: &'static str) -> (MockServer, SessionContext) {
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

async fn query_json(ctx: &SessionContext, sql: &str) -> Vec<Option<Value>> {
    query_col(ctx, sql)
        .await
        .into_iter()
        .map(|v| v.map(|s| serde_json::from_str(&s).unwrap()))
        .collect()
}

fn system_prompt(body: &Value) -> &str {
    assert_eq!(body["messages"][0]["role"], "system");
    body["messages"][0]["content"].as_str().unwrap()
}

#[tokio::test]
async fn example_v2_1() {
    let (server, ctx) = classify_server("urgent").await;
    let out = query_col(
        &ctx,
        r#"SELECT ai_classify('My password is leaked.', '["urgent", "not_urgent"]', MAP('version', '2.1'))"#,
    )
    .await;
    assert_eq!(
        out,
        vec![Some(
            r#"{"response":[{"value":"urgent"}],"metadata":{"version":"2.1"},"error_message":null}"#
                .into()
        )]
    );

    let bodies = request_bodies(&server).await;
    assert_eq!(bodies[0]["model"], "qwen2.5:14b-instruct");
    assert_eq!(bodies[0]["temperature"], 0.0);
    assert_eq!(
        bodies[0]["messages"][1]["content"],
        "My password is leaked."
    );
    let prompt = system_prompt(&bodies[0]);
    assert!(prompt.contains("- urgent\n- not_urgent\n"), "{prompt}");
}

#[tokio::test]
async fn plain_label_without_options() {
    let (_server, ctx) = classify_server("not_urgent").await;
    for sql in [
        r#"SELECT ai_classify('hello', '["urgent", "not_urgent"]')"#,
        r#"SELECT ai_classify('hello', '["urgent", "not_urgent"]', MAP('responseFormat', 'text'))"#,
        r#"SELECT ai_classify('hello', '["urgent", "not_urgent"]', MAP('responseFormat', 'text', 'version', '2.0'))"#,
    ] {
        assert_eq!(
            query_col(&ctx, sql).await,
            vec![Some("not_urgent".into())],
            "{sql}"
        );
    }
}

#[tokio::test]
async fn json_default_version_and_v2_0() {
    let (_server, ctx) = classify_server("not_urgent").await;
    let out = query_json(
        &ctx,
        r#"SELECT ai_classify('hello', '["urgent", "not_urgent"]', MAP('responseFormat', 'json'))"#,
    )
    .await;
    assert_eq!(out[0].as_ref().unwrap()["metadata"]["version"], "2.1");

    let out = query_json(
        &ctx,
        r#"SELECT ai_classify('hello', '["urgent", "not_urgent"]', MAP('version', '2.0'))"#,
    )
    .await;
    assert_eq!(
        out,
        vec![Some(json!({
            "response": ["not_urgent"], "metadata": {"version": "2.0"}, "error_message": null
        }))]
    );
}

#[tokio::test]
async fn multilabel_with_descriptions_and_instructions() {
    let (server, ctx) = classify_server("product_defect\nbilling_issue").await;
    let out = query_json(
        &ctx,
        r#"SELECT ai_classify(
             'I want a refund, the product arrived broken.',
             '{"billing_issue": "payments and refunds", "product_defect": "broken items", "shipping_issue": "late delivery"}',
             MAP('multilabel', 'true', 'instructions', 'Customer support tickets'))"#,
    )
    .await;
    assert_eq!(
        out[0].as_ref().unwrap()["response"],
        json!([{"value": "billing_issue"}, {"value": "product_defect"}])
    );

    let bodies = request_bodies(&server).await;
    let prompt = system_prompt(&bodies[0]);
    assert!(prompt.contains("one or more"), "{prompt}");
    assert!(
        prompt.contains("- billing_issue: payments and refunds\n"),
        "{prompt}"
    );
    assert!(
        prompt.contains("Additional instructions: Customer support tickets"),
        "{prompt}"
    );
}

#[tokio::test]
async fn model_option_overrides_default_model() {
    let (server, ctx) = classify_server("a").await;
    query_col(
        &ctx,
        r#"SELECT ai_classify('x', '["a", "b"]', MAP('model', 'other-model'))"#,
    )
    .await;
    assert_eq!(request_bodies(&server).await[0]["model"], "other-model");
}

#[tokio::test]
async fn missing_model_fails() {
    let (_server, ctx) = classify_server("a").await;
    ctx.sql("SET datafusion_ai_functions.default_model = ''")
        .await
        .unwrap();
    let err = ctx
        .sql(r#"SELECT ai_classify('x', '["a", "b"]')"#)
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
async fn unknown_answer_sets_error_message() {
    let (_server, ctx) = classify_server("I think it is probably fine").await;
    let plain = query_col(
        &ctx,
        r#"SELECT ai_classify('x', '["urgent", "not_urgent"]')"#,
    )
    .await;
    assert_eq!(plain, vec![None]);

    let out = query_json(
        &ctx,
        r#"SELECT ai_classify('x', '["urgent", "not_urgent"]', MAP('responseFormat', 'json'))"#,
    )
    .await;
    let v = out[0].as_ref().unwrap();
    assert_eq!(v["response"], Value::Null);
    assert!(
        v["error_message"]
            .as_str()
            .unwrap()
            .contains("not one of the labels"),
        "{v}"
    );
}

#[tokio::test]
async fn null_content_sends_no_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(Fixed("a"))
        .expect(4)
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();

    let out = query_col(
        &ctx,
        r#"SELECT ai_classify(txt, '["a", "b"]') FROM t ORDER BY id"#,
    )
    .await;
    assert_eq!(out.len(), 5);
    assert!(out[1].is_none());
    assert!(out.iter().enumerate().all(|(i, v)| i == 1 || v.is_some()));
}

#[tokio::test]
async fn invalid_arguments_fail_the_query() {
    let (_server, ctx) = classify_server("a").await;
    for sql in [
        r#"SELECT ai_classify('x', '["only_one"]')"#,
        r#"SELECT ai_classify('x', 'not json')"#,
        r#"SELECT ai_classify('x', '["a", "b"]', MAP('version', '1.0'))"#,
        r#"SELECT ai_classify('x', '["a", "b"]', MAP('enableRationales', 'true'))"#,
        r#"SELECT ai_classify('x', '["a", "b"]', MAP('bogus', 'x'))"#,
        r#"SELECT ai_classify('x', '["a", "b"]', MAP('responseFormat', 'xml'))"#,
        r#"SELECT ai_classify('x', '["a", "b"]', MAP('responseFormat', 'text', 'multilabel', 'true'))"#,
    ] {
        let res = match ctx.sql(sql).await {
            Ok(df) => df.collect().await.map(|_| ()),
            Err(e) => Err(e),
        };
        assert!(res.is_err(), "{sql} should fail");
    }
    assert!(ctx.sql("SELECT ai_classify('x')").await.is_err());
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
    let sql = r#"SELECT ai_classify('x', '["a", "b"]')"#;

    assert!(ctx.sql(sql).await.unwrap().collect().await.is_err());

    ctx.sql("SET datafusion_ai_functions.fail_on_error = false")
        .await
        .unwrap();
    assert_eq!(query_col(&ctx, sql).await, vec![None]);

    let sql = r#"SELECT ai_classify('x', '["a", "b"]', MAP('responseFormat', 'json'))"#;
    let out = query_json(&ctx, sql).await;
    let v = out[0].as_ref().unwrap();
    assert_eq!(v["response"], Value::Null);
    assert!(v["error_message"].is_string(), "{v}");
}

#[tokio::test]
async fn plan_uses_async_exec() {
    let (_server, ctx) = classify_server("a").await;
    let batches = ctx
        .sql(r#"EXPLAIN SELECT ai_classify(txt, '["a", "b"]') FROM t"#)
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
        .expect(2)
        .mount(&server)
        .await;
    let ctx = ctx_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();
    common::create_repeated(&ctx).await;

    let out = query_col(
        &ctx,
        r#"SELECT ai_classify(txt, '["x", "y"]') FROM r ORDER BY id"#,
    )
    .await;
    let (x, y) = (Some("x".to_string()), Some("y".to_string()));
    assert_eq!(out, vec![x.clone(), y, x.clone(), None, x]);
}

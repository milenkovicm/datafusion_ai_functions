mod common;

use common::{Echo, ctx_with_rule_for, query_col};
use datafusion::physical_plan::displayable;
use datafusion::prelude::SessionContext;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer};

async fn ctx() -> (MockServer, SessionContext) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(Echo)
        .mount(&server)
        .await;
    let ctx = ctx_with_rule_for(&server).await;
    ctx.sql("SET datafusion_ai_functions.default_model = 'm'")
        .await
        .unwrap();
    (server, ctx)
}

async fn plan(ctx: &SessionContext, sql: &str) -> String {
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    displayable(plan.as_ref()).indent(true).to_string()
}

/// The line right below the `AsyncFuncExec` line, trimmed.
fn below_async(plan: &str) -> &str {
    let mut lines = plan.lines().skip_while(|l| !l.contains("AsyncFuncExec"));
    lines.next().expect("no AsyncFuncExec");
    lines.next().unwrap().trim()
}

const UNION: &str = "SELECT ai_query(txt) FROM (SELECT * FROM t UNION ALL SELECT * FROM t)";

#[tokio::test]
async fn coalesces_multi_partition_input() {
    let (_server, ctx) = ctx().await;
    let p = plan(&ctx, UNION).await;
    assert_eq!(below_async(&p), "CoalescePartitionsExec", "{p}");

    let mut out = query_col(&ctx, UNION).await;
    out.sort();
    let mut expected: Vec<_> = ["a", "c", "d", "e"]
        .iter()
        .flat_map(|t| std::iter::repeat_n(Some(format!("m: {t}")), 2))
        .collect();
    expected.extend([None, None]);
    expected.sort();
    assert_eq!(out, expected);
}

#[tokio::test]
async fn drops_round_robin_repartition() {
    let (_server, ctx) = ctx().await;
    let sql = "SELECT ai_query(txt) FROM t WHERE id > 1";
    let p = plan(&ctx, sql).await;
    assert!(below_async(&p).starts_with("FilterExec"), "{p}");
    assert!(!p.contains("RoundRobin"), "{p}");
    assert!(!p.contains("CoalescePartitionsExec"), "{p}");
    assert_eq!(query_col(&ctx, sql).await.len(), 4);
}

#[tokio::test]
async fn can_be_disabled() {
    let (_server, ctx) = ctx().await;
    ctx.sql("SET datafusion_ai_functions.single_partition = false")
        .await
        .unwrap();
    let p = plan(&ctx, UNION).await;
    assert!(below_async(&p).starts_with("UnionExec"), "{p}");
    let p = plan(&ctx, "SELECT ai_query(txt) FROM t WHERE id > 1").await;
    assert!(p.contains("RoundRobin"), "{p}");
}

#[tokio::test]
async fn leaves_non_ai_plans_alone() {
    let (_server, ctx) = ctx().await;
    let sql = "SELECT id FROM t WHERE id > 1 UNION ALL SELECT id FROM t";
    let with_rule = plan(&ctx, sql).await;
    ctx.sql("SET datafusion_ai_functions.single_partition = false")
        .await
        .unwrap();
    assert_eq!(with_rule, plan(&ctx, sql).await);
}

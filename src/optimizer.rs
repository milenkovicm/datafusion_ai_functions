//! Physical optimizer rule that evaluates AI functions on a single partition.
//!
//! AI functions are bound by the model endpoint, not by local CPU, so running
//! them on many partitions brings little and multiplies the number of requests
//! in flight (`partitions × datafusion_ai_functions.max_concurrency`). This rule merges the
//! input of every `AsyncFuncExec` that calls an AI function into one partition,
//! making `datafusion_ai_functions.max_concurrency` the effective global limit.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::physical_expr::async_scalar_function::AsyncFuncExpr;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::async_func::AsyncFuncExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    ReplaceChildrenOptions,
};

use crate::config::AiQueryConfig;
use crate::udf::AI_FUNCTION_NAMES;

/// Inserts a `CoalescePartitionsExec` (or `SortPreservingMergeExec` when the
/// input is ordered) below each `AsyncFuncExec` that calls an AI function, or
/// drops a round robin repartition of a single partition found there.
///
/// If the `AsyncFuncExec` was hash partitioned (a parent may depend on it),
/// the hash partitioning is restored above it with a `RepartitionExec`.
///
/// Disabled with `SET datafusion_ai_functions.single_partition = false`.
#[derive(Debug, Default)]
pub struct AiSinglePartition;

impl AiSinglePartition {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for AiSinglePartition {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !AiQueryConfig::from_options(config).single_partition {
            return Ok(plan);
        }
        plan.transform_up(|node| {
            let Some(exec) = node.downcast_ref::<AsyncFuncExec>() else {
                return Ok(Transformed::no(node));
            };
            let input = exec.input();
            if input.output_partitioning().partition_count() <= 1 || !calls_ai_function(exec)? {
                return Ok(Transformed::no(node));
            }

            let merged: Arc<dyn ExecutionPlan> = match input.output_ordering() {
                // A round robin repartition only adds parallelism: drop it
                // instead of merging its output back.
                _ if is_round_robin_from_single(input) => Arc::clone(input.children()[0]),
                Some(ordering) => Arc::new(SortPreservingMergeExec::new(
                    ordering.clone(),
                    Arc::clone(input),
                )),
                None => Arc::new(CoalescePartitionsExec::new(Arc::clone(input))),
            };
            let original = node.output_partitioning().clone();
            let mut new = Arc::clone(&node).replace_children(
                vec![merged],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )?;
            if matches!(original, Partitioning::Hash(..)) {
                new = Arc::new(RepartitionExec::try_new(new, original)?);
            }
            Ok(Transformed::yes(new))
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "ai_single_partition"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Whether `plan` is a round robin `RepartitionExec` over a single partition.
fn is_round_robin_from_single(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<RepartitionExec>().is_some_and(|r| {
        matches!(r.partitioning(), Partitioning::RoundRobinBatch(_))
            && r.input().output_partitioning().partition_count() == 1
    })
}

/// Whether any of the async expressions of `exec` is an AI function.
fn calls_ai_function(exec: &AsyncFuncExec) -> Result<bool> {
    let mut found = false;
    exec.apply_expressions(&mut |expr| {
        let is_ai = expr
            .downcast_ref::<AsyncFuncExpr>()
            .and_then(|e| e.func.downcast_ref::<ScalarFunctionExpr>())
            .is_some_and(|f| AI_FUNCTION_NAMES.contains(&f.name()));
        if is_ai {
            found = true;
            Ok(TreeNodeRecursion::Stop)
        } else {
            Ok(TreeNodeRecursion::Continue)
        }
    })?;
    Ok(found)
}

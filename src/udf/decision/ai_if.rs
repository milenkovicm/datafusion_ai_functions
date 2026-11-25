//! `ai_if(input, condition [, options]) -> Boolean`: whether the
//! probability that `condition` holds for `input` is above 0.5.

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, BooleanArray};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::Result;

use super::ai_probability::{condition_question, probabilities};
use super::{DecisionFn, INPUT, Literals, OPTIONS, decision_udf};
use crate::decision::{Answer, Question};
use crate::udf::Arg;

pub(crate) const NAME: &str = "ai_if";
const THRESHOLD: f64 = 0.5;

pub(crate) struct IfFn;

impl DecisionFn for IfFn {
    const NAME: &'static str = NAME;
    const ARGS: &'static [Arg] = &[INPUT, Arg::string("condition"), OPTIONS];
    const REQUIRED: usize = 2;

    fn fixed_type() -> Option<DataType> {
        Some(DataType::Boolean)
    }

    fn questions(args: &Literals) -> Result<Vec<Question>> {
        condition_question(args)
    }

    fn finish(_questions: &[Question], rows: &[Option<Arc<Vec<Answer>>>]) -> Result<ArrayRef> {
        Ok(Arc::new(
            probabilities(rows)
                .map(|p| p.map(|p| p > THRESHOLD))
                .collect::<BooleanArray>(),
        ))
    }
}

decision_udf!(
    /// `ai_if(input, condition [, options])` async scalar UDF.
    AiIf,
    IfFn
);

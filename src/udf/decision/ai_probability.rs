//! `ai_probability(input, condition [, options]) -> Float64`: how
//! likely (0 to 1) it is that `condition` holds for `input`.

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Float64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::Result;

use super::{DecisionFn, INPUT, Literals, OPTIONS, decision_udf, first_answers};
use crate::decision::{Answer, Question, QuestionKind};
use crate::udf::Arg;

pub(crate) const NAME: &str = "ai_probability";

/// The single noul question about `condition`, shared with `ai_if`.
pub(crate) fn condition_question(args: &Literals) -> Result<Vec<Question>> {
    Ok(vec![Question {
        name: "condition".into(),
        instructions: args.required_string("condition")?.to_string(),
        kind: QuestionKind::Noul,
    }])
}

pub(crate) fn probabilities(
    rows: &[Option<Arc<Vec<Answer>>>],
) -> impl Iterator<Item = Option<f64>> + '_ {
    first_answers(rows).map(|a| match a {
        Some(Answer::Noul { probability }) => Some(*probability),
        _ => None,
    })
}

pub(crate) struct ProbabilityFn;

impl DecisionFn for ProbabilityFn {
    const NAME: &'static str = NAME;
    const ARGS: &'static [Arg] = &[INPUT, Arg::string("condition"), OPTIONS];
    const REQUIRED: usize = 2;

    fn fixed_type() -> Option<DataType> {
        Some(DataType::Float64)
    }

    fn questions(args: &Literals) -> Result<Vec<Question>> {
        condition_question(args)
    }

    fn finish(_questions: &[Question], rows: &[Option<Arc<Vec<Answer>>>]) -> Result<ArrayRef> {
        Ok(Arc::new(probabilities(rows).collect::<Float64Array>()))
    }
}

decision_udf!(
    /// `ai_probability(input, condition [, options])` async scalar UDF.
    AiProbability,
    ProbabilityFn
);

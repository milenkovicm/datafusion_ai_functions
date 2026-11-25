//! `ai_score(input, instructions, levels [, options]) -> Float64`:
//! the probability-weighted 0-based index of the level that fits `input`
//! best, from 0 to `n - 1`. `levels` are ordered lowest first.

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Float64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::Result;

use super::{DecisionFn, INPUT, Literals, OPTIONS, criteria_arg, decision_udf, first_answers};
use crate::decision::{Answer, Question, QuestionKind};
use crate::udf::Arg;

pub(crate) const NAME: &str = "ai_score";
pub(crate) const MIN_LEVELS: usize = 2;
pub(crate) const MAX_LEVELS: usize = 10;

pub(crate) struct ScoreFn;

impl DecisionFn for ScoreFn {
    const NAME: &'static str = NAME;
    const ARGS: &'static [Arg] = &[
        INPUT,
        Arg::string("instructions"),
        criteria_arg("levels"),
        OPTIONS,
    ];
    const REQUIRED: usize = 3;

    fn fixed_type() -> Option<DataType> {
        Some(DataType::Float64)
    }

    fn questions(args: &Literals) -> Result<Vec<Question>> {
        Ok(vec![Question {
            name: "score".into(),
            instructions: args.required_string("instructions")?.to_string(),
            kind: QuestionKind::Score(args.criteria("levels", MIN_LEVELS, MAX_LEVELS)?),
        }])
    }

    fn finish(_questions: &[Question], rows: &[Option<Arc<Vec<Answer>>>]) -> Result<ArrayRef> {
        Ok(Arc::new(
            first_answers(rows)
                .map(|a| match a {
                    Some(Answer::Score { score, .. }) => Some(*score),
                    _ => None,
                })
                .collect::<Float64Array>(),
        ))
    }
}

decision_udf!(
    /// `ai_score(input, instructions, levels [, options])` async scalar UDF.
    AiScore,
    ScoreFn
);

//! `ai_decide_classify(input, labels [, options]) -> Utf8`:
//! the label that fits `input` best, asked as a decision model choice
//! question. The result is always one of `labels` (or NULL). Besides
//! `model` and `on_error`, `options` takes `instructions` for the question.
//!
//! Unlike the chat based `ai_classify`, labels may be a list literal:
//! `ai_decide_classify(body, ['billing', 'technical'])`.

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, StringArray};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::Result;

use super::{DecisionFn, INPUT, Literals, OPTIONS, criteria_arg, decision_udf, first_answers};
use crate::decision::{Answer, Question, QuestionKind};
use crate::udf::Arg;

pub(crate) const NAME: &str = "ai_decide_classify";
pub(crate) const MIN_LABELS: usize = 2;
pub(crate) const MAX_LABELS: usize = 255;
const DEFAULT_INSTRUCTIONS: &str = "Which label best fits the input?";

pub(crate) struct ClassifyFn;

impl DecisionFn for ClassifyFn {
    const NAME: &'static str = NAME;
    const ARGS: &'static [Arg] = &[INPUT, criteria_arg("labels"), OPTIONS];
    const OPTION_KEYS: &'static [&'static str] = &["model", "on_error", "instructions"];
    const REQUIRED: usize = 2;

    fn fixed_type() -> Option<DataType> {
        Some(DataType::Utf8)
    }

    fn questions(args: &Literals) -> Result<Vec<Question>> {
        let instructions = args.option("instructions").unwrap_or(DEFAULT_INSTRUCTIONS);
        Ok(vec![Question {
            name: "label".into(),
            instructions: instructions.to_string(),
            kind: QuestionKind::Choice(args.criteria("labels", MIN_LABELS, MAX_LABELS)?),
        }])
    }

    fn finish(_questions: &[Question], rows: &[Option<Arc<Vec<Answer>>>]) -> Result<ArrayRef> {
        Ok(Arc::new(
            first_answers(rows)
                .map(|a| match a {
                    Some(Answer::Choice { choice, .. }) => Some(choice.as_str()),
                    _ => None,
                })
                .collect::<StringArray>(),
        ))
    }
}

decision_udf!(
    /// `ai_decide_classify(input, labels [, options])` async scalar UDF.
    AiDecideClassify,
    ClassifyFn
);

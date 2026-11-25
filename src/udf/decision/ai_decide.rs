//! `ai_decide(input, questions [, options]) -> Struct`: ask several
//! questions about `input` in one request.
//!
//! `questions` is a JSON object of question name to question:
//!
//! ```json
//! {"urgent": {"type": "noul", "instructions": "The customer needs help today"},
//!  "team":   {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": null}},
//!  "tone":   {"type": "score", "instructions": "How frustrated?", "criteria": ["calm", "annoyed", "furious"]}}
//! ```
//!
//! The result has one field per question:
//! `noul` (alias `predicate`) gives `{probability}`, `choice` gives
//! `{choice, confidence, probabilities}` and `score` gives
//! `{score, confidence, probabilities}`, where `probabilities` maps each
//! option or level to its probability. A refused question is NULL.

use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, Float64Array, Float64Builder, MapBuilder, StringBuilder, StructArray,
};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::datatypes::{DataType, Field, Fields};
use datafusion::common::{Result, exec_err};
use serde_json::Value;

use super::ai_decide_classify::{MAX_LABELS, MIN_LABELS};
use super::ai_score::{MAX_LEVELS, MIN_LEVELS};
use super::{
    DecisionFn, INPUT, Literals, OPTIONS, check_criteria, criteria_from_json, decision_udf,
};
use crate::decision::{Answer, Question, QuestionKind};
use crate::udf::Arg;

pub(crate) const NAME: &str = "ai_decide";
const MAX_QUESTIONS: usize = 200;

pub(crate) struct DecideFn;

impl DecisionFn for DecideFn {
    const NAME: &'static str = NAME;
    const ARGS: &'static [Arg] = &[INPUT, Arg::string("questions"), OPTIONS];
    const REQUIRED: usize = 2;

    fn fixed_type() -> Option<DataType> {
        None
    }

    fn questions(args: &Literals) -> Result<Vec<Question>> {
        parse_questions(args.name(), args.required_string("questions")?)
    }

    fn return_type(questions: &[Question]) -> Result<DataType> {
        Ok(DataType::Struct(fields(questions)))
    }

    fn finish(questions: &[Question], rows: &[Option<Arc<Vec<Answer>>>]) -> Result<ArrayRef> {
        let children = questions
            .iter()
            .enumerate()
            .map(|(i, q)| {
                let answers: Vec<Option<&Answer>> = rows
                    .iter()
                    .map(|r| {
                        r.as_ref()
                            .and_then(|a| a.get(i))
                            .filter(|a| **a != Answer::Refusal)
                    })
                    .collect();
                answer_array(&q.kind, &answers)
            })
            .collect::<Result<Vec<_>>>()?;
        let nulls = NullBuffer::from(rows.iter().map(Option::is_some).collect::<Vec<_>>());
        Ok(Arc::new(StructArray::try_new(
            fields(questions),
            children,
            Some(nulls),
        )?))
    }
}

decision_udf!(
    /// `ai_decide(input, questions [, options])` async scalar UDF.
    AiDecide,
    DecideFn
);

fn parse_questions(name: &str, raw: &str) -> Result<Vec<Question>> {
    let map = match serde_json::from_str(raw) {
        Ok(Value::Object(map)) => map,
        Ok(_) => return exec_err!("{name}: questions must be a JSON object of name to question"),
        Err(e) => return exec_err!("{name}: questions must be a JSON object: {e}"),
    };
    if !(1..=MAX_QUESTIONS).contains(&map.len()) {
        return exec_err!(
            "{name}: expected 1 to {MAX_QUESTIONS} questions, got {}",
            map.len()
        );
    }
    map.into_iter()
        .map(|(id, q)| {
            if id.trim().is_empty() {
                return exec_err!("{name}: question names must not be empty");
            }
            let Value::Object(mut q) = q else {
                return exec_err!("{name}: question '{id}' must be a JSON object");
            };
            let instructions = match q.remove("instructions") {
                None | Some(Value::Null) => id.clone(),
                Some(Value::String(s)) if s.trim().is_empty() => id.clone(),
                Some(Value::String(s)) => s,
                Some(_) => return exec_err!("{name}: instructions of '{id}' must be a string"),
            };
            let param = format!("criteria of '{id}'");
            let criteria = |min, max| match q.get("criteria") {
                Some(c) => {
                    let items = criteria_from_json(name, &param, c.clone())?;
                    check_criteria(name, &param, &items, min, max)?;
                    Ok(items)
                }
                None => exec_err!("{name}: question '{id}' needs criteria"),
            };
            let kind = match q.get("type").and_then(Value::as_str) {
                Some("noul" | "predicate") => QuestionKind::Noul,
                Some("choice") => QuestionKind::Choice(criteria(MIN_LABELS, MAX_LABELS)?),
                Some("score") => QuestionKind::Score(criteria(MIN_LEVELS, MAX_LEVELS)?),
                other => {
                    return exec_err!(
                        "{name}: type of '{id}' must be 'noul', 'choice' or 'score', got {}",
                        other.unwrap_or("nothing")
                    );
                }
            };
            Ok(Question {
                name: id,
                instructions,
                kind,
            })
        })
        .collect()
}

fn probabilities_builder() -> MapBuilder<StringBuilder, Float64Builder> {
    MapBuilder::new(None, StringBuilder::new(), Float64Builder::new())
}

fn probabilities_type() -> DataType {
    probabilities_builder().finish().data_type().clone()
}

fn answer_fields(kind: &QuestionKind) -> Fields {
    let f64_field = |n: &str| Field::new(n, DataType::Float64, true);
    let probabilities = Field::new("probabilities", probabilities_type(), true);
    match kind {
        QuestionKind::Noul => Fields::from(vec![f64_field("probability")]),
        QuestionKind::Choice(_) => Fields::from(vec![
            Field::new("choice", DataType::Utf8, true),
            f64_field("confidence"),
            probabilities,
        ]),
        QuestionKind::Score(_) => Fields::from(vec![
            f64_field("score"),
            f64_field("confidence"),
            probabilities,
        ]),
    }
}

fn fields(questions: &[Question]) -> Fields {
    questions
        .iter()
        .map(|q| Field::new(&q.name, DataType::Struct(answer_fields(&q.kind)), true))
        .collect()
}

/// One question's answers as a struct array; `None` is NULL.
fn answer_array(kind: &QuestionKind, answers: &[Option<&Answer>]) -> Result<ArrayRef> {
    let nulls = NullBuffer::from(answers.iter().map(Option::is_some).collect::<Vec<_>>());
    let children: Vec<ArrayRef> = match kind {
        QuestionKind::Noul => {
            let p: Float64Array = answers
                .iter()
                .map(|a| match a {
                    Some(Answer::Noul { probability }) => Some(*probability),
                    _ => None,
                })
                .collect();
            vec![Arc::new(p)]
        }
        QuestionKind::Choice(_) | QuestionKind::Score(_) => {
            let mut confidence = Float64Builder::with_capacity(answers.len());
            let mut probs = probabilities_builder();
            let mut first: Vec<ArrayRef> = Vec::new();
            let (mut choices, mut scores) = (StringBuilder::new(), Float64Builder::new());
            for a in answers {
                let entries = match a {
                    Some(Answer::Choice {
                        choice,
                        confidence: c,
                        probabilities,
                    }) => {
                        choices.append_value(choice);
                        confidence.append_value(*c);
                        Some(probabilities)
                    }
                    Some(Answer::Score {
                        score,
                        confidence: c,
                        probabilities,
                    }) => {
                        scores.append_value(*score);
                        confidence.append_value(*c);
                        Some(probabilities)
                    }
                    _ => {
                        choices.append_null();
                        scores.append_null();
                        confidence.append_null();
                        None
                    }
                };
                match entries {
                    Some(entries) => {
                        for (k, p) in entries {
                            probs.keys().append_value(k);
                            probs.values().append_value(*p);
                        }
                        probs.append(true)?;
                    }
                    None => probs.append(false)?,
                }
            }
            first.push(match kind {
                QuestionKind::Choice(_) => Arc::new(choices.finish()) as ArrayRef,
                _ => Arc::new(scores.finish()),
            });
            first.push(Arc::new(confidence.finish()));
            first.push(Arc::new(probs.finish()));
            first
        }
    };
    Ok(Arc::new(StructArray::try_new(
        answer_fields(kind),
        children,
        Some(nulls),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::Criterion;
    use datafusion::arrow::array::{AsArray, StringArray};

    const QUESTIONS: &str = r#"{
        "urgent": {"type": "noul", "instructions": "Needs help today"},
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": null}},
        "tone": {"type": "score", "criteria": ["calm", "furious"]}
    }"#;

    #[test]
    fn parses_questions() {
        let q = parse_questions("f", QUESTIONS).unwrap();
        assert_eq!(q.len(), 3);
        assert_eq!(q[0].kind, QuestionKind::Noul);
        assert_eq!(q[2].name, "tone");
        assert_eq!(q[2].instructions, "tone"); // defaults to the name
        assert_eq!(
            q[1].kind,
            QuestionKind::Choice(vec![
                Criterion {
                    label: "billing".into(),
                    description: Some("Payments".into())
                },
                Criterion::new("technical"),
            ])
        );
        assert_eq!(
            parse_questions("f", r#"{"a": {"type": "predicate"}}"#).unwrap()[0].kind,
            QuestionKind::Noul
        );
        for bad in [
            "[]",
            "{}",
            "nope",
            r#"{"a": 1}"#,
            r#"{"a": {"type": "maybe"}}"#,
            r#"{"a": {"type": "choice"}}"#,
            r#"{"a": {"type": "choice", "criteria": ["x"]}}"#,
            r#"{"a": {"type": "score", "criteria": {"x": 1, "y": null}}}"#,
            r#"{"a": {"type": "noul", "instructions": 3}}"#,
        ] {
            assert!(parse_questions("f", bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn builds_struct_matching_return_type() {
        let questions = parse_questions("f", QUESTIONS).unwrap();
        let row = Arc::new(vec![
            Answer::Noul { probability: 0.9 },
            Answer::Choice {
                choice: "billing".into(),
                confidence: 0.8,
                probabilities: vec![("billing".into(), 0.9), ("technical".into(), 0.1)],
            },
            Answer::Refusal,
        ]);
        let rows = vec![Some(row), None];
        let out = DecideFn::finish(&questions, &rows).unwrap();
        assert_eq!(out.data_type(), &DecideFn::return_type(&questions).unwrap());
        let s = out.as_struct();
        assert!(s.is_valid(0) && s.is_null(1));
        let team = s.column_by_name("team").unwrap().as_struct();
        let choice = team
            .column_by_name("choice")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(choice.value(0), "billing");
        assert!(s.column_by_name("tone").unwrap().is_null(0));
        let urgent = s.column_by_name("urgent").unwrap().as_struct();
        assert_eq!(
            urgent
                .column(0)
                .as_primitive::<datafusion::arrow::datatypes::Float64Type>()
                .value(0),
            0.9
        );
    }
}

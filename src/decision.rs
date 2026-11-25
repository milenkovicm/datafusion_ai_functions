//! Decision model requests.
//!
//! A decision model doesn't generate text: it answers questions about an
//! input with probabilities. Two wire formats are supported, selected with
//! `datafusion_ai_functions.decision_api`:
//!
//! - `openai`: OpenAI's `POST {base}/decisions`. Questions are an array of
//!   `predicate`, `choice` and `score` questions; answers come back in order.
//! - `systemone`: Ollama's `POST {base}/systemone`. Questions are an object
//!   of name to `noul`, `choice` or `score` question with `criteria`; answers
//!   come back as an object keyed by question name.
//!
//! Both go through the same `async-openai` client, so auth, timeouts and
//! retries are shared with the chat functions.

use async_openai::error::OpenAIError;
use async_openai::traits::RequestOptionsBuilder;
use async_openai::types::decisions::{
    AnswerResource, ChoiceOptionParam, ChoiceValueParam, ChoiceValueResource, DecisionInput,
    DecisionRequest, QuestionParam, QuestionParamChoice, QuestionParamInstructions,
    QuestionParamPredicate, QuestionParamScore, ScoreLevelParam,
};
use datafusion::common::{DataFusionError, Result, exec_datafusion_err, exec_err};
use serde_json::{Map, Value, json};

use crate::openai::OpenAiClient;

/// Wire format of the decision endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecisionApi {
    OpenAi,
    SystemOne,
}

impl DecisionApi {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "openai" => Ok(Self::OpenAi),
            "systemone" => Ok(Self::SystemOne),
            other => exec_err!(
                "datafusion_ai_functions.decision_api must be 'openai' or 'systemone', got '{other}'"
            ),
        }
    }
}

/// A choice option or a score level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Criterion {
    pub label: String,
    pub description: Option<String>,
}

impl Criterion {
    pub(crate) fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum QuestionKind {
    /// How likely the statement in `instructions` is true.
    Noul,
    /// Which option fits best.
    Choice(Vec<Criterion>),
    /// Which level fits best, levels ordered lowest first.
    Score(Vec<Criterion>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Question {
    pub name: String,
    pub instructions: String,
    pub kind: QuestionKind,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Answer {
    Noul {
        probability: f64,
    },
    Choice {
        choice: String,
        confidence: f64,
        /// Option label to probability.
        probabilities: Vec<(String, f64)>,
    },
    Score {
        /// Probability-weighted 0-based level index.
        score: f64,
        confidence: f64,
        /// Level label to probability, lowest level first.
        probabilities: Vec<(String, f64)>,
    },
    /// The model declined to answer this question.
    Refusal,
}

/// Ask all `questions` about `input` in one request. Answers are in
/// question order, and each matches its question's kind (or is a refusal).
pub(crate) async fn decide(
    client: &OpenAiClient,
    api: DecisionApi,
    model: &str,
    input: &str,
    questions: &[Question],
) -> Result<Vec<Answer>> {
    let ext = |e: OpenAIError| DataFusionError::External(Box::new(e));
    let answers = match api {
        DecisionApi::OpenAi => {
            let resp: Value = client
                .decisions()
                .create_byot(openai_request(model, input, questions))
                .await
                .map_err(ext)?;
            parse_openai(&resp, questions)?
        }
        DecisionApi::SystemOne => {
            let resp: Value = client
                .decisions()
                .path("/systemone")
                .map_err(ext)?
                .create_byot(systemone_request(model, input, questions))
                .await
                .map_err(ext)?;
            parse_systemone(&resp, questions)?
        }
    };
    check_answers(questions, &answers)?;
    Ok(answers)
}

fn openai_request(model: &str, input: &str, questions: &[Question]) -> DecisionRequest {
    let questions = questions
        .iter()
        .map(|q| {
            let name = Some(q.name.clone());
            let instructions = QuestionParamInstructions::Text(q.instructions.clone());
            match &q.kind {
                QuestionKind::Noul => {
                    QuestionParam::Predicate(QuestionParamPredicate { name, instructions })
                }
                QuestionKind::Choice(options) => QuestionParam::Choice(QuestionParamChoice {
                    name,
                    instructions,
                    choices: options
                        .iter()
                        .map(|c| ChoiceOptionParam {
                            value: ChoiceValueParam::Text(c.label.clone()),
                            description: c.description.clone(),
                        })
                        .collect(),
                }),
                QuestionKind::Score(levels) => QuestionParam::Score(QuestionParamScore {
                    name,
                    instructions,
                    levels: levels
                        .iter()
                        .map(|l| ScoreLevelParam {
                            label: l.label.clone(),
                            description: l.description.clone(),
                        })
                        .collect(),
                }),
            }
        })
        .collect();
    DecisionRequest {
        model: model.to_string(),
        input: DecisionInput::Text(input.to_string()),
        questions,
        ..Default::default()
    }
}

fn invalid_response(msg: impl std::fmt::Display) -> DataFusionError {
    exec_datafusion_err!("invalid decision response: {msg}")
}

fn choice_text(value: ChoiceValueResource) -> String {
    match value {
        ChoiceValueResource::Text(s) => s,
        ChoiceValueResource::Boolean(b) => b.to_string(),
    }
}

fn parse_openai(resp: &Value, questions: &[Question]) -> Result<Vec<Answer>> {
    let answers = resp
        .get("answers")
        .cloned()
        .ok_or_else(|| invalid_response("missing 'answers'"))?;
    let answers: Vec<AnswerResource> = serde_json::from_value(answers).map_err(invalid_response)?;
    if answers.len() != questions.len() {
        return Err(invalid_response(format!(
            "expected {} answers, got {}",
            questions.len(),
            answers.len()
        )));
    }
    Ok(answers
        .into_iter()
        .map(|a| match a {
            AnswerResource::Predicate(p) => Answer::Noul {
                probability: p.probability,
            },
            AnswerResource::Choice(c) => Answer::Choice {
                choice: choice_text(c.choice),
                confidence: c.confidence,
                probabilities: c
                    .probabilities
                    .into_iter()
                    .map(|p| (choice_text(p.value), p.probability))
                    .collect(),
            },
            AnswerResource::Score(s) => Answer::Score {
                score: s.score,
                confidence: s.confidence,
                probabilities: s
                    .probabilities
                    .into_iter()
                    .map(|p| (p.label, p.probability))
                    .collect(),
            },
            AnswerResource::Refusal(_) => Answer::Refusal,
        })
        .collect())
}

/// A score level as sent to `systemone`, which takes one string per level.
fn systemone_level(level: &Criterion) -> String {
    match &level.description {
        Some(d) => format!("{}: {d}", level.label),
        None => level.label.clone(),
    }
}

fn systemone_request(model: &str, input: &str, questions: &[Question]) -> Value {
    let mut qs = Map::new();
    for q in questions {
        let mut obj = Map::new();
        let kind = match &q.kind {
            QuestionKind::Noul => "noul",
            QuestionKind::Choice(_) => "choice",
            QuestionKind::Score(_) => "score",
        };
        obj.insert("type".into(), json!(kind));
        obj.insert("instructions".into(), json!(q.instructions));
        match &q.kind {
            QuestionKind::Noul => {}
            QuestionKind::Choice(options) => {
                let criteria = options
                    .iter()
                    .map(|c| (c.label.clone(), json!(c.description)))
                    .collect();
                obj.insert("criteria".into(), Value::Object(criteria));
            }
            QuestionKind::Score(levels) => {
                let criteria = levels.iter().map(|l| json!(systemone_level(l))).collect();
                obj.insert("criteria".into(), Value::Array(criteria));
            }
        }
        qs.insert(q.name.clone(), Value::Object(obj));
    }
    json!({"model": model, "state": input, "questions": qs})
}

fn parse_systemone(resp: &Value, questions: &[Question]) -> Result<Vec<Answer>> {
    let answers = resp
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_response("missing 'answers' object"))?;
    questions
        .iter()
        .map(|q| {
            let Some(a) = answers.get(&q.name) else {
                return Ok(Answer::Refusal);
            };
            let number = |key: &str| {
                a.get(key).and_then(Value::as_f64).ok_or_else(|| {
                    invalid_response(format!("answer '{}' has no number '{key}'", q.name))
                })
            };
            let probabilities = || a.get("probabilities").and_then(Value::as_object);
            Ok(match (a.get("type").and_then(Value::as_str), &q.kind) {
                (Some("noul"), _) => Answer::Noul {
                    probability: number("noul")?,
                },
                (Some("choice"), _) => Answer::Choice {
                    choice: a
                        .get("choice")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            invalid_response(format!("answer '{}' has no 'choice'", q.name))
                        })?
                        .to_string(),
                    confidence: number("confidence")?,
                    probabilities: probabilities()
                        .into_iter()
                        .flatten()
                        .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?)))
                        .collect(),
                },
                // Probabilities are keyed by level index ("0", "1", ...).
                (Some("score"), QuestionKind::Score(levels)) => {
                    let probs = probabilities();
                    Answer::Score {
                        score: number("score")?,
                        confidence: number("confidence")?,
                        probabilities: levels
                            .iter()
                            .enumerate()
                            .map(|(i, l)| {
                                let p = probs
                                    .and_then(|p| p.get(&i.to_string()))
                                    .and_then(Value::as_f64)
                                    .unwrap_or(0.0);
                                (l.label.clone(), p)
                            })
                            .collect(),
                    }
                }
                (Some("score"), _) => Answer::Score {
                    score: number("score")?,
                    confidence: number("confidence")?,
                    probabilities: Vec::new(),
                },
                _ => Answer::Refusal,
            })
        })
        .collect()
}

/// Every answer must be a refusal or match its question's kind; a choice
/// must be one of the options.
fn check_answers(questions: &[Question], answers: &[Answer]) -> Result<()> {
    for (q, a) in questions.iter().zip(answers) {
        let ok = match (&q.kind, a) {
            (_, Answer::Refusal) => true,
            (QuestionKind::Noul, Answer::Noul { .. }) => true,
            (QuestionKind::Choice(options), Answer::Choice { choice, .. }) => {
                options.iter().any(|o| &o.label == choice)
            }
            (QuestionKind::Score(_), Answer::Score { .. }) => true,
            _ => false,
        };
        if !ok {
            return Err(invalid_response(format!(
                "unexpected answer for question '{}': {a:?}",
                q.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn questions() -> Vec<Question> {
        vec![
            Question {
                name: "refund".into(),
                instructions: "The customer asks for a refund".into(),
                kind: QuestionKind::Noul,
            },
            Question {
                name: "team".into(),
                instructions: "Which team?".into(),
                kind: QuestionKind::Choice(vec![
                    Criterion {
                        label: "billing".into(),
                        description: Some("Payments".into()),
                    },
                    Criterion::new("technical"),
                ]),
            },
            Question {
                name: "tone".into(),
                instructions: "How frustrated?".into(),
                kind: QuestionKind::Score(vec![
                    Criterion::new("calm"),
                    Criterion {
                        label: "annoyed".into(),
                        description: Some("irritated".into()),
                    },
                    Criterion::new("furious"),
                ]),
            },
        ]
    }

    #[test]
    fn parses_api() {
        assert_eq!(DecisionApi::parse("").unwrap(), DecisionApi::OpenAi);
        assert_eq!(
            DecisionApi::parse(" SystemOne ").unwrap(),
            DecisionApi::SystemOne
        );
        assert!(DecisionApi::parse("ollama").is_err());
    }

    #[test]
    fn builds_openai_request() {
        let req = serde_json::to_value(openai_request("m", "hi", &questions())).unwrap();
        assert_eq!(
            req,
            json!({
                "model": "m",
                "input": "hi",
                "questions": [
                    {"type": "predicate", "name": "refund", "instructions": "The customer asks for a refund"},
                    {"type": "choice", "name": "team", "instructions": "Which team?",
                     "choices": [{"value": "billing", "description": "Payments"}, {"value": "technical"}]},
                    {"type": "score", "name": "tone", "instructions": "How frustrated?",
                     "levels": [{"label": "calm"}, {"label": "annoyed", "description": "irritated"}, {"label": "furious"}]}
                ]
            })
        );
    }

    #[test]
    fn builds_systemone_request() {
        let req = systemone_request("nimble", "hi", &questions());
        assert_eq!(
            req.to_string(),
            json!({
                "model": "nimble",
                "state": "hi",
                "questions": {
                    "refund": {"type": "noul", "instructions": "The customer asks for a refund"},
                    "team": {"type": "choice", "instructions": "Which team?",
                             "criteria": {"billing": "Payments", "technical": null}},
                    "tone": {"type": "score", "instructions": "How frustrated?",
                             "criteria": ["calm", "annoyed: irritated", "furious"]}
                }
            })
            .to_string()
        );
    }

    fn expected() -> Vec<Answer> {
        vec![
            Answer::Noul { probability: 0.9 },
            Answer::Choice {
                choice: "billing".into(),
                confidence: 0.8,
                probabilities: vec![("billing".into(), 0.95), ("technical".into(), 0.05)],
            },
            Answer::Score {
                score: 1.5,
                confidence: 0.3,
                probabilities: vec![
                    ("calm".into(), 0.0),
                    ("annoyed".into(), 0.5),
                    ("furious".into(), 0.5),
                ],
            },
        ]
    }

    #[test]
    fn parses_systemone_response() {
        let resp = json!({"model": "nimble", "answers": {
            "refund": {"type": "noul", "noul": 0.9},
            "team": {"type": "choice", "choice": "billing",
                     "probabilities": {"billing": 0.95, "technical": 0.05}, "confidence": 0.8},
            "tone": {"type": "score", "score": 1.5, "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
                     "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}, "confidence": 0.3}
        }});
        let answers = parse_systemone(&resp, &questions()).unwrap();
        assert_eq!(answers, expected());
        check_answers(&questions(), &answers).unwrap();

        let resp = json!({"answers": {"refund": {"type": "refusal"}}});
        let answers = parse_systemone(&resp, &questions()).unwrap();
        assert_eq!(answers, vec![Answer::Refusal; 3]);

        assert!(
            parse_systemone(
                &json!({"answers": {"refund": {"type": "noul"}}}),
                &questions()
            )
            .is_err()
        );
        assert!(parse_systemone(&json!({}), &questions()).is_err());
    }

    #[test]
    fn parses_openai_response() {
        let resp = json!({"model": "m", "answers": [
            {"type": "predicate", "name": "refund", "probability": 0.9},
            {"type": "choice", "name": "team", "choice": "billing", "confidence": 0.8,
             "probabilities": [{"value": "billing", "probability": 0.95}, {"value": "technical", "probability": 0.05}]},
            {"type": "score", "name": "tone", "score": 1.5, "confidence": 0.3,
             "probabilities": [{"value": 0, "label": "calm", "probability": 0.0},
                               {"value": 1, "label": "annoyed", "probability": 0.5},
                               {"value": 2, "label": "furious", "probability": 0.5}]}
        ]});
        assert_eq!(parse_openai(&resp, &questions()).unwrap(), expected());

        let resp = json!({"answers": [{"type": "refusal", "name": "refund"}]});
        assert!(parse_openai(&resp, &questions()).is_err()); // wrong count
        let qs = &questions()[..1];
        assert_eq!(parse_openai(&resp, qs).unwrap(), vec![Answer::Refusal]);
    }

    #[test]
    fn rejects_mismatched_answers() {
        let qs = questions();
        let mut answers = expected();
        answers[1] = Answer::Choice {
            choice: "sales".into(),
            confidence: 1.0,
            probabilities: vec![],
        };
        assert!(check_answers(&qs, &answers).is_err());
        answers[1] = Answer::Noul { probability: 0.1 };
        assert!(check_answers(&qs, &answers).is_err());
    }
}

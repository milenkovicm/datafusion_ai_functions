//! Decision functions (`ai_probability`, `ai_if`, `ai_score`,
//! `ai_decide_classify`, `ai_decide`) and the code they share.
//!
//! Each function is a [`DecisionFn`]: it turns its literal arguments into
//! questions and the answers into an Arrow array. Everything else is shared:
//! the signature, the plan time checks, the `options` MAP (`model`,
//! `on_error`), input conversion, deduplication and the concurrent request
//! loop.

pub(crate) mod ai_decide;
pub(crate) mod ai_decide_classify;
pub(crate) mod ai_if;
pub(crate) mod ai_probability;
pub(crate) mod ai_score;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use datafusion::arrow::array::{Array, ArrayRef, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::arrow::json::LineDelimitedWriter;
use datafusion::common::{
    DataFusionError, Result, ScalarValue, exec_datafusion_err, exec_err, plan_err,
};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, Signature, TypeSignature, Volatility,
};
use serde_json::Value;

use super::{
    Arg, STRING_TYPES, check_args, is_string_or_null, map_unique_rows, non_blank, option_entries,
    to_utf8,
};
use crate::config::AiQueryConfig;
use crate::decision::{Answer, Criterion, DecisionApi, Question, decide};
use crate::openai::{ClientCache, OpenAiClient};

pub use ai_decide::AiDecide;
pub use ai_decide_classify::AiDecideClassify;
pub use ai_if::AiIf;
pub use ai_probability::AiProbability;
pub use ai_score::AiScore;

fn is_input(t: &DataType) -> bool {
    !matches!(
        t,
        DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_)
    )
}

fn is_list_or_string_or_null(t: &DataType) -> bool {
    is_string_or_null(t)
        || matches!(
            t,
            DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(..)
        )
}

const INPUT: Arg = Arg::new(
    "input",
    "text, a number, a struct, a list or a map",
    is_input,
);
/// The trailing `MAP<STRING, STRING>` of optional settings.
const OPTIONS: Arg = Arg::options();
/// Option keys every decision function accepts.
const COMMON_OPTION_KEYS: &[&str] = &["model", "on_error"];
/// A list literal like `['a', 'b']` or a JSON string.
const fn criteria_arg(name: &'static str) -> Arg {
    Arg::new(
        name,
        "a list of strings or a JSON string",
        is_list_or_string_or_null,
    )
}

/// One decision function.
pub(crate) trait DecisionFn {
    const NAME: &'static str;
    /// Declared arguments: `input` first, [`OPTIONS`] last.
    const ARGS: &'static [Arg];
    /// Keys accepted in the options MAP (lower case).
    const OPTION_KEYS: &'static [&'static str] = COMMON_OPTION_KEYS;
    /// Number of mandatory arguments.
    const REQUIRED: usize;

    /// Return type when the questions aren't known at plan time because an
    /// argument isn't a literal yet. `None` means they must be known.
    fn fixed_type() -> Option<DataType>;

    /// Questions to ask, from the literal arguments.
    fn questions(args: &Literals) -> Result<Vec<Question>>;

    fn return_type(_questions: &[Question]) -> Result<DataType> {
        match Self::fixed_type() {
            Some(t) => Ok(t),
            None => exec_err!("{}: return type depends on the arguments", Self::NAME),
        }
    }

    /// Build the result. A `None` row is NULL; a refused answer is NULL
    /// for that question.
    fn finish(questions: &[Question], rows: &[Option<Arc<Vec<Answer>>>]) -> Result<ArrayRef>;
}

/// The literal arguments of one call, aligned with [`DecisionFn::ARGS`].
/// `None` is an absent or NULL argument (and the `input` slot).
pub(crate) struct Literals<'a> {
    name: &'static str,
    args: &'static [Arg],
    values: Vec<Option<&'a ScalarValue>>,
    /// The options MAP: lower case key to value, NULL values left out.
    options: Vec<(String, String)>,
}

impl<'a> Literals<'a> {
    fn new<F: DecisionFn>(values: Vec<Option<&'a ScalarValue>>) -> Result<Self> {
        let mut lits = Self {
            name: F::NAME,
            args: F::ARGS,
            values,
            options: Vec::new(),
        };
        if let Some(v) = lits.value(OPTIONS.name) {
            lits.options = parse_options(F::NAME, F::OPTION_KEYS, v)?;
        }
        Ok(lits)
    }

    pub(crate) fn name(&self) -> &'static str {
        self.name
    }

    /// An option value, `None` when not set or blank.
    pub(crate) fn option(&self, key: &str) -> Option<&str> {
        self.options
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.trim())
            .filter(|v| !v.is_empty())
    }

    fn value(&self, param: &str) -> Option<&'a ScalarValue> {
        let i = self.args.iter().position(|a| a.name == param)?;
        self.values.get(i).copied().flatten()
    }

    /// A string argument, `None` when absent or NULL.
    pub(crate) fn string(&self, param: &str) -> Result<Option<&'a str>> {
        match self.value(param) {
            None => Ok(None),
            Some(v) => match v.try_as_str() {
                Some(s) => Ok(s),
                None => exec_err!(
                    "{}: {param} must be a string, got {}",
                    self.name,
                    v.data_type()
                ),
            },
        }
    }

    /// A string argument that must be set and not blank.
    pub(crate) fn required_string(&self, param: &str) -> Result<&'a str> {
        match self.string(param)? {
            Some(s) if !s.trim().is_empty() => Ok(s),
            _ => exec_err!("{}: {param} must not be NULL or empty", self.name),
        }
    }

    /// Options or levels: a list literal like `['a', 'b']`, a JSON array of
    /// strings, or a JSON object of label to description (or null).
    pub(crate) fn criteria(&self, param: &str, min: usize, max: usize) -> Result<Vec<Criterion>> {
        let name = self.name;
        let Some(v) = self.value(param) else {
            return exec_err!("{name}: {param} must not be NULL");
        };
        let items = if let Some(list) = string_list(name, param, v)? {
            list.into_iter().map(Criterion::new).collect()
        } else if let Some(Some(s)) = v.try_as_str() {
            let json: Value = serde_json::from_str(s).map_err(|e| {
                exec_datafusion_err!(
                    "{name}: {param} must be a list or a JSON array or object: {e}"
                )
            })?;
            criteria_from_json(name, param, json)?
        } else {
            return exec_err!("{name}: {param} must be a list of strings or a JSON string");
        };
        check_criteria(name, param, &items, min, max)?;
        Ok(items)
    }
}

/// The strings of a list literal, or `None` if `v` isn't a list.
fn string_list(name: &str, param: &str, v: &ScalarValue) -> Result<Option<Vec<String>>> {
    let values = match v {
        ScalarValue::List(a) => a.value(0),
        ScalarValue::LargeList(a) => a.value(0),
        ScalarValue::FixedSizeList(a) => a.value(0),
        _ => return Ok(None),
    };
    if !STRING_TYPES.contains(values.data_type()) {
        return exec_err!(
            "{name}: {param} must be a list of strings, got {}",
            v.data_type()
        );
    }
    let s = to_utf8(&values)?;
    if s.null_count() > 0 {
        return exec_err!("{name}: {param} must not contain NULL");
    }
    Ok(Some((0..s.len()).map(|i| s.value(i).to_string()).collect()))
}

pub(crate) fn criteria_from_json(name: &str, param: &str, json: Value) -> Result<Vec<Criterion>> {
    match json {
        Value::Array(items) => items
            .into_iter()
            .map(|item| match item {
                Value::String(s) => Ok(Criterion::new(s)),
                other => exec_err!("{name}: {param} entries must be strings, got {other}"),
            })
            .collect(),
        Value::Object(map) => map
            .into_iter()
            .map(|(label, d)| match d {
                Value::Null => Ok(Criterion::new(label)),
                Value::String(d) => Ok(Criterion {
                    label,
                    description: non_blank(&d),
                }),
                other => exec_err!(
                    "{name}: description of '{label}' in {param} must be a string or null, got {other}"
                ),
            })
            .collect(),
        _ => exec_err!("{name}: {param} must be a JSON array or object"),
    }
}

pub(crate) fn check_criteria(
    name: &str,
    param: &str,
    items: &[Criterion],
    min: usize,
    max: usize,
) -> Result<()> {
    if !(min..=max).contains(&items.len()) {
        return exec_err!(
            "{name}: {param} must have {min} to {max} entries, got {}",
            items.len()
        );
    }
    for (i, c) in items.iter().enumerate() {
        if c.label.trim().is_empty() {
            return exec_err!("{name}: {param} must not contain empty labels");
        }
        if items[..i].iter().any(|o| o.label == c.label) {
            return exec_err!("{name}: duplicate '{}' in {param}", c.label);
        }
    }
    Ok(())
}

/// Read a literal options MAP, checking its keys against `allowed`.
fn parse_options(name: &str, allowed: &[&str], v: &ScalarValue) -> Result<Vec<(String, String)>> {
    if !matches!(v, ScalarValue::Map(_)) {
        return exec_err!(
            "{name}: options must be a MAP<STRING, STRING>, got {}",
            v.data_type()
        );
    }
    let entries = option_entries(name, &v.to_array_of_size(1)?)?;
    let mut options = Vec::new();
    for (key, value) in entries.into_iter().next().unwrap_or_default() {
        let key = key.to_ascii_lowercase();
        if !allowed.contains(&key.as_str()) {
            return exec_err!(
                "{name}: unknown option '{key}', expected one of: {}",
                allowed.join(", ")
            );
        }
        if let Some(value) = value {
            options.push((key, value));
        }
    }
    Ok(options)
}

/// The `on_error` option, `'fail'` or `'null'`: whether to fail the query.
fn parse_on_error(name: &str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "fail" => Ok(true),
        "null" => Ok(false),
        _ => exec_err!("{name}: on_error must be 'fail' or 'null', got '{value}'"),
    }
}

/// Any number of arguments up to the declared ones. Too few arguments are
/// reported by [`check_args`] instead of DataFusion's generic signature
/// error.
fn signature<F: DecisionFn>() -> Signature {
    let variants = std::iter::once(TypeSignature::Nullary)
        .chain((1..=F::ARGS.len()).map(TypeSignature::Any))
        .collect();
    Signature::one_of(variants, Volatility::Volatile)
}

/// Check argument types and literals, and compute the return field.
fn return_field<F: DecisionFn>(args: ReturnFieldArgs) -> Result<FieldRef> {
    let types: Vec<DataType> = args
        .arg_fields
        .iter()
        .map(|f| f.data_type().clone())
        .collect();
    check_args(F::NAME, &types, F::ARGS, F::REQUIRED)?;
    // `MAP(...)` and `['a', 'b']` are function calls until they're
    // simplified into literals. The options never change the return type, so
    // they're checked once they are literals.
    let not_literal = args
        .scalar_arguments
        .iter()
        .enumerate()
        .skip(1)
        .find(|(i, s)| s.is_none() && F::ARGS[*i].name != OPTIONS.name)
        .map(|(i, _)| i);
    let data_type = match not_literal {
        None => {
            let values = std::iter::once(None)
                .chain(
                    args.scalar_arguments
                        .iter()
                        .skip(1)
                        .map(|s| s.filter(|v| !v.is_null())),
                )
                .collect();
            let lits = Literals::new::<F>(values)?;
            if let Some(v) = lits.option("on_error") {
                parse_on_error(F::NAME, v)?;
            }
            F::return_type(&F::questions(&lits)?)?
        }
        Some(i) => match F::fixed_type() {
            Some(t) => t,
            None => return plan_err!("{}: {} must be a constant", F::NAME, F::ARGS[i].name),
        },
    };
    Ok(Arc::new(Field::new(F::NAME, data_type, true)))
}

async fn invoke<F: DecisionFn>(
    clients: &ClientCache,
    last_call: &LastCall,
    args: ScalarFunctionArgs,
) -> Result<ColumnarValue> {
    let values = args
        .args
        .iter()
        .enumerate()
        .map(|(i, a)| match a {
            _ if i == 0 => Ok(None),
            ColumnarValue::Scalar(s) => Ok(Some(s).filter(|s| !s.is_null())),
            ColumnarValue::Array(_) => {
                exec_err!("{}: {} must be a constant", F::NAME, F::ARGS[i].name)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let lits = Literals::new::<F>(values)?;
    let questions = F::questions(&lits)?;

    let cfg = AiQueryConfig::from_options(&args.config_options);
    let model = lits.option("model").unwrap_or(cfg.decision_model.trim());
    if model.is_empty() {
        return exec_err!(
            "{}: no decision model configured, set datafusion_ai_functions.decision_model or the 'model' option",
            F::NAME
        );
    }
    let fail_on_error = match lits.option("on_error") {
        Some(v) => parse_on_error(F::NAME, v)?,
        None => cfg.fail_on_error,
    };
    let api = DecisionApi::parse(&cfg.decision_api)?;
    let inputs = input_rows(&args.args[0], args.number_rows)?;
    let client_cfg = cfg.decision_client_config();

    let key = CallKey {
        endpoint: client_cfg.endpoint.clone(),
        api,
        model: model.to_string(),
        questions: questions.clone(),
        fail_on_error,
    };
    let rows = match last_call.get(&key, &inputs) {
        Some(rows) => rows,
        None => {
            let client = clients.get(&client_cfg)?;
            let call = Call {
                name: F::NAME,
                client: &client,
                api,
                model,
                questions: &questions,
                fail_on_error,
            };
            let rows = call.run(cfg.max_concurrency, &inputs).await?;
            last_call.put(key, inputs, rows.clone());
            rows
        }
    };
    Ok(ColumnarValue::Array(F::finish(&questions, &rows)?))
}

type Rows = Vec<Option<Arc<Vec<Answer>>>>;

/// How long [`LastCall`] answers stay usable.
const LAST_CALL_TTL: Duration = Duration::from_secs(5);

/// The answers of a function's most recent call.
///
/// DataFusion's `push_down_leaf_projections` rule can copy a call once per
/// field access: `SELECT d['a'], d['b'] FROM (SELECT ai_decide(...) AS d ...)`
/// evaluates `ai_decide` twice. `AsyncFuncExec` runs the copies back to back
/// on the same batch, so the copies are answered from here instead of
/// sending every request again.
#[derive(Default)]
pub(crate) struct LastCall {
    inner: Mutex<Option<CachedCall>>,
}

/// Doesn't print the cached inputs.
impl std::fmt::Debug for LastCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LastCall").finish_non_exhaustive()
    }
}

struct CachedCall {
    key: CallKey,
    inputs: Vec<Option<String>>,
    rows: Rows,
    at: Instant,
}

/// Everything besides the inputs that the answers depend on.
#[derive(Debug, PartialEq)]
struct CallKey {
    endpoint: String,
    api: DecisionApi,
    model: String,
    questions: Vec<Question>,
    fail_on_error: bool,
}

impl LastCall {
    fn get(&self, key: &CallKey, inputs: &[Option<String>]) -> Option<Rows> {
        let cached = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        cached
            .as_ref()
            .filter(|c| c.at.elapsed() < LAST_CALL_TTL && c.key == *key && c.inputs == inputs)
            .map(|c| c.rows.clone())
    }

    fn put(&self, key: CallKey, inputs: Vec<Option<String>>, rows: Rows) {
        *self.inner.lock().unwrap_or_else(|e| e.into_inner()) = Some(CachedCall {
            key,
            inputs,
            rows,
            at: Instant::now(),
        });
    }
}

/// The input as text per row. Strings are sent as they are, nested values
/// (structs, lists, maps) as JSON, anything else cast to a string.
fn input_rows(input: &ColumnarValue, num_rows: usize) -> Result<Vec<Option<String>>> {
    let array = input.to_array(num_rows)?;
    let t = array.data_type();
    if *t == DataType::Null {
        return Ok(vec![None; num_rows]);
    }
    if t.is_nested() {
        return json_rows(&array);
    }
    let s = to_utf8(&array)?;
    Ok((0..s.len())
        .map(|i| s.is_valid(i).then(|| s.value(i).to_string()))
        .collect())
}

fn json_rows(array: &ArrayRef) -> Result<Vec<Option<String>>> {
    let batch = RecordBatch::try_from_iter([("v", Arc::clone(array))])?;
    let mut writer = LineDelimitedWriter::new(Vec::new());
    writer.write(&batch)?;
    writer.finish()?;
    let buf = writer.into_inner();
    let mut lines = buf.split(|b| *b == b'\n');
    (0..array.len())
        .map(|i| {
            let line = lines.next().unwrap_or_default();
            if array.is_null(i) {
                return Ok(None);
            }
            let row: Value =
                serde_json::from_slice(line).map_err(|e| DataFusionError::External(Box::new(e)))?;
            Ok(Some(row["v"].to_string()))
        })
        .collect()
}

/// Everything shared by the requests of one call on one batch.
struct Call<'a> {
    name: &'static str,
    client: &'a OpenAiClient,
    api: DecisionApi,
    model: &'a str,
    questions: &'a [Question],
    fail_on_error: bool,
}

impl Call<'_> {
    /// Answers for each row. Identical inputs are asked once. A failed
    /// request (or, with `fail_on_error`, a refusal) fails the call, or
    /// gives a NULL row otherwise.
    async fn run(
        &self,
        concurrency: usize,
        inputs: &[Option<String>],
    ) -> Result<Vec<Option<Arc<Vec<Answer>>>>> {
        map_unique_rows(
            concurrency,
            inputs.len(),
            |i| inputs[i].as_deref(),
            |row| async move {
                let input = inputs[row].as_deref().unwrap_or_default();
                match self.ask(input).await {
                    Ok(a) => Ok(Some(Arc::new(a))),
                    Err(e) if self.fail_on_error => Err(e),
                    Err(e) => {
                        log::warn!("{} request failed, returning NULL: {e}", self.name);
                        Ok(None)
                    }
                }
            },
        )
        .await
    }

    async fn ask(&self, input: &str) -> Result<Vec<Answer>> {
        let answers = decide(self.client, self.api, self.model, input, self.questions).await?;
        if self.fail_on_error
            && let Some(i) = answers.iter().position(|a| *a == Answer::Refusal)
        {
            return exec_err!(
                "{}: the model declined to answer question '{}'",
                self.name,
                self.questions[i].name
            );
        }
        Ok(answers)
    }
}

/// The first answer of each row, `None` for NULL rows and refusals.
pub(crate) fn first_answers(
    rows: &[Option<Arc<Vec<Answer>>>],
) -> impl Iterator<Item = Option<&Answer>> {
    rows.iter().map(|r| {
        r.as_ref()
            .and_then(|a| a.first())
            .filter(|a| **a != Answer::Refusal)
    })
}

/// Defines a decision UDF struct backed by the [`DecisionFn`] `$f`.
macro_rules! decision_udf {
    ($(#[$meta:meta])* $t:ident, $f:ty) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $t {
            signature: datafusion::logical_expr::Signature,
            clients: std::sync::Arc<crate::openai::ClientCache>,
            last_call: crate::udf::decision::LastCall,
        }

        impl Default for $t {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $t {
            pub fn new() -> Self {
                Self::with_clients(Default::default())
            }

            /// Use `clients` instead of a private HTTP client cache, so
            /// several functions share one connection pool.
            pub(crate) fn with_clients(clients: std::sync::Arc<crate::openai::ClientCache>) -> Self {
                Self {
                    signature: crate::udf::decision::signature::<$f>(),
                    clients,
                    last_call: Default::default(),
                }
            }
        }

        impl PartialEq for $t {
            fn eq(&self, _other: &Self) -> bool {
                true
            }
        }

        impl Eq for $t {}

        impl std::hash::Hash for $t {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                std::hash::Hash::hash(<$f as crate::udf::decision::DecisionFn>::NAME, state);
            }
        }

        impl datafusion::logical_expr::ScalarUDFImpl for $t {
            fn name(&self) -> &str {
                <$f as crate::udf::decision::DecisionFn>::NAME
            }

            fn signature(&self) -> &datafusion::logical_expr::Signature {
                &self.signature
            }

            fn return_type(
                &self,
                _arg_types: &[datafusion::arrow::datatypes::DataType],
            ) -> datafusion::common::Result<datafusion::arrow::datatypes::DataType> {
                <$f as crate::udf::decision::DecisionFn>::fixed_type().ok_or_else(|| {
                    datafusion::common::internal_datafusion_err!(
                        "{}: use return_field_from_args",
                        <$f as crate::udf::decision::DecisionFn>::NAME
                    )
                })
            }

            fn return_field_from_args(
                &self,
                args: datafusion::logical_expr::ReturnFieldArgs,
            ) -> datafusion::common::Result<datafusion::arrow::datatypes::FieldRef> {
                crate::udf::decision::return_field::<$f>(args)
            }

            fn invoke_with_args(
                &self,
                _args: datafusion::logical_expr::ScalarFunctionArgs,
            ) -> datafusion::common::Result<datafusion::logical_expr::ColumnarValue> {
                datafusion::common::exec_err!(
                    "{} can only be invoked asynchronously",
                    <$f as crate::udf::decision::DecisionFn>::NAME
                )
            }
        }

        #[async_trait::async_trait]
        impl datafusion::logical_expr::async_udf::AsyncScalarUDFImpl for $t {
            async fn invoke_async_with_args(
                &self,
                args: datafusion::logical_expr::ScalarFunctionArgs,
            ) -> datafusion::common::Result<datafusion::logical_expr::ColumnarValue> {
                crate::udf::decision::invoke::<$f>(&self.clients, &self.last_call, args).await
            }
        }
    };
}
pub(crate) use decision_udf;

#[cfg(test)]
mod tests {
    use super::*;

    fn lits(values: Vec<Option<ScalarValue>>) -> (Vec<Option<ScalarValue>>, &'static [Arg]) {
        (values, ai_score::ScoreFn::ARGS)
    }

    fn criteria(v: ScalarValue, min: usize, max: usize) -> Result<Vec<Criterion>> {
        let (values, args) = lits(vec![None, None, Some(v)]);
        let l = Literals {
            name: "f",
            args,
            values: values.iter().map(Option::as_ref).collect(),
            options: Vec::new(),
        };
        l.criteria("levels", min, max)
    }

    fn labels(c: &[Criterion]) -> Vec<&str> {
        c.iter().map(|c| c.label.as_str()).collect()
    }

    #[test]
    fn parses_criteria() {
        let list = ScalarValue::List(ScalarValue::new_list_nullable(
            &[ScalarValue::from("a"), ScalarValue::from("b")],
            &DataType::Utf8,
        ));
        assert_eq!(labels(&criteria(list, 2, 10).unwrap()), ["a", "b"]);
        let c = criteria(ScalarValue::from(r#"{"z": "last", "a": null}"#), 2, 10).unwrap();
        assert_eq!(labels(&c), ["z", "a"]);
        assert_eq!(c[0].description.as_deref(), Some("last"));
        assert_eq!(c[1].description, None);
        assert_eq!(
            labels(&criteria(ScalarValue::from(r#"["x", "y"]"#), 2, 10).unwrap()),
            ["x", "y"]
        );

        for bad in [
            r#"["a"]"#,
            r#"["a", "a"]"#,
            r#"["a", " "]"#,
            r#"["a", 1]"#,
            "a,b",
            "1",
        ] {
            assert!(criteria(ScalarValue::from(bad), 2, 10).is_err(), "{bad}");
        }
        let eleven: Vec<_> = (0..11).map(|i| format!("\"{i}\"")).collect();
        let eleven = format!("[{}]", eleven.join(","));
        assert!(criteria(ScalarValue::from(eleven.as_str()), 2, 10).is_err());
        assert!(criteria(ScalarValue::Int32(Some(1)), 2, 10).is_err());
        let ints = ScalarValue::List(ScalarValue::new_list_nullable(
            &[ScalarValue::Int32(Some(1)), ScalarValue::Int32(Some(2))],
            &DataType::Int32,
        ));
        assert!(criteria(ints, 2, 10).is_err());
    }

    #[test]
    fn parses_on_error() {
        assert!(parse_on_error("f", "FAIL").unwrap());
        assert!(!parse_on_error("f", " null ").unwrap());
        assert!(parse_on_error("f", "skip").is_err());
    }

    #[test]
    fn converts_nested_input_to_json() {
        use datafusion::arrow::array::{Int32Array, StringArray, StructArray};
        let s = StructArray::from(vec![
            (
                Arc::new(Field::new("id", DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("body", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec![Some("hi"), None])) as ArrayRef,
            ),
        ]);
        let rows = input_rows(&ColumnarValue::Array(Arc::new(s)), 2).unwrap();
        assert_eq!(
            rows,
            [
                Some(r#"{"id":1,"body":"hi"}"#.into()),
                Some(r#"{"id":2}"#.into())
            ]
        );
        let n = input_rows(&ColumnarValue::Scalar(ScalarValue::Int64(Some(7))), 2).unwrap();
        assert_eq!(n, [Some("7".into()), Some("7".into())]);
    }
}

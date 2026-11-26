# datafusion_ai_functions [![Basic Flow](https://github.com/milenkovicm/datafusion_ai_functions/actions/workflows/basic.yml/badge.svg)](https://github.com/milenkovicm/datafusion_ai_functions/actions/workflows/basic.yml)

AI SQL functions for [Apache DataFusion](https://datafusion.apache.org/). They call an LLM through any OpenAI-compatible Chat Completions API (OpenAI, Ollama, vLLM, LiteLLM, ...), or a decision model through OpenAI's Decisions API or Ollama's `systemone` endpoint.

> [!WARNING]
> This is a vibe-coded solution built as a proof of concept. It has not been fully reviewed, so use it with caution.

> [!NOTE]
> [spark_ai_functions](https://github.com/milenkovicm/spark_ai_functions) provides similar functionality for [Apache Spark](https://spark.apache.org).

| Function | Purpose |
| --- | --- |
| `ai_query(prompt [, options])` | Send a prompt, get the model's answer |
| `ai_complete(prompt [, options])` | Like `ai_query`, with optional response metadata |
| `ai_classify(content, labels [, options])` | Pick the matching label(s) for a text |
| `ai_summarize(content [, max_words])` | Summarize a text |
| `ai_analyze_sentiment(content)` | `positive`, `negative`, `neutral` or `mixed` |
| `ai_probability(input, condition [, options])` | Probability that a statement holds (decision model) |
| `ai_if(input, condition [, options])` | Whether that probability is above 0.5 (decision model) |
| `ai_score(input, instructions, levels [, options])` | Rate on ordered levels (decision model) |
| `ai_decide_classify(input, labels [, options])` | Pick one label (decision model) |
| `ai_decide(input, questions [, options])` | Several questions in one request (decision model) |

All functions are DataFusion async scalar UDFs (`AsyncScalarUDFImpl`), so HTTP requests don't block the executor. In the physical plan they show up as `AsyncFuncExec`.

## Usage

```rust
use datafusion::prelude::SessionContext;
use datafusion_ai_functions::register_ai_functions;

let ctx = SessionContext::new();
register_ai_functions(&ctx); // registers the UDFs and their `datafusion_ai_functions.*` config options

ctx.sql("SET datafusion_ai_functions.endpoint = 'http://localhost:11434/v1'").await?;
ctx.sql("SET datafusion_ai_functions.default_model = 'qwen2.5:14b-instruct'").await?; // used by all functions
ctx.sql("SELECT ai_query('Say hi')").await?.show().await?;
```

You can also configure it in code instead of with SQL:

```rust
use datafusion_ai_functions::{AiQueryConfig, register_ai_functions, session_config_with_ai_query};

let mut cfg = AiQueryConfig::from_env(); // reads OPENAI_BASE_URL / OPENAI_API_KEY
cfg.default_model = "gpt-4o-mini".into();
cfg.max_concurrency = 16;
let ctx = SessionContext::new_with_config(session_config_with_ai_query(cfg));
register_ai_functions(&ctx); // keeps the config above, adds defaults only if none is set
```

`register_ai_functions` registers all functions, and they share one HTTP client. To register only some of them, use `register_ai_query` (just `ai_query`, plus default config) or the individual UDFs: `ai_query_udf()`, `ai_complete_udf()`, `ai_classify_udf()`, `ai_summarize_udf()`, `ai_analyze_sentiment_udf()`, `ai_probability_udf()`, `ai_if_udf()`, `ai_score_udf()`, `ai_decide_classify_udf()` and `ai_decide_udf()`. UDFs created this way don't share a client.

## Common behaviour

This section is about the chat functions. The decision functions have their own rules, see [Decision functions](#decision-functions).

- Text arguments accept `Utf8`, `LargeUtf8` and `Utf8View`. A NULL text gives a NULL result and sends no request.
- Each non-NULL row sends one chat completion request. `ai_classify`, `ai_analyze_sentiment` and `ai_summarize` instead ask each distinct input (with the same settings) once per record batch. The result is the content of the first choice.
- `options` is an optional `MAP<STRING, STRING>`. Keys are case-insensitive, and a NULL value counts as not set. Options are read per row, so `MAP('model', model_col)` lets each row use a different model.
- Unknown option keys, invalid values, or no model configured fail the query before any request is sent.
- The model is the `model` option where the function has one, else `datafusion_ai_functions.default_model`.
- Request errors follow `datafusion_ai_functions.fail_on_error`, or the per-call option where there is one: either the query fails or the row is NULL.
- An answer that hits the token limit before any text counts as a request error. This happens with reasoning models that spend `max_tokens` on thinking. An answer cut off part way is returned as is, with a warning in the log.
- All functions return `Utf8`. Structured results are JSON strings; parse them with JSON functions or `CAST` them.
- All functions are `Volatile`, so DataFusion never constant-folds or caches them.

## Functions

### ai_query

`ai_query(prompt [, options]) -> Utf8`

`datafusion_ai_functions.system_prompt` is sent as the system message.

| Key | Default | Description |
| --- | --- | --- |
| `model` | `datafusion_ai_functions.default_model` | Model name sent to the endpoint |
| `max_tokens` | `datafusion_ai_functions.max_tokens` | Passed as `max_completion_tokens` |
| `temperature` | `datafusion_ai_functions.temperature` | Sampling temperature |
| `responseFormat` | none | `text`, `json_object`, or an OpenAI `response_format` JSON object, e.g. `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": true}}` |
| `failOnError` | `datafusion_ai_functions.fail_on_error` | `false` turns a failed request into NULL for that row, `true` fails the query |

```sql
SELECT ai_query(
    'Extract the main topic of: ' || text,
    MAP(
        'model', 'qwen2.5:1.5b',
        'max_tokens', '100',
        'failOnError', 'false',
        'responseFormat', '{"type": "json_schema", "json_schema": {"name": "topic", "strict": true,
            "schema": {"type": "object", "properties": {"topic": {"type": "string"}},
                       "required": ["topic"], "additionalProperties": false}}}'
    )
) AS extracted
FROM docs;
-- {"topic": "Apache DataFusion"}
```

### ai_complete

`ai_complete(prompt [, options]) -> Utf8`

`datafusion_ai_functions.system_prompt` is sent as the system message.

| Key | Default | Description |
| --- | --- | --- |
| `model` | `datafusion_ai_functions.default_model` | Model name sent to the endpoint |
| `temperature` | `datafusion_ai_functions.temperature` | Sampling temperature |
| `top_p` | unset | Nucleus sampling, `0` to `1` |
| `max_tokens` | `datafusion_ai_functions.max_tokens` | Passed as `max_completion_tokens` |
| `response_format` | none | `{"type": "json", "schema": {...}}`, sent as an OpenAI `json_schema` format named `response`. `text`, `json_object` and a full OpenAI `response_format` JSON object also work. |
| `show_details` | `false` | `true` returns the answer with metadata (see below) |
| `fail_on_error` | `datafusion_ai_functions.fail_on_error` | `false` turns a failed request into NULL for that row |

```sql
SELECT ai_complete(
    'Extract the order details from this customer note: ' || note,
    MAP(
        'temperature', '0',
        'response_format', '{"type": "json", "schema": {"type": "object",
            "properties": {"items_count": {"type": "number"}, "address": {"type": "string"}},
            "required": ["items_count", "address"]}}'
    )
) AS extracted
FROM notes;
-- {"items_count": 3, "address": "1 Main St"}
```

With `show_details = 'true'` the result is JSON with metadata:

```json
{"choices":[{"messages":"Snowflakes form when..."}],"created":1708536426,"model":"m",
 "usage":{"completion_tokens":10,"prompt_tokens":22,"total_tokens":32}}
```

With `show_details` and `response_format`, the parsed answer goes into `structured_output` instead of `choices`:

```json
{"structured_output":[{"raw_message":{"items_count":3,"address":"1 Main St"},"type":"json"}],
 "created":1708536426,"model":"m","usage":{"completion_tokens":20,"prompt_tokens":40,"total_tokens":60}}
```

`usage` is `{}` if the endpoint doesn't report it. If the structured answer isn't valid JSON, `raw_message` holds it as a string.

### ai_classify

`ai_classify(content, labels [, options]) -> Utf8` (label, or JSON)

- `labels` is a JSON string: either an array of label names (`'["urgent", "not_urgent"]'`) or an object that maps each label to a description (`'{"billing_issue": "payments and refunds", ...}'`). You need 2 to 500 unique labels of 1 to 100 characters, and descriptions can be up to 1000 characters. Invalid labels fail the query.
- If `labels` is NULL, the result is NULL and no request is sent.

| Key | Default | Description |
| --- | --- | --- |
| `responseFormat` | see below | `text` returns just the label, `json` returns the JSON envelope |
| `version` | `2.1` | JSON format. `2.1` gives `"response": [{"value": "urgent"}]`. `2.0` gives `"response": ["urgent"]`. |
| `multilabel` | `false` | `true` returns every label that applies (needs JSON output) |
| `instructions` | none | Extra task or domain instructions added to the prompt |
| `model` | `datafusion_ai_functions.default_model` | Model to use |

```sql
SELECT ai_classify('My password is leaked.', '["urgent", "not_urgent"]');
-- urgent

SELECT ai_classify('My password is leaked.', '["urgent", "not_urgent"]', MAP('version', '2.1'));
-- {"response":[{"value":"urgent"}],"metadata":{"version":"2.1"},"error_message":null}
```

- The model is told to reply with only the label name(s). The reply is matched against the labels: exact match first, then case-insensitive, ignoring quotes, bullets and a trailing period.
- The result is just the label by default. It's JSON when `responseFormat` is `json`, or when `responseFormat` isn't set and `version` or `multilabel = 'true'` is.
- Text output is always a single label, so `responseFormat = 'text'` with `multilabel = 'true'` fails the query. With `responseFormat = 'text'`, `version` is ignored.
- If the reply matches no label, the text result is NULL. In JSON, `response` is `null` and `error_message` explains why.
- If a request fails and `datafusion_ai_functions.fail_on_error` is `false`, the text result is NULL and the JSON result has the request error in `error_message`.
- Requests use temperature `0` unless `datafusion_ai_functions.temperature` is set. `datafusion_ai_functions.system_prompt` isn't used.

### ai_summarize

`ai_summarize(content [, max_words]) -> Utf8`

```sql
SELECT ai_summarize(text, 20) AS summary FROM docs;
```

- `max_words` is an integer literal or column. It's a best-effort target, and the model may go slightly over it. The default is `50`, `0` means no limit, NULL uses the default, and a negative value fails the query.
- The model is told to reply with only the summary. Surrounding whitespace and quotes are removed.
- `datafusion_ai_functions.system_prompt` isn't used.

### ai_analyze_sentiment

`ai_analyze_sentiment(content) -> Utf8`

```sql
SELECT ai_analyze_sentiment('I am happy');
-- positive
```

- Returns `positive`, `negative`, `neutral` or `mixed`, or NULL if the model's reply isn't one of those (it's told to reply `unknown` when it can't tell).
- Requests use temperature `0` unless `datafusion_ai_functions.temperature` is set. `datafusion_ai_functions.system_prompt` isn't used.

## Decision functions

A decision model doesn't write text. It answers questions about an input with probabilities: how likely a statement is (`noul`, also called `predicate`), which option fits (`choice`), or which level on a scale fits (`score`). These functions follow the [Spice decision functions](https://spice.ai/blog/decision-models).

```sql
-- local Ollama with the `nimble` decision model
SET datafusion_ai_functions.decision_api = 'systemone';
SET datafusion_ai_functions.decision_endpoint = 'http://localhost:11434/v1';
SET datafusion_ai_functions.decision_model = 'nimble';

SELECT id, subject FROM tickets
WHERE status = 'open' AND ai_if(body, 'The customer is asking for a refund');

SELECT id, ai_probability(body, 'The customer threatens to cancel') AS churn_risk
FROM tickets ORDER BY churn_risk DESC;

SELECT id, ai_decide_classify(body, ['billing', 'technical', 'other'],
                              MAP('instructions', 'Which team should handle this ticket?', 'on_error', 'null')) AS team
FROM tickets;

SELECT id, ai_score(body, 'How frustrated is the customer?', ['calm', 'annoyed', 'furious']) AS frustration
FROM tickets ORDER BY frustration DESC;

SELECT id, d['team']['choice'] AS team, d['urgent']['probability'] AS p_urgent
FROM (SELECT id, ai_decide(body, '{
        "team":   {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments and refunds", "technical": null}},
        "urgent": {"type": "noul", "instructions": "The customer needs help today"}
      }') AS d FROM tickets);
```

| Function | Returns |
| --- | --- |
| `ai_probability(input, condition [, options])` | `Float64` from 0 to 1 |
| `ai_if(input, condition [, options])` | `Boolean`: true when the probability is above 0.5 |
| `ai_score(input, instructions, levels [, options])` | `Float64` from 0 to n-1: the probability-weighted 0-based index of the level. `levels` has 2 to 10 levels, lowest first. |
| `ai_decide_classify(input, labels [, options])` | `Utf8`, always one of the 2 to 255 `labels` |
| `ai_decide(input, questions [, options])` | `Struct` with one field per question |

- `input` can be text, which is sent as is, or a struct, list or map, which is sent as JSON. Other types are cast to text, and binary fails the query. A NULL input gives NULL and sends no request.
- Every other argument must be a constant, including the `options` MAP. Bad labels, levels or questions fail the query before any request is sent.
- `labels` and `levels` are a list literal (`['billing', 'technical']`) or a JSON string. The JSON can be an array of names or an object of name to description (or `null`), e.g. `'{"billing": "Payments and refunds", "technical": null}'`.
- `questions` (for `ai_decide`) is a JSON object of 1 to 200 questions, keyed by name. Each has a `type` (`noul`, `predicate`, `choice` or `score`), optional `instructions` (the name by default), and for `choice` and `score` its `criteria`, in the same format as `labels`.
- `ai_decide` fields: `noul` gives `{probability}`, `choice` gives `{choice, confidence, probabilities}`, and `score` gives `{score, confidence, probabilities}`. `probabilities` is a `Map<Utf8, Float64>` of option or level to probability.
- `options` is an optional `MAP<STRING, STRING>`, e.g. `ai_if(body, 'is spam', MAP('model', 'nimble', 'on_error', 'null'))`. Keys are case-insensitive, a NULL value counts as not set, and unknown keys fail the query.

| Key | Default | Description |
| --- | --- | --- |
| `model` | `datafusion_ai_functions.decision_model` | Decision model. One of them must be set. |
| `on_error` | `datafusion_ai_functions.fail_on_error` | `'fail'` fails the query, `'null'` returns NULL. When the model declines a question, that's an error too. With `'null'`, the row is NULL, or in `ai_decide` only that question's field is. |
| `instructions` | `Which label best fits the input?` | `ai_decide_classify` only: the question asked about the labels |

- Each non-NULL row sends one request, and identical inputs in a batch are asked once. `ai_decide` asks all of its questions in a single request. Two separate functions on the same input still send two requests.
- DataFusion may copy a call once for each field you access, e.g. for `d['a']` and `d['b']`. The copies reuse the answers of a recent identical call, so they don't send the requests again. Each function keeps its last 8 calls for 5 seconds, so an identical call (same function, arguments, settings and batch) within 5 seconds also reuses those answers.
- `datafusion_ai_functions.system_prompt`, `temperature` and `max_tokens` aren't used.

### Decision endpoint

| `decision_api` | Request |
| --- | --- |
| `openai` (default) | `POST {decision_endpoint}/decisions` (OpenAI Decisions API) |
| `systemone` | `POST {decision_endpoint}/systemone` (Ollama, e.g. `nimble`) |

`decision_endpoint` and `decision_secret` fall back to `endpoint` and `secret`. Timeouts, retries and `max_concurrency` are the same as for the chat functions.

### ai_parse_document & ai_extract

To be added soon

## Configuration

Options are DataFusion session config options under the `datafusion_ai_functions.` prefix. They're read when the query runs, so a `SET` takes effect on the next query.

| Option | Default | Description |
| --- | --- | --- |
| `datafusion_ai_functions.endpoint` | `https://api.openai.com/v1` | OpenAI-compatible base URL, including the version path (`/v1`) |
| `datafusion_ai_functions.secret` | empty | Sent as `Authorization: Bearer <token>`. When empty, no auth header is sent. Like every session option it's visible in `SHOW ALL` and `information_schema.df_settings`. |
| `datafusion_ai_functions.max_concurrency` | `8` | Maximum requests in flight at once for one function call on one record batch (see [Request scheduling](#request-scheduling)) |
| `datafusion_ai_functions.timeout_ms` | `60000` | Timeout for each request |
| `datafusion_ai_functions.max_retries` | `3` | Retries with exponential backoff on 429, 5xx and connect errors |
| `datafusion_ai_functions.fail_on_error` | `true` | `true` fails the query on a request error. `false` returns NULL for that row and logs a warning. |
| `datafusion_ai_functions.system_prompt` | empty | Optional system message for `ai_query` and `ai_complete` |
| `datafusion_ai_functions.max_tokens` | unset | Passed as `max_completion_tokens` |
| `datafusion_ai_functions.temperature` | unset | Sampling temperature |
| `datafusion_ai_functions.default_model` | empty | Model used by all chat functions unless a call sets the `model` option. Either one must be set. |
| `datafusion_ai_functions.decision_api` | `openai` | `openai` or `systemone`, see [Decision endpoint](#decision-endpoint) |
| `datafusion_ai_functions.decision_endpoint` | empty | Base URL for decision functions. When empty, `endpoint` is used. |
| `datafusion_ai_functions.decision_secret` | empty | Token for decision functions. When empty, `secret` is used. |
| `datafusion_ai_functions.decision_model` | empty | Decision model used unless a call sets the `model` option |
| `datafusion_ai_functions.single_partition` | `true` | Evaluate AI functions on one partition. Only has an effect when the `AiSinglePartition` rule is installed, see [Single partition](#single-partition). |

The HTTP client is cached and reused across batches and queries, and `register_ai_functions` shares one client between all functions. A new client is built only for a new combination of `endpoint`, `secret`, `timeout_ms` and `max_retries`. The last few combinations are kept.

### Request scheduling

Rows are **not** batched into API calls. With `ai_query` and `ai_complete`, each non-NULL row sends its own chat completion request, so 1,000 rows means 1,000 requests.

- At most `max_concurrency` requests are in flight at once. When any one finishes, the next row's request starts, so a slow or retrying request doesn't hold up the others.
- `max_concurrency` applies per partition. A query that evaluates a function on 8 partitions can have 8 × `max_concurrency` requests in flight, unless the [single partition](#single-partition) rule is installed.
- Results keep the row order.
- With `fail_on_error = true`, the first failed request stops the call: no more requests are sent and in-flight ones are cancelled.
- `ai_classify`, `ai_analyze_sentiment` and `ai_summarize` ask identical inputs once per record batch, and every matching row gets that answer. Inputs count as identical when the text and the settings (labels and options, or `max_words`) are the same. `ai_query` and `ai_complete` don't deduplicate, so a value that appears many times costs one request per row.

The Chat Completions API has no multi-prompt request. The OpenAI Batch API is asynchronous (it can take up to 24 hours), so it doesn't fit a query running inside a UDF.

### Single partition

AI functions are limited by the model endpoint, not by local CPU, so running them on many partitions mostly multiplies the requests in flight. The `AiSinglePartition` physical optimizer rule merges the input of every `AsyncFuncExec` that calls an AI function into one partition, which makes `max_concurrency` the limit for the whole query.

- It inserts a `CoalescePartitionsExec`, or a `SortPreservingMergeExec` when the input is sorted, below the `AsyncFuncExec`.
- A round robin repartition of a single partition is removed instead.
- If the `AsyncFuncExec` output was hash partitioned, the hash partitioning is restored above it.
- Plans without AI functions aren't changed.

Physical optimizer rules can only be set while the session state is built, so `register_ai_functions` can't install it. Use `ai_session_context`, which also registers all functions:

```rust
use datafusion::prelude::SessionConfig;
use datafusion_ai_functions::ai_session_context;

let ctx = ai_session_context(SessionConfig::new());
```

Or add the rule to your own `SessionStateBuilder` with `with_ai_optimizer_rules`:

```rust
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_ai_functions::{register_ai_functions, with_ai_optimizer_rules};

let builder = SessionStateBuilder::new()
    .with_config(SessionConfig::new())
    .with_default_features();
let ctx = SessionContext::new_with_state(with_ai_optimizer_rules(builder).build());
register_ai_functions(&ctx);
```

With the rule installed, `SET datafusion_ai_functions.single_partition = false` turns it off for a session.

## Running the demo

The demo uses a local [Ollama](https://ollama.com/) server by default:

```sh
ollama run qwen2.5:1.5b
cargo run --example query
cargo run --example complete
```

Or point it at another endpoint:

```sh
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-... AI_MODEL=gpt-4o-mini \
  cargo run --example query
```

To run the `ai_classify`, `ai_summarize` and `ai_analyze_sentiment` examples (defaults to `qwen2.5:14b-instruct` on local Ollama):

```sh
ollama pull qwen2.5:14b-instruct
cargo run --example classify
cargo run --example summarize
cargo run --example sentiment
```

To run the decision function examples (defaults to `nimble` on local Ollama's `/v1/systemone`):

```sh
cargo run --example decisions
```

## Tests

```sh
cargo test
```

The integration tests in `tests/` use a `wiremock` mock OpenAI server, so they don't need network access or a real model.

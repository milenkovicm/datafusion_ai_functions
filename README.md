# datafusion_ai_functions [![Basic Flow](https://github.com/milenkovicm/datafusion_ai_functions/actions/workflows/basic.yml/badge.svg)](https://github.com/milenkovicm/datafusion_ai_functions/actions/workflows/basic.yml)

AI SQL functions for [Apache DataFusion](https://datafusion.apache.org/). They call an LLM through any OpenAI-compatible Chat Completions API (OpenAI, Ollama, vLLM, LiteLLM, ...).

| Function | Purpose |
| --- | --- |
| `ai_query(prompt [, options])` | Send a prompt, get the model's answer |
| `ai_complete(prompt [, options])` | Like `ai_query`, with optional response metadata |
| `ai_classify(content, labels [, options])` | Pick the matching label(s) for a text |
| `ai_summarize(content [, max_words])` | Summarize a text |
| `ai_analyze_sentiment(content)` | `positive`, `negative`, `neutral` or `mixed` |

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
use datafusion_ai_functions::{AiQueryConfig, ai_query_udf, session_config_with_ai_query};

let mut cfg = AiQueryConfig::from_env(); // reads OPENAI_BASE_URL / OPENAI_API_KEY
cfg.default_model = "gpt-4o-mini".into();
cfg.max_concurrency = 16;
let ctx = SessionContext::new_with_config(session_config_with_ai_query(cfg));
ctx.register_udf(ai_query_udf());
```

`register_ai_query` and `ai_query_udf` only register `ai_query`. Use `ai_complete_udf()`, `ai_classify_udf()`, `ai_summarize_udf()` and `ai_analyze_sentiment_udf()` for the others.

## Common behaviour

- Text arguments accept `Utf8`, `LargeUtf8` and `Utf8View`. A NULL text gives a NULL result and sends no request.
- Each non-NULL row sends one chat completion request. The result is the content of the first choice.
- `options` is an optional `MAP<STRING, STRING>`. Keys are case-insensitive, and a NULL value counts as not set. Options are read per row, so `MAP('model', model_col)` lets each row use a different model.
- Unknown option keys, invalid values, or no model configured fail the query before any request is sent.
- The model is the `model` option where the function has one, else `datafusion_ai_functions.default_model`.
- Request errors follow `datafusion_ai_functions.fail_on_error`, or the per-call option where there is one: either the query fails or the row is NULL.
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

### ai_parse_document & ai_extract

To be added soon

## Configuration

Options are DataFusion session config options under the `datafusion_ai_functions.` prefix. They're read when the query runs, so a `SET` takes effect on the next query.

| Option | Default | Description |
| --- | --- | --- |
| `datafusion_ai_functions.endpoint` | `https://api.openai.com/v1` | OpenAI-compatible base URL, including the version path (`/v1`) |
| `datafusion_ai_functions.api_token` | empty | Sent as `Authorization: Bearer <token>`. When empty, no auth header is sent. Like every session option it's visible in `SHOW ALL` and `information_schema.df_settings`. |
| `datafusion_ai_functions.max_concurrency` | `8` | Maximum requests in flight at once for one function call on one record batch (see [Request scheduling](#request-scheduling)) |
| `datafusion_ai_functions.timeout_ms` | `60000` | Timeout for each request |
| `datafusion_ai_functions.max_retries` | `3` | Retries with exponential backoff on 429, 5xx and connect errors |
| `datafusion_ai_functions.fail_on_error` | `true` | `true` fails the query on a request error. `false` returns NULL for that row and logs a warning. |
| `datafusion_ai_functions.system_prompt` | empty | Optional system message for `ai_query` and `ai_complete` |
| `datafusion_ai_functions.max_tokens` | unset | Passed as `max_completion_tokens` |
| `datafusion_ai_functions.temperature` | unset | Sampling temperature |
| `datafusion_ai_functions.default_model` | empty | Model used by all functions unless a call sets the `model` option. Either one must be set. |

The HTTP client is cached and reused across batches and queries, and `register_ai_functions` shares one client between all functions. A new client is built only for a new combination of `endpoint`, `api_token`, `timeout_ms` and `max_retries`. The last few combinations are kept.

### Request scheduling

Rows are **not** batched into API calls. Each non-NULL row sends its own chat completion request, so 1,000 rows means 1,000 requests.

- At most `max_concurrency` requests are in flight at once. When one finishes, the next row's request starts, so a slow or retrying request doesn't hold up the others.
- Results keep the row order.
- With `fail_on_error = true`, the first failed request stops the call: no more requests are sent and in-flight ones are cancelled.
- Identical inputs aren't deduplicated. A value that appears many times in a column costs one request per row.

The Chat Completions API has no multi-prompt request. The OpenAI Batch API is asynchronous (it can take up to 24 hours), so it doesn't fit a query running inside a UDF.

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

## Tests

```sh
cargo test
```

The integration tests in `tests/` use a `wiremock` mock OpenAI server, so they don't need network access or a real model.

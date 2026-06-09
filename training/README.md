# Training Data Plan

This folder contains the plan and future artifacts for generating training and test data for `smolnalysis`.

## Goal

The final app should answer user questions about data available in the Open Data Portal of the City of Munich.

High-level flow:

```text
User question
  -> model emits a CKAN/API tool call
  -> app executes the tool call and persists a dataframe
  -> model emits dataframe filter/aggregation parameters
  -> app validates and executes the dataframe query
  -> model emits validated OpenUI JSON/OpenUI-Lang from the result
  -> Gradio frontend renders the OpenUI components
```

The model should not memorize Munich open data. It should learn three structured behaviors:

1. Choose the right API/tool call for a user request.
2. Generate valid dataframe query parameters for filtering, selecting, sorting, and aggregating persisted dataframes.
3. Convert filtered/aggregated tabular results into valid OpenUI components.

Munich's open-data portal exposes CKAN-style JSON APIs, including dataset search and dataset/resource endpoints. This makes it a good fit for tool calling rather than memorization.

Reference: https://opendata.muenchen.de/pages/hilfe

## Recommended Model Setup

Primary base model:

```text
Qwen/Qwen3-4B-Instruct-2507
```

Why:

- Small enough to fit the hackathon's small-model framing.
- Strong multilingual capability for German and English queries.
- Good function/tool-calling support.
- Suitable for LoRA/QLoRA fine-tuning.

Alternative:

```text
HuggingFaceTB/SmolLM3-3B
```

Use this if the submission should lean harder into the Hugging Face "smol" story or a tiny-model prize angle.

Recommended first implementation:

```text
Base model: Qwen3-4B-Instruct-2507
Fine-tuning: one QLoRA/PEFT adapter
Adapter task mix:
  - 30% CKAN tool-calling examples
  - 35% dataframe query examples
  - 25% OpenUI generation examples
  - 10% repair/fallback/negative examples
```

Use explicit task routing prompts:

```text
Task: select_ckan_tool
```

```text
Task: generate_dataframe_query
```

```text
Task: render_openui
```

This keeps one model running while the app hardcodes the chain behavior for testing. Later, the app can add a router or separate LoRA adapters if needed.

## Proposed Tool Layer

Expose a small wrapper around the CKAN API instead of asking the model to invent URLs.

Initial tools:

```text
search_datasets(query, tags?, groups?, limit?)
get_dataset(package_id)
inspect_resource(resource_id)
fetch_resource(resource_id, limit?) -> dataframe_id
query_dataframe(dataframe_id, operation_spec) -> result_dataframe_id + result profile + sample rows
```

The model should emit structured tool calls only. The Python backend should validate tool names and arguments before executing them.

Implemented training helpers:

```text
training/scripts/open_data_tools.py
training/scripts/build_simple_query_context.py
training/scripts/generate_user_queries.py
training/scripts/generate_get_filter_datasets.py
```

The current fine-tuning data for the "get and filter" task is split into two tool-specific datasets:

```text
training/data/generated/retrieval_query_tool.train.jsonl
training/data/generated/retrieval_query_tool.validation.jsonl
training/data/generated/retrieval_query_tool.test.jsonl
training/data/generated/filter_tool.train.jsonl
training/data/generated/filter_tool.validation.jsonl
training/data/generated/filter_tool.test.jsonl
```

These are chat-style JSONL files suitable for LoRA SFT. The assistant target is a JSON string containing `tool_name` and `arguments`.

Use-case split:

```text
1. Retrieval adapter:
   natural-language user request
     -> search_open_data(query, limit)

2. Filter adapter:
   same user request + fetched dataframe profile/columns/operators/example values
     -> query_dataframe(dataframe_id, operation_spec)
```

The retrieval adapter must not receive candidate dataset metadata in the prompt. It should learn to turn a user request like "wieviele toiletten wurden im jahr 2020 eroeffnet" into a processable CKAN search query. The filter adapter receives the dataframe schema after retrieval, which makes it much easier to generate valid filter parameters.

To generate more human-like seed queries with an OpenAI-compatible chat model:

First build the simplified LLM input context:

```bash
uv run python training/scripts/build_simple_query_context.py
```

This writes:

```text
training/data/generated/simple_query_context.jsonl
training/data/generated/simple_query_context.csv
```

Each row contains only the dataset name/title, description, resource metadata, columns, and one example row.

```bash
uv run python training/scripts/generate_user_queries.py \
  --model gpt-4o-mini \
  --queries-per-resource 8 \
  --max-resources 50 \
  --overwrite
```

This writes:

```text
training/data/generated/llm_user_queries.jsonl
training/data/generated/llm_user_queries.summary.json
```

Each LLM prompt receives the dataset description, resource name, column names, dtypes, example values, and sample rows. The output contains natural user questions plus a short language and intent label. The model call follows the same `ChatOpenAI(...).with_structured_output(..., method="json_schema")` pattern used in `/DLF`.

### Dataframe Persistence

After `fetch_resource(...)`, the backend should materialize the resource into a dataframe and store it behind an ID.

Example persisted dataframe handle:

```json
{
  "dataframe_id": "df_001",
  "package_id": "vornamen-von-neugeborenen",
  "resource_id": "6388c83a-266d-437c-824a-7bbcb7ceec63",
  "row_count": 482,
  "columns": [
    {"name": "vorname", "dtype": "string", "examples": ["Felix", "Emma", "Clara"]},
    {"name": "anzahl", "dtype": "integer", "examples": [89, 81, 79]},
    {"name": "geschlecht", "dtype": "string", "examples": ["m", "w"]}
  ]
}
```

Do not train the model to write pandas code. Train it to emit a constrained dataframe operation spec that Python validates and executes.

Initial dataframe operation schema:

```json
{
  "dataframe_id": "df_001",
  "filters": [
    {
      "column": "geschlecht",
      "operator": "eq",
      "value": "w"
    }
  ],
  "group_by": ["geschlecht"],
  "aggregate": [
    {
      "column": "anzahl",
      "function": "sum",
      "alias": "total_anzahl"
    }
  ],
  "select": ["geschlecht", "total_anzahl"],
  "sort": [
    {
      "column": "total_anzahl",
      "direction": "desc"
    }
  ],
  "limit": 10
}
```

Supported operators:

```text
eq
neq
gt
gte
lt
lte
contains
in
between
```

Supported aggregations:

```text
count
sum
mean
median
min
max
```

The backend should reject invalid specs before execution:

```text
unknown column
invalid operator
wrong value type
invalid aggregate on non-numeric column
sort column not present
limit too large
empty result
ambiguous filter value
```

Structured error example:

```json
{
  "ok": false,
  "error_type": "unknown_column",
  "message": "Column 'gender' does not exist.",
  "available_columns": ["vorname", "anzahl", "geschlecht"],
  "suggestions": [
    {
      "column": "geschlecht",
      "reason": "Closest semantic match to gender."
    }
  ]
}
```

These errors should become repair examples in the training set.

## Data Artifacts

Create four dataset layers.

### 1. Catalog Snapshot

One JSONL row per Munich dataset/resource.

Example:

```json
{
  "package_id": "vornamen-von-neugeborenen",
  "title": "Vornamen von Neugeborenen",
  "notes": "Dataset description from the portal.",
  "tags": ["bevoelkerung", "vornamen"],
  "groups": ["society"],
  "resources": [
    {
      "resource_id": "abc123",
      "format": "CSV",
      "name": "Vornamen CSV",
      "url": "https://...",
      "schema": ["jahr", "vorname", "anzahl"],
      "sample_rows": [
        {"jahr": 2024, "vorname": "Emilia", "anzahl": 42}
      ]
    }
  ]
}
```

This is source material, not fine-tuning data.

### 2. Tool-Calling Dataset

Each example teaches the model to choose the right CKAN tool.

Single-turn example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You choose validated CKAN tools for Munich open data."
    },
    {
      "role": "user",
      "content": "Wie haben sich die beliebtesten Vornamen in Muenchen entwickelt?"
    },
    {
      "role": "assistant",
      "tool_call": {
        "name": "search_datasets",
        "arguments": {
          "query": "Vornamen Neugeborene Muenchen",
          "limit": 5
        }
      }
    }
  ]
}
```

Multi-turn example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You choose validated CKAN tools for Munich open data."
    },
    {
      "role": "user",
      "content": "Zeig mir die haeufigsten Babynamen."
    },
    {
      "role": "assistant",
      "tool_call": {
        "name": "search_datasets",
        "arguments": {
          "query": "Vornamen Neugeborene",
          "limit": 5
        }
      }
    },
    {
      "role": "tool",
      "name": "search_datasets",
      "content": "{\"results\": [{\"package_id\": \"vornamen-von-neugeborenen\", \"title\": \"Vornamen von Neugeborenen\"}]}"
    },
    {
      "role": "assistant",
      "tool_call": {
        "name": "fetch_resource",
        "arguments": {
          "resource_id": "abc123",
          "limit": 500
        }
      }
    }
  ]
}
```

### 3. Dataframe Query Dataset

Each example teaches the model to generate valid dataframe operation parameters from a user question and dataframe profile.

Input:

```json
{
  "task": "generate_dataframe_query",
  "user_question": "Show the most common girls' names.",
  "dataframe": {
    "dataframe_id": "df_001",
    "package_id": "vornamen-von-neugeborenen",
    "resource_id": "6388c83a-266d-437c-824a-7bbcb7ceec63",
    "row_count": 482,
    "columns": [
      {"name": "vorname", "dtype": "string", "role": "category", "examples": ["Felix", "Emma", "Clara"]},
      {"name": "anzahl", "dtype": "integer", "role": "metric", "examples": [89, 81, 79]},
      {"name": "geschlecht", "dtype": "string", "role": "category", "examples": ["m", "w"]}
    ]
  }
}
```

Target:

```json
{
  "dataframe_id": "df_001",
  "filters": [
    {
      "column": "geschlecht",
      "operator": "eq",
      "value": "w"
    }
  ],
  "select": ["vorname", "anzahl", "geschlecht"],
  "sort": [
    {
      "column": "anzahl",
      "direction": "desc"
    }
  ],
  "limit": 10
}
```

Generate dataframe query examples for:

```text
filter category == value
filter numeric > threshold
filter year/date == value
top-k by numeric column
group by category and count
group by category and sum numeric
trend over year/date
select relevant columns
sort ascending/descending
limit result
unsupported or ambiguous request
repair after invalid column/operator/filter error
```

Repair example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Generate valid dataframe query JSON. Use only provided columns and values."
    },
    {
      "role": "user",
      "content": "Question: Show the top girls' names.\n\nDataframe profile: {\"dataframe_id\":\"df_001\",\"columns\":[{\"name\":\"vorname\"},{\"name\":\"anzahl\"},{\"name\":\"geschlecht\"}]}"
    },
    {
      "role": "assistant",
      "content": "{\"dataframe_id\":\"df_001\",\"filters\":[{\"column\":\"gender\",\"operator\":\"eq\",\"value\":\"female\"}],\"sort\":[{\"column\":\"anzahl\",\"direction\":\"desc\"}],\"limit\":10}"
    },
    {
      "role": "tool",
      "name": "query_dataframe",
      "content": "{\"ok\":false,\"error_type\":\"unknown_column\",\"available_columns\":[\"vorname\",\"anzahl\",\"geschlecht\"],\"suggestions\":[{\"column\":\"geschlecht\",\"reason\":\"Closest semantic match to gender.\"}]}"
    },
    {
      "role": "assistant",
      "content": "{\"dataframe_id\":\"df_001\",\"filters\":[{\"column\":\"geschlecht\",\"operator\":\"eq\",\"value\":\"w\"}],\"select\":[\"vorname\",\"anzahl\",\"geschlecht\"],\"sort\":[{\"column\":\"anzahl\",\"direction\":\"desc\"}],\"limit\":10}"
    }
  ]
}
```

### 4. OpenUI Generation Dataset

Each example teaches the model to turn dataframe query results into UI components.

Input:

```json
{
  "task": "render_openui",
  "user_question": "Show the trend in newborn names over time.",
  "query_result": {
    "dataset_title": "Vornamen von Neugeborenen",
    "package_id": "vornamen-von-neugeborenen",
    "resource_id": "6388c83a-266d-437c-824a-7bbcb7ceec63",
    "dataframe_id": "df_001_result",
    "columns": ["jahr", "vorname", "anzahl"],
    "rows": [
      {"jahr": 2024, "vorname": "Emilia", "anzahl": 42},
      {"jahr": 2024, "vorname": "Noah", "anzahl": 39}
    ]
  }
}
```

Target:

```json
{
  "components": [
    {
      "type": "InsightCard",
      "title": "Newborn name trend",
      "text": "The dataset contains yearly counts of newborn first names in Munich."
    },
    {
      "type": "LineChart",
      "title": "Names over time",
      "x": "jahr",
      "y": "anzahl",
      "series": "vorname"
    },
    {
      "type": "DataTable",
      "title": "Source rows",
      "rows": [
        {"jahr": 2024, "vorname": "Emilia", "anzahl": 42},
        {"jahr": 2024, "vorname": "Noah", "anzahl": 39}
      ]
    }
  ],
  "sources": [
    {
      "title": "Vornamen von Neugeborenen",
      "package_id": "vornamen-von-neugeborenen"
    }
  ]
}
```

Prefer training the model to emit JSON first. Convert JSON to OpenUI-Lang deterministically in Python. This is easier to validate than free-form OpenUI-Lang.

The OpenUI model should receive compact query results, not the full raw dataframe. Statistics and aggregations should be computed by Python before the OpenUI generation step.

## Example Generation Strategy

Start with manually designed seed questions, then generate paraphrases and scenarios from the real catalog snapshot.

Target size for the first useful dataset:

```text
50 seed questions
x 5 paraphrases each
x 2-3 tool-result scenarios
= 500-750 examples
```

Target size for fine-tuning:

```text
train: 800-1500 examples
validation: 100-200 examples
test: 100-200 examples
```

Coverage categories:

```text
dataset search
dataset disambiguation
resource fetching
filtering by year/category/district
aggregation requests
dataframe query repair
chart requests
map/geospatial requests
schema inspection
missing/dirty data requests
unsupported requests
no matching dataset
ambiguous query
German query
English query
mixed German/English query
```

## Train/Test Split

Split by dataset/package ID, not by random rows.

Bad split:

```text
Same dataset appears in train and test with different question wording.
```

Good split:

```text
Entire dataset IDs are held out for validation and test.
```

This tests whether the model learned the tool-calling and rendering behavior instead of memorizing dataset names.

## Evaluation Metrics

Use automatic checks before doing any manual demo review.

Tool-calling metrics:

```text
tool_call_json_valid
tool_name_exact_match
required_arguments_present
selected_dataset_correct
selected_resource_correct
valid_filters
no_hallucinated_tool_names
```

Dataframe query metrics:

```text
dataframe_query_json_valid
known_dataframe_id
known_columns_only
valid_operator
valid_value_type
valid_aggregation
query_executes_without_error
query_result_non_empty_when_expected
repair_success_after_tool_error
```

OpenUI metrics:

```text
openui_json_valid
openui_schema_valid
component_type_correct
required_components_present
no_hallucinated_columns
source_dataset_included
renders_without_frontend_error
```

Manual test set:

```text
20-30 realistic demo questions
```

Keep these separate from training and validation. They should represent the final hackathon demo experience.

## Suggested Folder Structure

```text
training/
  README.md
  data/
    raw/
      munich_catalog.jsonl
    generated/
      tool_calls.train.jsonl
      tool_calls.validation.jsonl
      tool_calls.test.jsonl
      dataframe_queries.train.jsonl
      dataframe_queries.validation.jsonl
      dataframe_queries.test.jsonl
      dataframe_repairs.train.jsonl
      dataframe_repairs.validation.jsonl
      dataframe_repairs.test.jsonl
      openui.train.jsonl
      openui.validation.jsonl
      openui.test.jsonl
    eval/
      demo_questions.jsonl
      heldout_packages.txt
  scripts/
    fetch_catalog.py
    generate_tool_examples.py
    profile_dataframes.py
    generate_dataframe_query_examples.py
    generate_openui_examples.py
    validate_dataframe_query.py
    validate_examples.py
    evaluate_model_outputs.py
```

## Immediate Next Steps

1. Implement a CKAN catalog fetcher for the Munich portal.
2. Save package/resource metadata and sample rows to `training/data/raw/munich_catalog.jsonl`.
3. Implement dataframe persistence and a `query_dataframe(...)` backend function.
4. Define exact JSON schemas for CKAN tool calls, dataframe operation specs, dataframe repair responses, and OpenUI output.
5. Write validators before generating large amounts of data.
6. Create 50 high-quality seed questions manually.
7. Generate paraphrases, dataframe query scenarios, repair scenarios, and OpenUI render scenarios.
8. Evaluate base-model prompting before starting LoRA fine-tuning.

The first milestone is not a fine-tuned model. It is a reliable dataset generator plus validators. Fine-tuning should only start once the target formats and eval harness are stable.

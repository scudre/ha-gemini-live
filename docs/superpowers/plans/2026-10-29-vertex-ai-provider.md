# Google Vertex AI Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "Google Vertex AI" as a provider that authenticates with an uploaded service account JSON key and reuses the existing Gemini Live adapter.

**Architecture:** A new `vertex.py` module owns everything Vertex-specific: parsing the key, building the `genai.Client`, the setup-time connection test, and runtime error text. The config flow gains a Vertex branch, and Vertex STT and conversation entities subclass the Gemini ones and override only client creation and error text. TTS is unchanged because unknown providers already fall back to `GeminiLiveTTS`.

**Tech Stack:** Python 3.14, Home Assistant 2026.10 (config flow, `FileSelector`, `file_upload`, Repairs), `google-genai>=2.25.0`, `google-auth` (bundled with google-genai), pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-10-29-vertex-ai-provider-design.md`

## Global Constraints

- The feature is built on branch `vertex-ai-provider`, cut from upstream `8a851db`, in a worktree at `/Users/cscuderi/workspace/gemini_live/ha-gemini-live-vertex`. Fork-only commits never go on that branch. (Task 6 handles the fork's `main`.)
- Tests run with the venv in the main checkout: `/Users/cscuderi/workspace/gemini_live/ha-gemini-live/.venv/bin/python -m pytest -q`, run from the worktree root. Baseline: 196 passed.
- Provider value is `"vertex"`. The dropdown label is `"Google Vertex AI"`. The entry title and the entity `integration_name` are `"Gemini Live (Vertex AI)"`.
- Default location is `us`. Location options are `us`, `eu`, `global`, `us-central1`, `us-east4`, `europe-west4`, `asia-northeast1`, and custom values are allowed.
- Model options are `gemini-3.8-live` (default) and `gemini-live-2.5-flash-native-audio`, and custom values are allowed.
- Required key fields are `type == "service_account"`, `client_email`, `private_key`, `token_uri` and `project_id`. The project comes from `project_id`, never from a form field.
- Never log or display key contents, including in error messages.
- Google's errors are never classified. Setup and runtime pass `str(exc)` through verbatim, falling back to the exception class name when the string is empty.
- Vertex entries never show AI Studio wording or links (`ai.dev`, `ai.studio`, "prepayment credits", "Google AI project").
- There is no reauth flow.
- AI Studio (`gemini`), OpenAI, Qwen and PersonaPlex behaviour stays byte-for-byte the same. All 196 existing tests must keep passing.
- `strings.json` and `translations/en.json` are kept identical.
- Match upstream style: module docstrings, `# noqa: PLC0415` on lazy Google imports, and sparse comments.

## Review Focus

1. **Reconfigure or options submitted without a new upload.** The stored key must be kept. Reconfigure calls `async_update_reload_and_abort(..., options={})`, so the credentials must be put back into `user_input`. Tested in Task 4 by `test_reconfigure_without_upload_keeps_credentials`.
2. **Re-submitting the form after a setup error, without uploading again.** `process_uploaded_file` deletes the upload after it is read, so the parsed key has to be held on the flow and the upload field becomes optional on the re-render. Tested in Task 4 by `test_retry_after_connection_error_does_not_require_reupload`.
3. **A binary, non-UTF-8 or non-object JSON upload.** This must produce the `invalid_key_file` form error, not a traceback. Tested in Task 1 by `test_parse_rejects_non_json_bytes` and `test_parse_rejects_json_array`.
4. **A connection test that hangs**, for example on an unsupported region. It must time out (20 s) and show `cannot_connect` with a timeout message. Tested in Task 2 by `test_validate_connection_times_out`.
5. **A Google exception with an empty message.** The error shown must still say something, namely the exception class name. Tested in Task 2 by `test_error_text_falls_back_to_class_name`.

---

### Task 1: Worktree, constants, and service-account key parsing

**Files:**
- Create: `custom_components/gemini_live/vertex.py`
- Modify: `custom_components/gemini_live/const.py` (after the `PROVIDER_*` block, about line 37)
- Test: `tests/test_vertex.py`

**Interfaces:**
- Produces (in `const.py`):
  - Constants: `PROVIDER_VERTEX = "vertex"`, `CONF_VERTEX_CREDENTIALS = "vertex_credentials"` (the stored dict), `CONF_VERTEX_KEY_FILE = "vertex_key_file"` (the form upload id, never stored) and `CONF_VERTEX_LOCATION = "vertex_location"`.
  - Defaults and lists: `VERTEX_DEFAULT_LOCATION = "us"`, `VERTEX_LOCATIONS: list[str]`, `VERTEX_DEFAULT_MODEL = "gemini-3.8-live"` and `VERTEX_AVAILABLE_MODELS: list[str]`.
  - Helper: `is_google_provider(provider: str | None) -> bool`, which is True for `PROVIDER_GEMINI` and `PROVIDER_VERTEX`.
- Produces (in `vertex.py`): `class VertexKeyError(ValueError)` with attribute `reason: str`, where `reason` is one of `"invalid_key_file"`, `"not_service_account"` or `"missing_key_fields"`. These match the form error keys added in Task 4. Also `parse_service_account_key(raw: bytes | str) -> dict[str, Any]`.

- [ ] **Step 1: Create the worktree and confirm the baseline**

```bash
cd /Users/cscuderi/workspace/gemini_live/ha-gemini-live
git worktree add ../ha-gemini-live-vertex -b vertex-ai-provider 8a851db
cd ../ha-gemini-live-vertex
/Users/cscuderi/workspace/gemini_live/ha-gemini-live/.venv/bin/python -m pytest -q
```
Expected: `196 passed`.

- [ ] **Step 2: Write the failing tests** in `tests/test_vertex.py`

```python
_KEY = {"type": "service_account", "project_id": "proj", "private_key": "-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----\n",
        "client_email": "sa@proj.iam.gserviceaccount.com", "token_uri": "https://oauth2.googleapis.com/token", "client_id": "1"}

def test_parse_accepts_service_account_bytes():  # json.dumps(_KEY).encode() -> dict equal to _KEY
def test_parse_rejects_non_json_bytes():          # b"\xff\xfe garbage" -> VertexKeyError, reason == "invalid_key_file"
def test_parse_rejects_json_array():              # b"[1, 2]" -> reason == "invalid_key_file"
def test_parse_rejects_other_credential_type():   # {**_KEY, "type": "authorized_user"} -> reason == "not_service_account"
def test_parse_rejects_missing_fields():          # _KEY without "private_key" -> reason == "missing_key_fields"
def test_key_error_message_never_contains_private_key():  # for each failure above: _KEY["private_key"] not in str(err)
def test_is_google_provider():                    # gemini, vertex -> True; openai, qwen, personaplex, None -> False
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.../.venv/bin/python -m pytest -q tests/test_vertex.py`
Expected: an ImportError, because `gemini_live.vertex` doesn't exist yet.

- [ ] **Step 4: Implement the constants and `parse_service_account_key`**

Decode bytes as UTF-8. A `UnicodeDecodeError` or `json.JSONDecodeError`, or a parse result that isn't a `dict`, gives `invalid_key_file`. Return the whole parsed dict, since google-auth reads the optional fields too.

- [ ] **Step 5: Run the full suite**

Run: `.../.venv/bin/python -m pytest -q`
Expected: all pass (196 + 7).

- [ ] **Step 6: Commit**

```bash
git add custom_components/gemini_live/const.py custom_components/gemini_live/vertex.py tests/test_vertex.py
git commit -m "Add Vertex AI constants and service account key parsing"
```

---

### Task 2: Vertex client, connection test, and error text

**Files:**
- Modify: `custom_components/gemini_live/vertex.py`
- Test: `tests/test_vertex.py`

**Interfaces:**
- Consumes: `parse_service_account_key` from Task 1. Also `GeminiLiveClient` from `gemini.py`, which is unchanged.
- Produces:
  - `VERTEX_CONNECT_TIMEOUT = 20` (seconds).
  - `create_vertex_genai_client(credentials_info: dict[str, Any], location: str, http_options: Any = None) -> Any`. This is synchronous and must be called in an executor. It uses `service_account.Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/cloud-platform"])` and `genai.Client(vertexai=True, project=info["project_id"], location=location, credentials=..., http_options=http_options)`.
  - `async async_create_vertex_gemini_client(hass, credentials_info, location, affective_dialog: bool = False) -> GeminiLiveClient`. When `affective_dialog` is set it passes `types.HttpOptions(api_version="v1beta1")`, otherwise `None`.
  - `async async_validate_vertex_connection(hass, credentials_info, location, model) -> str | None`. It opens and closes `client.aio.live.connect(model=model, config={"response_modalities": ["AUDIO"]})` inside `asyncio.timeout(VERTEX_CONNECT_TIMEOUT)`. It returns `None` on success. On a timeout it returns `"Timed out after 20 seconds connecting to Vertex AI"`, and on any other exception it returns `vertex_error_text(exc)`.
  - `vertex_error_text(exc: BaseException) -> str`, which returns `str(exc).strip() or type(exc).__name__`.
  - `VERTEX_RUNTIME_ERROR_PREFIX = "Gemini Live (Vertex AI) request failed: "`.
  - `vertex_runtime_error_message(exc: BaseException) -> str | None`. It walks `__cause__`/`__context__` (like `_user_visible_api_error` in `stt.py`) and looks for a `google.genai.errors.APIError` whose `code != 1000`. If it finds one it returns `VERTEX_RUNTIME_ERROR_PREFIX + vertex_error_text(err) + ". Check the Google Cloud project, or use Reconfigure to update the service account key, location or model."`. Otherwise it returns `None`.

- [ ] **Step 1: Write the failing tests** in `tests/test_vertex.py`

Fake `google`, `google.genai`, `google.genai.types` and `google.oauth2.service_account` through `monkeypatch.setitem(sys.modules, ...)`, following `test_gemini_client_pins_v1beta` in `tests/test_gemini.py`. `_Hass.async_add_executor_job(func, *args)` returns `func(*args)`.

```python
async def test_vertex_client_uses_project_from_key_and_location():  # Client kwargs: vertexai=True, project="proj", location="us", credentials=<fake>, http_options=None
async def test_vertex_client_affective_dialog_uses_v1beta1():       # captured api_version == "v1beta1"
async def test_validate_connection_success_returns_none():          # fake live.connect async CM enters/exits -> None
async def test_validate_connection_passes_google_message_through(): # connect raises Exception("403 PERMISSION_DENIED: nope") -> "403 PERMISSION_DENIED: nope"
async def test_validate_connection_times_out(monkeypatch):          # VERTEX_CONNECT_TIMEOUT patched to 0.01, connect sleeps -> "Timed out after ..." 
def test_error_text_falls_back_to_class_name():                     # vertex_error_text(RuntimeError()) == "RuntimeError"
def test_runtime_message_wraps_api_error():                         # real google.genai.errors.APIError(403, {"error": {"message": "denied"}}) chained as __cause__ -> startswith(VERTEX_RUNTIME_ERROR_PREFIX), contains "Reconfigure", no "ai.dev"/"ai.studio"
def test_runtime_message_ignores_normal_close_and_other_errors():   # APIError code 1000 -> None; ValueError("x") -> None
```

The last two tests import the real `google.genai.errors`, so they must not install the fake module.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.../.venv/bin/python -m pytest -q tests/test_vertex.py`
Expected: the new tests fail with ImportError or AttributeError.

- [ ] **Step 3: Implement the functions listed in the Interfaces block in `vertex.py`**

- [ ] **Step 4: Run the full suite**

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git commit -am "Add Vertex AI client, connection check and error text"
```

---

### Task 3: Runtime entities and Repairs for Vertex

**Files:**
- Modify: `custom_components/gemini_live/stt.py`
  - The `_report_api_error` / `_clear_api_issues` block, about lines 256-305.
  - `LiveModelSTT`, about lines 483-515 and 1590-1640.
  - Add a `VertexGeminiSTT` class after `GeminiLiveSTT`.
  - The platform setup dispatch, about line 474.
- Modify: `custom_components/gemini_live/conversation.py`
  - The dispatch, about lines 80-86.
  - `LiveModelConversationAgent`, about lines 110-120 and 275-282.
  - The error branch, about line 456.
  - Add a `VertexGeminiConversationAgent` class after `GeminiLiveConversationAgent`.
- Modify: `custom_components/gemini_live/strings.json` and `translations/en.json`, adding the `issues.vertex_api_error` key.
- Test: `tests/test_vertex_runtime.py`

**Interfaces:**
- Consumes from Task 1: `PROVIDER_VERTEX`, `CONF_VERTEX_CREDENTIALS`, `CONF_VERTEX_LOCATION` and `VERTEX_DEFAULT_LOCATION`.
- Consumes from Task 2: `async_create_vertex_gemini_client`, `vertex_runtime_error_message` and `VERTEX_RUNTIME_ERROR_PREFIX`.
- Produces:
  - `LiveModelSTT._credentials(self, config: dict[str, Any]) -> Any`, which by default returns `config.get(CONF_API_KEY)`. `async_process_audio_stream` uses it in place of reading `CONF_API_KEY` directly, and passes the result through as the existing `api_key` argument.
  - `LiveModelConversationAgent._credentials`, the same method on the conversation side.
  - `LiveModelConversationAgent._api_error_message(exc) -> str | None`, a staticmethod that returns `_user_visible_api_error(exc)` by default. Line 456 calls `self._api_error_message(exc)`, which keeps the existing providers' behaviour identical.
  - `VertexGeminiSTT(GeminiLiveSTT)` and `VertexGeminiConversationAgent(GeminiLiveConversationAgent)`. Both have `integration_name = "Gemini Live (Vertex AI)"`. `_credentials` returns `config.get(CONF_VERTEX_CREDENTIALS)`. `_async_create_client(credentials)` calls `async_create_vertex_gemini_client(self.hass, credentials, location, affective_dialog=...)`, where the location is the merged entry config's `CONF_VERTEX_LOCATION` (default `VERTEX_DEFAULT_LOCATION`). `_api_error_message` returns `vertex_runtime_error_message(exc)`. The conversation agent's `error_response` is `"Sorry, I could not get a response from Gemini Live (Vertex AI)."`
  - Both dispatch dicts map `PROVIDER_VERTEX` to the new classes.
  - In `_report_api_error`, a new first branch for `user_message.startswith(VERTEX_RUNTIME_ERROR_PREFIX)` sets:
    - issue id: `f"vertex_api_error_{entry.entry_id}"`
    - `translation_key`: `"vertex_api_error"`
    - `learn_more_url`: `"https://console.cloud.google.com/vertex-ai"`
    - placeholders: `entry_title` and `error`, where `error` is the message with the prefix removed
  - `_clear_api_issues` also deletes `f"vertex_api_error_{entry.entry_id}"`.
  - Issue copy, written into both JSON files:
    - Title: `"Gemini Live (Vertex AI) request failed"`
    - Description: `"Vertex AI rejected a Gemini Live request for **{entry_title}**. Google's error: {error}\n\nCheck the Google Cloud project's Vertex AI API, billing and service account permissions, or use **Reconfigure** on the integration to update the service account key, location or model. This issue will clear automatically after Gemini Live connects successfully."`

- [ ] **Step 1: Write the failing tests** in `tests/test_vertex_runtime.py`

Build entries with `SimpleNamespace(entry_id="e1", title="Vertex", data={...}, options={})`. Patch `async_create_vertex_gemini_client` in the module under test and capture its arguments.

```python
def test_vertex_provider_dispatches_to_vertex_entities():          # stt + conversation async_setup_entry with provider "vertex" add VertexGeminiSTT / VertexGeminiConversationAgent
def test_vertex_credentials_come_from_stored_key():                # VertexGeminiSTT._credentials(config) is config[CONF_VERTEX_CREDENTIALS]; GeminiLiveSTT._credentials(config) is config[CONF_API_KEY]
async def test_vertex_stt_client_uses_entry_location():            # location "eu" in entry -> captured location == "eu", credentials dict passed through
async def test_vertex_client_defaults_location_to_us():            # no location in entry -> "us"
def test_vertex_runtime_errors_never_use_ai_studio_text():         # VertexGeminiSTT._api_error_message(APIError 429 "you exceeded your current quota") starts with VERTEX_RUNTIME_ERROR_PREFIX and contains none of "ai.dev", "ai.studio", "prepayment", "Google AI project"
def test_vertex_error_creates_vertex_repairs_issue(monkeypatch):   # _report_api_error with a Vertex message -> async_create_issue called with issue_id "vertex_api_error_e1", translation_key "vertex_api_error", placeholders {"entry_title": "Vertex", "error": <google text>}
def test_clear_api_issues_includes_vertex_issue(monkeypatch):      # deleted ids include "vertex_api_error_e1"
def test_gemini_conversation_error_behaviour_unchanged():          # GeminiLiveConversationAgent._api_error_message(quota error) == stt._QUOTA_USER_MESSAGE
def test_vertex_issue_strings_present_and_files_identical():       # strings.json == translations/en.json; issues.vertex_api_error has title + description with {entry_title} and {error}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.../.venv/bin/python -m pytest -q tests/test_vertex_runtime.py`
Expected: FAIL.

- [ ] **Step 3: Implement the interfaces above in `stt.py`, `conversation.py` and the two JSON files**

- [ ] **Step 4: Run the full suite**

Expected: all pass, including the untouched Gemini, OpenAI and Qwen error tests.

- [ ] **Step 5: Commit**

```bash
git commit -am "Run Vertex AI entries through the Gemini Live entities with Vertex errors"
```

---

### Task 4: Config flow (setup, reconfigure, options)

**Files:**
- Modify: `custom_components/gemini_live/config_flow.py`
  - `PROVIDER_SELECTOR`
  - `_model_selector`
  - `_provider_schema`, about lines 130-310
  - `async_step_provider`, `async_step_reconfigure` and `GeminiLiveOptionsFlowHandler.async_step_init`
- Modify: `custom_components/gemini_live/manifest.json`, adding `"dependencies": ["file_upload"]`
- Modify: `custom_components/gemini_live/strings.json` and `translations/en.json`, covering the `config.step.{user,provider,reconfigure}`, `options.step.init`, `config.error` and `options.error` keys
- Test: `tests/test_config_flow.py`

**Interfaces:**
- Consumes from Task 1: the constants, `is_google_provider` and `parse_service_account_key`/`VertexKeyError`.
- Consumes from Task 2: `async_validate_vertex_connection`.
- Produces:
  - Dropdown: `PROVIDER_SELECTOR` gains `SelectOptionDict(value=PROVIDER_VERTEX, label="Google Vertex AI")`, placed after Google Gemini.
  - `_model_selector(models: list[str], custom_value: bool = False)`.
  - `_provider_schema(PROVIDER_VERTEX, current, hass)` has the following fields. All other Gemini fields stay, with Gemini voices.
    - `CONF_VERTEX_KEY_FILE`: a `FileSelector(FileSelectorConfig(accept=".json,application/json"))`. It is `vol.Required` when `CONF_VERTEX_CREDENTIALS` is not in `current`, and `vol.Optional` otherwise. There is no `CONF_API_KEY` field.
    - `CONF_VERTEX_LOCATION`: required, with default `current.get(..., VERTEX_DEFAULT_LOCATION)`. It uses a dropdown `SelectSelector` over `VERTEX_LOCATIONS` with `custom_value=True`.
    - `CONF_MODEL`: default `current.get(CONF_MODEL, VERTEX_DEFAULT_MODEL)`. It uses `_model_selector(VERTEX_AVAILABLE_MODELS, custom_value=True)`.
  - Every `provider == PROVIDER_GEMINI` check in `_provider_schema` becomes `is_google_provider(provider)`.
  - `async _async_read_vertex_key(hass, user_input: dict, stored_credentials: dict | None) -> str | None` handles the upload:
    - If the upload id is present, it pops it, reads the file with `process_uploaded_file` inside one executor job, parses it, and sets `user_input[CONF_VERTEX_CREDENTIALS]`. A `VertexKeyError` returns its `reason`.
    - Without an upload, it sets `user_input[CONF_VERTEX_CREDENTIALS] = stored_credentials`.
    - It strips whitespace from the location.
    - It returns `None` on success.
  - `async _async_check_vertex_connection(hass, user_input: dict, current: dict | None) -> str | None` skips the check (returns `None`) when `current` is not `None` and credentials, location and model are all unchanged. Otherwise it returns `async_validate_vertex_connection(...)`.
  - The flow order for Vertex is the same in all three steps:
    1. Read the key. On an error, re-show the form with `errors={"base": reason}`.
    2. Run the existing `_needs_model_specific_refresh` check.
    3. Run `_strip_unsupported_settings`.
    4. Check the connection. On an error, re-show the form with `errors={"base": "cannot_connect"}` and `description_placeholders` including `{"error": text}`. The provider step also passes `"provider": "Google Vertex AI"`.
    5. Save.
    Each re-render passes `current` as `{**user_input, CONF_VERTEX_CREDENTIALS: <parsed or stored>}`, which makes the upload optional.
  - `GeminiLiveConfigFlow` keeps the parsed key in `self._vertex_credentials`, which starts as `None` and is set after a successful parse. That's the `stored_credentials` for a new entry.
  - The entry title for Vertex is `"Gemini Live (Vertex AI)"`, and the provider placeholder is `"Google Vertex AI"`.
- Strings (identical text in both files, and the same keys under `config.step.provider`, `config.step.reconfigure` and `options.step.init`):
  - Field labels (`data`):
    - `vertex_key_file`: "Service account key (JSON)"
    - `vertex_location`: "Location"
  - Field descriptions (`data_description`):
    - `vertex_key_file`: "Upload the JSON key of a Google Cloud service account with the Vertex AI User role. The project is read from the key. When editing, leave empty to keep the current key."
    - `vertex_location`: "Vertex AI location that serves the live model, for example us, eu, global or a region such as us-central1."
  - Errors, in both `config.error` and `options.error`:
    - `invalid_key_file`: "The uploaded file is not a valid JSON key file."
    - `not_service_account`: "The uploaded JSON is not a service account key. Create a key for a service account in the Google Cloud console."
    - `missing_key_fields`: "The service account key is missing required fields (client_email, private_key, token_uri, project_id)."
    - `cannot_connect`: "Could not connect to Vertex AI: {error}"
  - The `config.step.user.description` gains the words "Google Vertex AI".

- [ ] **Step 1: Write the failing tests** in `tests/test_config_flow.py`

Use the existing `_validator`/`_available_models` helpers. Flow-step tests build the flow with a fake `hass`, where `async_add_executor_job(func, *args)` returns `func(*args)`. `config_flow.process_uploaded_file` and `config_flow.async_validate_vertex_connection` are monkeypatched.

```python
def test_vertex_is_offered_as_a_provider():                      # "vertex" in PROVIDER_SELECTOR options with label "Google Vertex AI"
def test_vertex_schema_has_upload_location_and_no_api_key():     # keys include vertex_key_file (Required), vertex_location default "us"; no api_key; models == VERTEX_AVAILABLE_MODELS, custom_value True; search_grounding present
def test_vertex_upload_optional_when_key_stored():               # _provider_schema(PROVIDER_VERTEX, {CONF_VERTEX_CREDENTIALS: {...}}) marks vertex_key_file Optional
async def test_new_vertex_entry_stores_parsed_key():             # provider step with upload -> create_entry title "Gemini Live (Vertex AI)", data has vertex_credentials dict, vertex_location "us", model, provider "vertex", no vertex_key_file
async def test_bad_key_file_shows_form_error():                  # upload bytes "nope" -> form errors {"base": "invalid_key_file"}; connection check not called
async def test_connection_failure_passes_google_error_through(): # validate returns "403 PERMISSION_DENIED" -> errors {"base": "cannot_connect"}, placeholders["error"] == "403 PERMISSION_DENIED", placeholders["provider"] == "Google Vertex AI"
async def test_retry_after_connection_error_does_not_require_reupload():  # second submit without vertex_key_file succeeds using the key parsed on first submit
async def test_reconfigure_without_upload_keeps_credentials():   # entry has credentials; reconfigure submit without upload, same location/model -> data_updates keep vertex_credentials; validate NOT called
async def test_reconfigure_location_change_revalidates():        # location "us" -> "eu" -> validate called with "eu"
async def test_options_flow_new_upload_replaces_key():           # options init with upload -> saved options contain the new key dict
def test_gemini_schema_unchanged():                              # PROVIDER_GEMINI schema still has api_key and no vertex fields
def test_vertex_strings_present_and_files_identical():           # strings.json == en.json; config.error and options.error have the four keys; cannot_connect contains "{error}"
def test_manifest_depends_on_file_upload():                      # "file_upload" in manifest["dependencies"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.../.venv/bin/python -m pytest -q tests/test_config_flow.py`
Expected: the new tests fail.

- [ ] **Step 3: Implement the interfaces above**

- [ ] **Step 4: Run the full suite**

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git commit -am "Add Google Vertex AI provider to the config flow"
```

---

### Task 5: README and changelog on the feature branch, then push it

**Files:**
- Modify: `README.md`
  - Update the intro sentences that list providers (lines 1-10) to mention Vertex AI.
  - Add a new `## Google Vertex AI setup` section right before `## QwenCloud setup`.
- Modify: `CHANGELOG.md`, adding an entry under `## Unreleased`.

**Interfaces:** none.

- [ ] **Step 1: Write the README section**, following the shape of the QwenCloud section. It should cover:
  - **Why:** Vertex is billed to a Google Cloud project, postpay.
  - **Steps:**
    1. Enable the Vertex AI API.
    2. Create a service account with the **Vertex AI User** role (`roles/aiplatform.user`).
    3. Create a JSON key.
    4. In Home Assistant, choose **Google Vertex AI**, upload the key, and pick a location (default `us`) and a model (`gemini-3.8-live` default, or `gemini-live-2.5-flash-native-audio`, or type another from the [supported models table](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api)).
  - **Validation:** setup connects once at Submit, and Google's error is shown verbatim if that fails.
  - **Storage:** the key is stored in Home Assistant's config entry storage like other API keys.
  - **Changing settings:** use **Reconfigure** to change the key, location or model.
  - **Feature parity:** Search grounding, tools and voices work as with Google Gemini.
  - **Billing:** usage and Search grounding are billed to the Google Cloud project.

- [ ] **Step 2: Add the CHANGELOG entry**: "Added Google Vertex AI as a provider for Gemini Live: upload a service account JSON key during setup, choose a location and model, and bill usage to a Google Cloud project instead of AI Studio prepaid credits. Setup verifies the connection and shows Google's error verbatim; runtime errors and Repairs issues use Vertex-specific text."

- [ ] **Step 3: Run the full suite one last time**

Expected: all pass. Then `git status`, which should show a clean tree apart from these two files.

- [ ] **Step 4: Commit and push the feature branch**

```bash
git commit -am "Document Google Vertex AI setup"
git push -u origin vertex-ai-provider
```

---

### Task 6: Fork `main` — drop the `vertex:` overload, merge, and rewrite the fork note

**Files:**
- Modify (on `main` in `/Users/cscuderi/workspace/gemini_live/ha-gemini-live`): `custom_components/gemini_live/gemini.py`, via the revert, and `README.md`, for the fork note.

**Interfaces:** none.

- [ ] **Step 1: Revert the overload and the old fork note**

```bash
cd /Users/cscuderi/workspace/gemini_live/ha-gemini-live
git revert --no-edit 1e3a2bd da8f928
```
Expected: `gemini.py` and the README top match upstream `8a851db` (check with `git diff 8a851db -- custom_components README.md`, which should be empty).

- [ ] **Step 2: Merge the feature branch**

Run `git merge --no-ff vertex-ai-provider -m "Merge Google Vertex AI provider"`, then run the full suite. Expected: all pass.

- [ ] **Step 3: Add the new fork note** at the very top of `README.md`. It should cover:
  - This is a fork of matt123p/ha-gemini-live that adds a **Google Vertex AI** provider.
  - **Why:** AI Studio requires prepaid billing, while Vertex is postpay.
  - **How:** pick **Google Vertex AI** as the provider and upload a service account JSON key. It links to the "Google Vertex AI setup" section.
  - **Installing:** install it through HACS as a custom repository at `https://github.com/scudre/ha-gemini-live`.
  - **PR:** the change is offered upstream from branch `vertex-ai-provider`.

- [ ] **Step 4: Commit and push**

```bash
git commit -am "Update fork note for the Vertex AI provider"
git push origin main
```

- [ ] **Step 5: Manual end-to-end check (owner, in Home Assistant)**

Update the integration through HACS and restart. Then go to Add integration → Gemini Live → **Google Vertex AI**, upload the key, and leave location `us` and model `gemini-3.8-live`. Expected: the entry is created. A test in the Assist dialog gets a spoken reply. If Submit shows an error, Google's message is what to look up.

# Google Vertex AI provider — design

Date: 2026-10-29
Repo: fork `scudre/ha-gemini-live` of `matt123p/ha-gemini-live` (v1.0.11, upstream `8a851db`)

## Intent

Let Home Assistant users run Gemini Live through **Google Cloud Vertex AI** (postpay,
billed to a GCP project) instead of an AI Studio API key (which now requires prepaid
billing). It must be polished enough for other people to set up from the UI alone, and
clean enough to offer upstream to matt123p as a PR so the fork can eventually go away.

## Goals

- A new provider **"Google Vertex AI"** in the existing Provider dropdown, next to
  Google Gemini, OpenAI, QwenCloud and fal.ai PersonaPlex.
- Authentication by **uploading the service account JSON key in the setup form**. No
  file copied onto the HA host, no path typed anywhere. Works with HACS-only installs.
- A bad key file gets a specific error, and any connection failure shows an error **on
  the form at Submit** with Google's message, not on the first voice request.
- Everything that works for Google Gemini (voice selection, tools, Search grounding,
  transcription, sound responses, barge-in, system instruction, detailed logging) works
  identically for Vertex. Only setup and connection differ.
- Tests and a README section to upstream's standard.

## Non-goals

- Vertex API keys (express mode or service-account-bound "authorization keys").
  Investigated and rejected: express mode doesn't support Live models; bound keys require
  a Cloud organization (personal accounts like the owner's don't have one) and Live
  support is undocumented.
- Fetching the model list from Google. Vertex's model listing has no "supports Live"
  field, so a fetched list would be a name-based guess.
- Application Default Credentials / GCE metadata server / workload identity.
- Changing anything about the existing Google Gemini (AI Studio), OpenAI, Qwen or
  PersonaPlex providers.

## Anti-goals (would be a failure even if it works)

- Overloading the API key field. The existing `vertex:<project>:<location>:<path>`
  patch (commit `da8f928`) is **removed entirely**; nobody but the owner has used it, so
  no migration.
- Requiring shell access to the HA host.
- Hard-to-review sprawl that matt123p wouldn't merge: the change should read like the
  existing provider code and reuse the Gemini adapter rather than duplicate it.

## Decided

**Provider.** New constant (e.g. `PROVIDER_VERTEX = "vertex"`). Entry title
"Gemini Live (Vertex AI)" or similar, label "Google Vertex AI" in the dropdown.

**Setup form fields for Vertex** (replacing the API key field; all other Gemini fields
stay):

| Field | Behaviour |
|---|---|
| Service account key | HA file upload selector (`selector.FileSelector`, accept `.json`). Read via `homeassistant.components.file_upload.process_uploaded_file`. |
| Location | Dropdown that accepts custom values. **Default `us`.** Offer `us`, `eu`, `global`, and a few common regions (e.g. `us-central1`, `us-east4`, `europe-west4`, `asia-northeast1`). |
| Model | Dropdown that accepts custom values (`SelectSelectorConfig(custom_value=True)`). Options: **`gemini-3.8-live` (default)**, `gemini-live-2.5-flash-native-audio`. Source: Supported models table at https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api (the third listed model, `gemini-3.5-transcribe-live-preview`, is transcription-only and excluded). |

The project ID is **not** a field; it comes from the key's `project_id`.

**What is stored.** Parse the uploaded JSON, require `type == "service_account"` and the
fields `client_email`, `private_key`, `token_uri`, `project_id`. Store the parsed key
dict in the config entry data (same storage and protection as existing API keys in
`.storage/core.config_entries`). Don't write the file anywhere else. Never log key
contents.

**Validation at Submit.** Two layers:

1. *Local checks (ours, stable):* the upload is valid JSON, `type == "service_account"`,
   required fields present. Each gets its own form error.
2. *Live connection check:* build credentials with
   `service_account.Credentials.from_service_account_info(..., scopes=[cloud-platform])`,
   create `genai.Client(vertexai=True, project=..., location=..., credentials=...)`, and
   open then immediately close a Live session for the chosen model (blocking work in the
   executor, time-bounded). **Do not classify Google's errors.** Google's error codes and
   messages for Live are undocumented and may change, so any failure shows one form error
   ("Could not connect to Vertex AI") with **Google's error message passed through
   verbatim** (via a description placeholder) so the user can look it up. Owner's
   decision.

**Runtime.** `async_create_gemini_client` gains a Vertex path (credentials from the
stored dict, `vertexai=True`, project, location). Affective dialog uses
`api_version="v1beta1"` on Vertex (AI Studio keeps `v1beta`). The STT, TTS and
conversation entities for Vertex are the existing Gemini classes; they currently treat
any unknown provider as Gemini, but places that test `provider == PROVIDER_GEMINI` for
Gemini-only features (Search grounding, affective dialog, thinking level, form
placeholders) must also include Vertex — a small "is Google provider" helper is the
expected shape. Places that require `CONF_API_KEY` to be present
(`conversation.py`, `stt.py`) must accept a Vertex entry without one.

**Runtime errors.** Upstream's runtime error handling in `stt.py` / `conversation.py`
is written for AI Studio (matched strings and Repairs links such as `ai.dev/rate-limit`,
`ai.studio/projects`, "prepayment credits are depleted", "your Google AI project"). For
Vertex entries, none of that AI Studio wording or those links may appear. Instead show
Vertex-specific text: name Vertex AI / the GCP project, pass Google's error message
through verbatim, and point to **Reconfigure** to fix the key, location or model. Same
principle as setup: don't try to classify Google's errors. AI Studio behaviour is
unchanged. Owner's decision.

**No reauth flow.** No `async_step_reauth` / re-upload prompt. Users fix a revoked or
replaced key through the existing Reconfigure path. Owner's decision.

**Reconfigure and Options.** Upstream re-renders the full provider schema in both the
Reconfigure step and the Options flow. For Vertex there, the key upload is
**optional**: leave it empty to keep the stored key, upload a new one to replace it.
Location and model are editable. Re-validate on Submit when key, location or model
changed.

**Docs.** README: a "Vertex AI setup" section modelled on the existing "QwenCloud setup"
section — create a service account, grant **Vertex AI User**
(`roles/aiplatform.user`), enable the Vertex AI API, create and download a JSON key,
upload it in HA; model and location notes; billing note. Rewrite the fork note at the top
of the README to describe this provider instead of the `vertex:` key. `strings.json` and
`translations/en.json` get labels, descriptions and error texts for the new fields.

**Tests.** Alongside `tests/test_config_flow.py` and `tests/test_gemini.py`: provider
appears; valid upload creates an entry with parsed key, project, location, model;
each local check produces its form error; a failed connection shows the generic error
with Google's message passed through; reconfigure with and without a new upload; client
construction for Vertex vs AI Studio (including `v1beta1`/`v1beta`); Vertex runtime
errors use Vertex wording and never AI Studio text or links. All tests mock Google; no
real Vertex calls in the test suite.

## Left to the builder

- Exact constant/config key names, entry title wording, the precise region list.
- How to structure the Vertex branch in `_provider_schema` and the client factory, as
  long as it follows upstream's existing style.
- Whether affective dialog / thinking level apply to the Vertex model IDs (upstream's
  lists are keyed on AI Studio model names; check Vertex docs, default to hidden if
  unsure).
- How to keep the setup-time Live connection test cheap and time-bounded.
- Whether this design doc stays in the PR to upstream or only lives on the fork.

**Branching (Claude's call).** Build the feature on a branch cut from upstream `8a851db`
(e.g. `vertex-ai-provider`) so it can go to matt123p as a clean PR with no fork-only
changes. The fork's `main` then gets: a revert of the `vertex:` patch and the old fork
note, a merge of the feature branch, and the rewritten fork note. Fork-only commits never
go on the feature branch.

## Context

- HA runs in Docker; the integration is installed via HACS custom repository
  `https://github.com/scudre/ha-gemini-live`.
- Owner's service account already has `roles/aiplatform.user` and works for Vertex via
  LiteLLM.
- SDK: `google-genai>=2.25.0` (verified against 2.34.0).

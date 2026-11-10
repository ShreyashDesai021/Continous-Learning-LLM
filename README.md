# EvoLLM Enterprise — 3.0

An upload-first enterprise knowledge prototype with **Baseline RAG**, **CAKE**, **CAKE + AMO**, and **Compare all**. This release corrects the earlier evidence-selection and cache-path issues and replaces the interface with a focused workspace design.

## Start on Windows

1. Extract the entire ZIP to a **new folder**. Python 3.11 or 3.12 is recommended.
2. Run **setup.bat**. It creates the virtual environment, installs core dependencies, and runs the offline tests.
3. Run **start.bat**, keep the terminal open, and visit **http://127.0.0.1:8000**.
4. Sign in with **admin** / **EvoLLM-Admin-2026!**.
5. Add your company documents under **Documents**, then ask a question under **Chat**.

A new workspace is empty. No company policies or sample answers are installed automatically. The original research documents can be imported explicitly under **Compare & evaluate → Load optional research demo** in an empty workspace.

An employee account is available as **employee** / **EvoLLM-Employee-2026!**. Administrators can create accounts and arbitrary department names under **Settings → Manage team accounts**. Document read access can be `all`, `admin`, or a department name. Each account can change its password in Settings.

To upgrade an existing installation, stop its server and back up `runtime/` first. Copy the old `runtime/` into the new project folder to preserve accounts, uploads and conversations. Schema migrations run automatically. Old source records default to Policy; use **View source → Edit source settings** to mark old FAQs/clarifications as Reference where appropriate. For a clean workspace, leave the new `runtime/` absent.

## Enable actual neural RAG

The default **Source excerpts** engine works offline after dependencies are installed. It is an explicit diagnostic/fallback mode, not a mock language model. It quotes retrieved sources with citations. Its timings do not establish LLM performance.

For **BGE-M3 + Qdrant + Qwen2.5-1.5B-Instruct**, run **setup-ml.bat** after setup.bat. This installs neural dependencies, downloads the pretrained model files, and enables the neural configuration in `.env`. Restart with start.bat. Model downloads require several GB of storage; CPU inference needs sufficient RAM and can be slow. Weights are not bundled in the ZIP.

**Settings → Check model setup** reports whether local configuration files are found. A successful generated answer is still required to verify weights and inference on your machine. Missing model files produce an error; the application never silently substitutes fake LLM results.

Manual commands (Windows PowerShell; no activation-policy changes needed):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run.py
```

Optional neural setup:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe scripts/configure_ml.py
```

On macOS/Linux, use `python3 -m venv .venv`, `.venv/bin/python` in place of the Windows Python path, and `bash start.sh`. The optional scripts are otherwise the same. If port 8000 is occupied, use `python run.py --port 8001` with your environment's Python.

## Workspace experience

- **Chat:** compact pipeline selector, follow-up questions, saved conversations, cited answers, source details and measured side-by-side comparisons. Qwen responses stream as provisional text; the final answer is checked for valid citation IDs before it is saved. Source excerpts complete without simulated token generation.
- **Documents:** search, original file downloads, source text, version diffs, approval/reference settings, new revisions and access restrictions.
- **Compare & evaluate:** same-question comparisons plus labelled dataset imports. Timing repeats report median and P95. No benchmark outcome figures are hardcoded.
- **Memory / Activity:** bounded context reuse, logical archival/consolidation, usage observations and administrator audit records.
- **Settings:** generator selection, model status, cache controls and team accounts.

The interface uses bundled SVG illustrations/icons, a restrained slate/indigo theme, responsive layouts, keyboard submission, loading/error states, and reduced-motion support. It needs no Node build, CDN or external font service.

## Document metadata

Upload PDF, DOCX, TXT, Markdown or CSV. Scanned PDFs need external OCR. Text files must be UTF-8. Maximum file size is 12 MiB.

Choose a collection name relevant to your company. Use **Documents → New version** to prefill an existing source’s metadata. For revisions of one source, reuse its **document key** and preserve section headings or `[C1]` clause labels. Use a different key for an independent source. Effective-to is exclusive. Upload time alone never establishes supersession.

Select **Reference or clarification** for FAQs and explanatory articles. Such sources can explain an answer but cannot independently override an approved policy. Drafts and withdrawn records cannot govern. Unverified approval triggers review. For independent documents about the same rule, an explicit rule key or clause-to-rule mapping can establish alignment; otherwise the system uses conservative lexical alignment. This is not general semantic contradiction detection.

Read access and applicability are separate: an employee can be allowed to read a policy that applies to a different location. Scope is matched against query/context; department access is enforced before retrieval and cache reuse.

Authority ranks are administrator-entered metadata. The system cannot infer reliable organizational authority from arbitrary prose.

## What changed from 2.x

- Removed runtime branches for particular policy families or demo document IDs. Query planning and resolution operate on uploaded metadata and text.
- Separated governing, historical, announced and supporting evidence. A question about a proposal/FAQ can retain that source to explain why it does not override the rule.
- Comparisons preserve both requested locations/environments rather than filtering one away.
- Related evidence is ranked on a shared corpus score basis, avoiding comparisons of independently computed BM25 scores on different-sized groups.
- A valid AMO hit checks the revision **before** loading the corpus, embedding or searching. Usage writes are buffered and flushed on memory inspection/shutdown.
- CAKE and CAKE + AMO share the same resolution path. Caching must preserve selected evidence, not change the answer to appear better.
- Dense Qdrant indexes are reused in-process instead of rebuilt on every question.
- Research data is optional; custom company evaluation datasets can be imported.
- The interface no longer displays extractive phrase matches as answer accuracy.

## Evaluate your own company documents

See **docs/EVALUATION_FORMAT.md** for JSON imports. Labels remain separate from the inference pipeline and are never supplied to the model. Use source IDs from the source viewer/API, not document titles alone. Your dataset should include current rules, historical questions, comparisons, explanations, future changes, unsupported questions and relevant update events.

Metrics distinguish **raw retrieval recall**, **selected evidence recall/precision**, **full acceptable evidence-set coverage**, and **generated claim-phrase matches**. Claim checks are literal strings, not a semantic correctness judge. They are absent for Source excerpts. A question without labels shows unmeasured values rather than an invented score.

Cold means empty AMO query/context caches. Warm runs explicitly prime the same question set. Model/index initialization may affect early runs; all reported timings are engine latency, excluding network/rendering and chat-history persistence. Read per-question exports and review answers manually before making research claims. The original 25 questions are development data, not a held-out evaluation.

## Verification

```text
python -m pytest -q
python -m pytest ml_tests -q
python scripts/verify_release.py
python scripts/browser_smoke.py
```

The first command is the offline application suite. ML tests require requirements-ml.txt and use tiny local fixtures rather than pretrained weights. Browser tests require Playwright and Chromium separately. Verification details and executed test reports are in **docs/VERIFICATION.md**.

The original research implementation remains under `src/`, `tests/`, `data/` and `results/`. It is retained for traceability and is not the new application's live answering engine. Old result files include mock-generation experiments and are not the new release's results.

## Configuration and local state

Copy `.env.example` to `.env` before launch if needed. Set `EVOLLM_WORKSPACE_NAME` to your company name, `EVOLLM_RETRIEVAL` to `lexical` or `bge`, and `EVOLLM_GENERATOR` to `evidence` or `qwen`. Existing locally downloaded model directories can be assigned through `EVOLLM_EMBEDDING_MODEL` and `EVOLLM_LLM_MODEL`. The download script reads `.env` and accepts `--only embedding` or `--only llm`.

Default credentials are documented for the local prototype. Set custom password variables before accounts are first created; existing accounts are not silently overwritten. To reset, stop the server and rename `runtime/` as a backup before restarting. That folder contains your database, source files and private state.

## Scope and honest limits

This is a local enterprise prototype, not a certified production deployment. It has local authentication, tenant checks, department access, version history and audit records. It does not implement SSO, malware scanning, OCR, a distributed job/index service, or a full human conflict-adjudication workflow. Explainability is relational lineage and decisions, not a Neo4j graph. No LangGraph integration is claimed.

Approval, lineage and rule alignment need reliable metadata. Conservative lexical conflict alignment can require review of semantically equivalent paraphrases or miss differently worded contradictions. Historical bare years use a year-end target; “throughout” requests retain interval evidence, but complex date ranges need explicit dates and review. CSV parsing preserves row fields; it is not a relational query engine.

Memory utility scores are advisory; BM25 or fused lexical/dense ranking remains the retrieval ranking. Context caches are bounded and process-local. Their revision invalidation is intentionally conservative across a tenant. Usage observations persist when flushed. The server binds to localhost; do not expose it directly to the internet.

Pretrained BGE-M3/Qwen inference was **not verified in this build environment**. Their adapters are implemented and exercised with local fixtures, including real Transformers generation/streaming. Run the actual model configuration and evaluation on your machine before reporting neural accuracy or latency. No universal superiority over enterprise chatbot products is claimed.

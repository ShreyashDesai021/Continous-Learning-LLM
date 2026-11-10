# Version 3.0 verification — 28 October 2026

## Executed checks

| Check | Result |
|---|---|
| Offline application suite | **69 passed** |
| Optional local-fixture ML adapters | **2 passed** |
| Original research tests not requiring pretrained BGE downloads | **32 passed** |
| Combined suite | **103 passed; 3 original weight-dependent tests deselected** |
| Browser workflow | **13 checks passed; no captured browser JavaScript errors** |
| Python compilation / frontend JavaScript syntax | Passed |
| Release manifest and clean archive verifier | Checked during packaging |

Actual test reports are included as `test-results.xml` and `test-results-offline.xml`. The original concurrency-test scheduling assumption was corrected in 2.0.1 and remains covered without assuming a particular thread completion order.

The combined command is:

```bash
python -m pytest enterprise_tests ml_tests tests -q \
  -k 'not remediation_1_deferred_snapshot_indexing and not remediation_4_paraphrase_cache_hit and not remediation_5_ood_abstention' \
  --junitxml=docs/test-results.xml
```

## Behaviour verified

- Empty default workspace and explicit optional research import.
- Arbitrary uploaded collection/rule names outside the research corpus.
- Current versus historical policy selection and pending announcements.
- Explanatory retention of drafts/references without granting them governing status.
- Cross-location/environment comparison without suppressing one requested scope.
- Independent source rule alignment; repeated clause labels alone do not establish a conflict.
- Previously diagnosed FAQ, meal comparison, knowledge-base clarification, unapproved proposal and deployment comparison cases retain the required source evidence.
- Valid AMO hits bypass corpus loading, embedding and search; cached evidence matches CAKE's selected evidence.
- Upload, deletion and metadata revision prevent stale context reuse; permissions and tenant boundaries remain enforced.
- Conversation ownership, referential follow-up scope and legacy history migration.
- User-supplied evaluations, unmeasured metrics without labels, cold/warm evidence consistency and measured repeated-run timing.
- Account creation for arbitrary departments, own-password changes, and administrator-only source edits.

## Browser checks

Sign-in; empty-workspace onboarding; three-way comparison; citations and version history; upload and library refresh; querying a new upload; custom-department account creation; approval metadata editing; memory; 25-question development evaluation; saved conversation history; activity log; and mobile layout without horizontal page overflow.

The browser checks used an isolated temporary database and Chromium 153. No user's real account or company documents were used. The interface was visually inspected at desktop and mobile sizes, including motion-disabled screenshots for layout review. Playwright is optional for development checks, not an application dependency.

## Neural adapter tests and limits

The embedding fixture exercises real SentenceTransformer calls and real Qdrant ranking/fusion using a tiny locally built embedding fixture. It is not pretrained BGE-M3 inference.

The generation fixture saves/reloads a tiny random Qwen2 model and local tokenizer, runs the actual Transformers generation and streaming path, and verifies refusal of invalid source citations. It is not a Qwen2.5-1.5B-Instruct quality benchmark.

**Pretrained BGE-M3/Qwen2.5-1.5B inference and throughput were not verified in this build environment.** setup-ml.bat and the download/configuration scripts are provided. Weights are not bundled. The three deselected original tests preload those BGE weights; they are not reported as passing here. The earlier user screenshot separately showed the original 35-test suite passing in the user's own Windows environment.

Windows batch launchers were inspected and are supplied with direct virtual-environment Python commands. They were not executed on a Windows host here. The offline core suite was executed on Python 3.12/Linux; rerun setup.bat locally to verify your installation.

## Measured development evaluation

`development-evaluation-excerpts.json` contains an actual three-repeat run on the optional 25-question development corpus in Source excerpts mode. It is a saved report, never a source of UI scores. Claim-accuracy fields are unmeasured because no LLM generated the excerpts. The report includes per-question sources, measured samples, raw/selected evidence metrics, and cold/warm consistency.

Results are not guarantees for another corpus or hardware. There is no assertion that AMO cold requests must be faster: routing, validation and cache population have costs. The AMO claim to test is reduced repeated retrieval work while preserving valid evidence. Warm timings exclude explicit priming. First model/index initialization can affect cold results.

## Scope

A local prototype has been implemented and tested. Tests establish specified behaviours, not unrestricted semantic correctness or production readiness. See README for limitations in lexical rule alignment, metadata requirements, complex temporal reasoning, access-management scope and model setup. No enterprise-product superiority, production certification, Neo4j graph or LangGraph orchestration is claimed.

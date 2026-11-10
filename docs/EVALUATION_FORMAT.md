# Import a company evaluation

In **Compare & evaluate**, choose **Import labelled questions** and upload a JSON file in this shape:

```json
{
  "name": "Company policy evaluation",
  "questions": [
    {
      "id": "question-1",
      "question": "What is the current storage allowance?",
      "as_of": "2026-10-28",
      "context": {"locations": ["North office"]},
      "acceptable_evidence_sets": [["YOUR_DOCUMENT_ID#C1"]],
      "required_claims": ["140 litres"],
      "forbidden_claims": ["200 litres"]
    }
  ]
}
```

Replace source IDs with actual IDs from the source viewer's text or `GET /api/documents/{id}/clauses`. Section-based sources may have IDs such as `document-id#S1`. Alternative complete evidence sets can be listed as separate arrays. Do not use an empty evidence set: omit labels instead when a metric is unmeasured.

There can be 1–100 questions. Context can include `locations`, `conditions` and `employee_category`. Use document IDs and snapshots belonging to your workspace. An upload made today is not available in a repository snapshot from last year; to ask about an earlier rule using documents available now, keep the snapshot current and include the historical effective date in the question.

Claim phrase labels are evaluated only for Qwen-generated answers. Literal phrase checks do not prove factual correctness and can reject equivalent wording. Raw and selected evidence metrics use source IDs and are separate from claims. Missing labels show an unmeasured value.

All three pipelines share the same generator settings and permitted corpus. Warm AMO uses valid repeated contexts; the export includes selected evidence IDs and a comparison against the cold pass. No expected answer or evidence label is sent into retrieval or generation.

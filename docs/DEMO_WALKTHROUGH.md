# Demonstrate the prototype

## Use your own company documents

Start with the empty workspace. Upload a real policy, provide its collection, document key, version and effective date, then ask a question. Choose Source excerpts for an offline evidence check or enable Local Qwen for a generated answer. Use Compare all to inspect the same question in each pipeline.

## Version update and cache invalidation

1. Upload `demo_uploads/travel_v1.md` as collection `travel`, document key `finance_travel`, version `1`, effective from `2026-01-01`, approved, authority 80.
2. Keep the knowledge snapshot at today. Ask `What is the travel allowance per day?` in CAKE + AMO. Repeat it and inspect **Run details → Context: Reused**.
3. Upload `travel_v2.md` with the same key, collection and authority, version `2`, effective from today.
4. Ask again. The revised INR 1200 source should govern and the cache should miss because the knowledge revision changed.
5. Ask `What was the travel allowance per day on 2026-02-01?` while keeping the knowledge snapshot at today. The INR 900 source should apply historically.
6. Open a citation to see original text and version changes.

## Explanation and scope

Upload a draft proposal for a different allowance. Ask the current rule: the proposal must not govern. Ask whether the proposal changed the rule: it can be retained as **supporting** evidence with a reason, without becoming an approved policy.

Upload two same-family rules with different locations/environments. Ask for a comparison: both requested scopes should remain available. Upload independent equal-authority contradictory statements with matching explicit rule keys: the system should flag review rather than invent a winner.

## Optional original research dataset

In an empty workspace, choose **Compare & evaluate → Load optional research demo**. It imports the original controlled files, not model answers. Use the labelled 25-question development set to inspect evidence and timing; it is not a held-out study.

**Settings → Manage team accounts** lets an administrator create departments beyond the initial accounts. Upload department-restricted documents and verify that another department cannot retrieve or view them.

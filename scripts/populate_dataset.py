"""
Populate development dataset for Enterprise CLLM Conventional RAG Baseline.
Creates all 20 development documents, metadata, questions, and ground-truth answers.
"""

import json
from pathlib import Path

BASE_DIR = Path("data")

# 1. Directory Structure
DOCS_DIR = BASE_DIR / "inference" / "development" / "documents"
META_DIR = BASE_DIR / "inference" / "development"
EVAL_DIR = BASE_DIR / "evaluation" / "development"
GT_DIR = BASE_DIR / "ground_truth" / "development"
RUNNER_DIR = BASE_DIR / "runner"

for d in [DOCS_DIR, META_DIR, EVAL_DIR, GT_DIR, RUNNER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 2. Documents
DOCUMENTS = {
    # Family 1: Remote Work
    "D101": """# Remote Working Policy
Document ID: D101
Version: 1.0
Department: HR
Publication Date: 2024-12-15
Effective Date: 2025-01-01
Authority: HR Director (approved corporate policy)
Scope: locations=['Pune', 'Mumbai'], employee_category='permanent'

[C1] Permanent employees may work remotely for a maximum of two days per week.
[C2] This policy applies to the Pune and Mumbai offices from 1 January 2025.
[C3] Approved by the HR Director.
""",
    "D102": """# Remote Working Policy
Document ID: D102
Version: 2.0
Department: HR
Publication Date: 2026-01-15
Effective Date: 2026-02-01
Authority: HR Director (approved corporate policy)
Scope: locations=['Pune', 'Mumbai'], employee_category='permanent'

[C1] From 1 February 2026, permanent employees may work remotely for a maximum of three days per week.
[C2] This policy applies to the Pune and Mumbai offices and supersedes Remote Working Policy version 1.0 from that date.
[C3] Approved by the HR Director.
""",
    "D103": """# Remote Working Policy
Document ID: D103
Version: 3.0
Department: HR
Publication Date: 2026-06-15
Effective Date: 2026-07-01
Authority: HR Director (approved corporate policy)
Scope: locations=['Pune', 'Mumbai'], employee_category='permanent'

[C1] From 1 July 2026, permanent employees may work remotely for a maximum of four days per week.
[C2] This policy applies to the Pune and Mumbai offices and supersedes Remote Working Policy version 2.0 from that date.
[C3] Approved by the HR Director.
""",
    "D104": """# Remote Working Policy FAQ
Document ID: D104
Version: FAQ 2026-06
Department: HR
Publication Date: 2026-06-18
Effective Date: 2026-07-01
Authority: HR Operations (approved guide)
Scope: locations=['Pune', 'Mumbai'], employee_category='permanent'

[C1] Permanent staff in Pune and Mumbai can use up to four remote-working days each week from 1 July 2026.
[C2] This FAQ explains the Remote Working Policy version 3.0 provisions.
[C3] Approved by HR Operations.
""",

    # Family 2: Office Meals
    "D201": """# Pune Late-Working Meal Allowance
Document ID: D201
Version: Pune 1.0
Department: Finance
Publication Date: 2026-01-02
Effective Date: 2026-01-05
Authority: Finance Controller (approved office supplement)
Scope: locations=['Pune'], employee_category='all', conditions=['work after 21:00', 'one meal per eligible working day']

[C1] Pune employees working after 21:00 may claim up to INR 250 for one meal per eligible working day.
[C2] This Finance-approved office supplement takes effect on 5 January 2026.
""",
    "D202": """# Mumbai Late-Working Meal Allowance
Document ID: D202
Version: Mumbai 1.0
Department: Finance
Publication Date: 2026-01-02
Effective Date: 2026-01-05
Authority: Finance Controller (approved office supplement)
Scope: locations=['Mumbai'], employee_category='all', conditions=['work after 21:00', 'one meal per eligible working day']

[C1] Mumbai employees working after 21:00 may claim up to INR 350 for one meal per eligible working day.
[C2] This Finance-approved office supplement takes effect on 5 January 2026.
""",
    "D203": """# Finance Help Article: Pune Evening Meals
Document ID: D203
Version: KB edition 1
Department: Finance
Publication Date: 2026-02-02
Effective Date: 2026-01-05
Authority: Finance Controller (approved knowledge-base restatement)
Scope: locations=['Pune'], employee_category='all', conditions=['work after 21:00', 'one meal per eligible working day']

[C1] For Pune staff who work beyond 9 p.m., the reimbursement ceiling is INR 250 for a single meal on that working day.
[C2] This article restates the Pune office supplement effective 5 January 2026 and introduces no policy change.
""",
    "D204": """# Mumbai Late-Working Meal Allowance
Document ID: D204
Version: Mumbai 2.0
Department: Finance
Publication Date: 2026-06-01
Effective Date: 2026-06-15
Authority: Finance Controller (approved office supplement)
Scope: locations=['Mumbai'], employee_category='all', conditions=['work after 21:00', 'one meal per eligible working day']

[C1] From 15 June 2026, the meal reimbursement ceiling for Mumbai employees working after 21:00 is INR 400 per eligible working day.
[C2] This approved supplement replaces Mumbai version 1.0. Pune office provisions remain unchanged.
""",

    # Family 3: Support Callback
    "D301": """# Standard Support Callback Guide
Document ID: D301
Version: null
Department: Support
Publication Date: null
Effective Date: null
Authority: Imported support wiki; approver unknown
Scope: locations=['Pune', 'Mumbai'], employee_category='support staff', conditions=['standard-plan incidents', 'received during business hours']

[C1] For standard-plan customer incidents received during business hours, the support team must call the customer within four business hours.
""",
    "D302": """# Standard Support Onboarding Note
Document ID: D302
Version: null
Department: Support
Publication Date: null
Effective Date: null
Authority: Imported onboarding note; approver unknown
Scope: locations=['Pune', 'Mumbai'], employee_category='support staff', conditions=['standard-plan incidents', 'received during business hours']

[C1] For standard-plan customer incidents received during business hours, the support team must call the customer within eight business hours.
""",
    "D303": """# Support Recordkeeping Notice
Document ID: D303
Version: 1.0
Department: Support
Publication Date: 2026-04-03
Effective Date: 2026-04-03
Authority: Approved Support Operations notice
Scope: locations=['Pune', 'Mumbai'], employee_category='support staff', conditions=['standard-plan incidents', 'received during business hours']

[C1] Every callback for a standard-plan incident received during business hours must be recorded in the service-desk ticket.
[C2] This notice establishes recordkeeping requirements only and does not define a callback deadline.
[C3] Approved by Support Operations.
""",
    "D304": """# Support Recordkeeping Accessible Guide
Document ID: D304
Version: 1.0 accessible
Department: Support
Publication Date: 2026-04-05
Effective Date: 2026-04-03
Authority: Approved Support Operations notice
Scope: locations=['Pune', 'Mumbai'], employee_category='support staff', conditions=['standard-plan incidents', 'received during business hours']

[C1] For each standard-plan incident received in business hours, enter the callback in its service-desk ticket.
[C2] This accessible guide restates the callback recordkeeping requirement without altering operational policy.
[C3] Approved by Support Operations.
""",

    # Family 4: Backup Retention
    "D401": """# Production Backup Standard
Document ID: D401
Version: 1.0
Department: IT
Publication Date: 2026-01-05
Effective Date: 2026-01-10
Authority: IT Director; approved standard
Scope: locations=['Pune', 'Mumbai'], employee_category='IT staff', conditions=['production application backups']

[C1] Production application backups must be retained for 30 days.
[C2] This standard is effective from 10 January 2026. Only an IT Director-approved revision may change this retention requirement.
""",
    "D402": """# Infrastructure Team Discussion
Document ID: D402
Version: null
Department: IT
Publication Date: 2026-03-05
Effective Date: null
Authority: Team discussion export; unapproved suggestion
Scope: locations=['Pune', 'Mumbai'], employee_category='IT staff', conditions=['production application backups']

[C1] We should retain production application backups for only seven days to reduce storage consumption.
[C2] This is a team suggestion and has not been approved as a change to the backup standard.
""",
    "D403": """# Production Backup Standard, Version 2.0
Document ID: D403
Version: 2.0
Department: IT
Publication Date: 2026-04-15
Effective Date: 2026-05-01
Authority: IT Director; approved standard revision
Scope: locations=['Pune', 'Mumbai'], employee_category='IT staff', conditions=['production application backups']

[C1] Production application backups must be retained for 45 days from 1 May 2026.
[C2] This is the IT Director-approved version 2.0 of the Production Backup Standard.
""",
    "D404": """# Production Backup Standard Reminder
Document ID: D404
Version: 2.0 reminder
Department: IT
Publication Date: 2026-04-18
Effective Date: 2026-05-01
Authority: IT Director; approved standard revision
Scope: locations=['Pune', 'Mumbai'], employee_category='IT staff', conditions=['production application backups']

[C1] Keep production application backups for a period of 45 days, as required from 1 May 2026.
[C2] Operational reminder for the approved backup standard version 2.0.
""",

    # Family 5: Deployment
    "D501": """# Production Deployment SOP
Document ID: D501
Version: Production 1.0
Department: Engineering
Publication Date: 2026-01-08
Effective Date: 2026-01-15
Authority: Release Manager-approved SOP
Scope: locations=['Pune', 'Mumbai'], employee_category='engineering staff', conditions=['production deployments']

[C1] Production deployments require two reviewers.
[C2] Production deployment logs must be retained for 90 days.
[C3] Production deployment health checks must run for 15 minutes.
[C4] This SOP is effective from 15 January 2026.
""",
    "D502": """# Development Deployment SOP
Document ID: D502
Version: Development 1.0
Department: Engineering
Publication Date: 2026-02-01
Effective Date: 2026-02-05
Authority: Release Manager-approved SOP
Scope: locations=['Pune', 'Mumbai'], employee_category='engineering staff', conditions=['development-environment deployments']

[C1] Development-environment deployments require one reviewer.
[C2] Development-environment deployment logs must be retained for 14 days.
[C3] This SOP applies only to development environments from 5 February 2026.
""",
    "D503": """# Production Deployment SOP, Version 1.1
Document ID: D503
Version: Production 1.1
Department: Engineering
Publication Date: 2026-03-20
Effective Date: 2026-04-01
Authority: Release Manager-approved SOP
Scope: locations=['Pune', 'Mumbai'], employee_category='engineering staff', conditions=['production deployments']

[C1] Production deployments require two reviewers.
[C2] Production deployment logs must be retained for 90 days.
[C3] Production deployment health checks must run for 30 minutes.
[C4] From 1 April 2026, this revision replaces the health-check duration in version 1.0. Approval and log-retention requirements remain unchanged.
""",
    "D504": """# Production Deployment SOP, Accessible Reissue
Document ID: D504
Version: Production 1.1 reissue
Department: Engineering
Publication Date: 2026-04-10
Effective Date: 2026-04-01
Authority: Release Manager-approved SOP
Scope: locations=['Pune', 'Mumbai'], employee_category='engineering staff', conditions=['production deployments']

[C1] Obtain approvals from two reviewers for a production deployment.
[C2] Keep production deployment logs for 90 days.
[C3] Monitor production deployment health for 30 minutes.
[C4] This accessible-text reissue restates version 1.1, effective 1 April 2026, without changing its requirements.
"""
}

for doc_id, text in DOCUMENTS.items():
    (DOCS_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")

# 3. Metadata
METADATA = [
    {"document_id": "D101", "policy_family": "remote_work", "version": "1.0", "publication_date": "2024-12-15", "ingested_at": "2024-12-16T00:00:00Z", "effective_from": "2025-01-01", "effective_to": None, "department": "HR", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "permanent", "conditions": []}, "source_id": "HR_01", "authority": {"issuer": "HR Director", "source_class": "corporate_policy", "approval_status": "approved"}, "event_type": "INITIAL_IMPORT", "title": "Remote Working Policy v1.0"},
    {"document_id": "D102", "policy_family": "remote_work", "version": "2.0", "publication_date": "2026-01-15", "ingested_at": "2026-01-16T00:00:00Z", "effective_from": "2026-02-01", "effective_to": None, "department": "HR", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "permanent", "conditions": []}, "source_id": "HR_02", "authority": {"issuer": "HR Director", "source_class": "corporate_policy", "approval_status": "approved"}, "event_type": "ADDED", "title": "Remote Working Policy v2.0"},
    {"document_id": "D103", "policy_family": "remote_work", "version": "3.0", "publication_date": "2026-06-15", "ingested_at": "2026-06-16T00:00:00Z", "effective_from": "2026-07-01", "effective_to": None, "department": "HR", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "permanent", "conditions": []}, "source_id": "HR_03", "authority": {"issuer": "HR Director", "source_class": "corporate_policy", "approval_status": "approved"}, "event_type": "ADDED", "title": "Remote Working Policy v3.0"},
    {"document_id": "D104", "policy_family": "remote_work", "version": "FAQ 2026-06", "publication_date": "2026-06-18", "ingested_at": "2026-06-19T00:00:00Z", "effective_from": "2026-07-01", "effective_to": None, "department": "HR", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "permanent", "conditions": []}, "source_id": "HR_04", "authority": {"issuer": "HR Operations", "source_class": "guide", "approval_status": "approved"}, "event_type": "ADDED", "title": "Remote Working Policy FAQ"},

    {"document_id": "D201", "policy_family": "office_meals", "version": "Pune 1.0", "publication_date": "2026-01-02", "ingested_at": "2026-01-03T00:00:00Z", "effective_from": "2026-01-05", "effective_to": None, "department": "Finance", "scope": {"locations": ["Pune"], "employee_category": "all", "conditions": ["work after 21:00", "one meal per eligible working day"]}, "source_id": "FIN_01", "authority": {"issuer": "Finance Controller", "source_class": "office_supplement", "approval_status": "approved"}, "event_type": "INITIAL_IMPORT", "title": "Pune Late-Working Meal Allowance"},
    {"document_id": "D202", "policy_family": "office_meals", "version": "Mumbai 1.0", "publication_date": "2026-01-02", "ingested_at": "2026-01-03T00:00:00Z", "effective_from": "2026-01-05", "effective_to": None, "department": "Finance", "scope": {"locations": ["Mumbai"], "employee_category": "all", "conditions": ["work after 21:00", "one meal per eligible working day"]}, "source_id": "FIN_02", "authority": {"issuer": "Finance Controller", "source_class": "office_supplement", "approval_status": "approved"}, "event_type": "ADDED", "title": "Mumbai Late-Working Meal Allowance"},
    {"document_id": "D203", "policy_family": "office_meals", "version": "KB edition 1", "publication_date": "2026-02-02", "ingested_at": "2026-02-03T00:00:00Z", "effective_from": "2026-01-05", "effective_to": None, "department": "Finance", "scope": {"locations": ["Pune"], "employee_category": "all", "conditions": ["work after 21:00", "one meal per eligible working day"]}, "source_id": "FIN_03", "authority": {"issuer": "Finance Controller", "source_class": "knowledge_base", "approval_status": "approved"}, "event_type": "ADDED", "title": "Finance Help Article: Pune Evening Meals"},
    {"document_id": "D204", "policy_family": "office_meals", "version": "Mumbai 2.0", "publication_date": "2026-06-01", "ingested_at": "2026-06-02T00:00:00Z", "effective_from": "2026-06-15", "effective_to": None, "department": "Finance", "scope": {"locations": ["Mumbai"], "employee_category": "all", "conditions": ["work after 21:00", "one meal per eligible working day"]}, "source_id": "FIN_04", "authority": {"issuer": "Finance Controller", "source_class": "office_supplement", "approval_status": "approved"}, "event_type": "ADDED", "title": "Mumbai Late-Working Meal Allowance v2.0"},

    {"document_id": "D301", "policy_family": "support_callback", "version": None, "publication_date": None, "ingested_at": "2026-04-01T00:00:00Z", "effective_from": None, "effective_to": None, "department": "Support", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "support staff", "conditions": ["standard-plan incidents", "received during business hours"]}, "source_id": "SUPP_01", "authority": {"issuer": "Support Wiki", "source_class": "wiki", "approval_status": "unknown"}, "event_type": "INITIAL_IMPORT", "title": "Standard Support Callback Guide"},
    {"document_id": "D302", "policy_family": "support_callback", "version": None, "publication_date": None, "ingested_at": "2026-04-02T00:00:00Z", "effective_from": None, "effective_to": None, "department": "Support", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "support staff", "conditions": ["standard-plan incidents", "received during business hours"]}, "source_id": "SUPP_02", "authority": {"issuer": "Support Onboarding", "source_class": "note", "approval_status": "unknown"}, "event_type": "ADDED", "title": "Standard Support Onboarding Note"},
    {"document_id": "D303", "policy_family": "support_callback", "version": "1.0", "publication_date": "2026-04-03", "ingested_at": "2026-04-04T00:00:00Z", "effective_from": "2026-04-03", "effective_to": None, "department": "Support", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "support staff", "conditions": ["standard-plan incidents", "received during business hours"]}, "source_id": "SUPP_03", "authority": {"issuer": "Support Operations", "source_class": "notice", "approval_status": "approved"}, "event_type": "ADDED", "title": "Support Recordkeeping Notice"},
    {"document_id": "D304", "policy_family": "support_callback", "version": "1.0 accessible", "publication_date": "2026-04-05", "ingested_at": "2026-04-06T00:00:00Z", "effective_from": "2026-04-03", "effective_to": None, "department": "Support", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "support staff", "conditions": ["standard-plan incidents", "received during business hours"]}, "source_id": "SUPP_04", "authority": {"issuer": "Support Operations", "source_class": "notice", "approval_status": "approved"}, "event_type": "ADDED", "title": "Support Recordkeeping Accessible Guide"},

    {"document_id": "D401", "policy_family": "backup_retention", "version": "1.0", "publication_date": "2026-01-05", "ingested_at": "2026-01-06T00:00:00Z", "effective_from": "2026-01-10", "effective_to": None, "department": "IT", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "IT staff", "conditions": ["production application backups"]}, "source_id": "IT_01", "authority": {"issuer": "IT Director", "source_class": "standard", "approval_status": "approved"}, "event_type": "INITIAL_IMPORT", "title": "Production Backup Standard v1.0"},
    {"document_id": "D402", "policy_family": "backup_retention", "version": None, "publication_date": "2026-03-05", "ingested_at": "2026-03-06T00:00:00Z", "effective_from": None, "effective_to": None, "department": "IT", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "IT staff", "conditions": ["production application backups"]}, "source_id": "IT_02", "authority": {"issuer": "Infrastructure Team", "source_class": "discussion", "approval_status": "unapproved"}, "event_type": "ADDED", "title": "Infrastructure Team Discussion"},
    {"document_id": "D403", "policy_family": "backup_retention", "version": "2.0", "publication_date": "2026-04-15", "ingested_at": "2026-04-16T00:00:00Z", "effective_from": "2026-05-01", "effective_to": None, "department": "IT", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "IT staff", "conditions": ["production application backups"]}, "source_id": "IT_03", "authority": {"issuer": "IT Director", "source_class": "standard", "approval_status": "approved"}, "event_type": "ADDED", "title": "Production Backup Standard v2.0"},
    {"document_id": "D404", "policy_family": "backup_retention", "version": "2.0 reminder", "publication_date": "2026-04-18", "ingested_at": "2026-04-19T00:00:00Z", "effective_from": "2026-05-01", "effective_to": None, "department": "IT", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "IT staff", "conditions": ["production application backups"]}, "source_id": "IT_04", "authority": {"issuer": "IT Director", "source_class": "standard", "approval_status": "approved"}, "event_type": "ADDED", "title": "Production Backup Standard Reminder"},

    {"document_id": "D501", "policy_family": "deployment", "version": "Production 1.0", "publication_date": "2026-01-08", "ingested_at": "2026-01-09T00:00:00Z", "effective_from": "2026-01-15", "effective_to": None, "department": "Engineering", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "engineering staff", "conditions": ["production deployments"]}, "source_id": "ENG_01", "authority": {"issuer": "Release Manager", "source_class": "sop", "approval_status": "approved"}, "event_type": "INITIAL_IMPORT", "title": "Production Deployment SOP v1.0"},
    {"document_id": "D502", "policy_family": "deployment", "version": "Development 1.0", "publication_date": "2026-02-01", "ingested_at": "2026-02-02T00:00:00Z", "effective_from": "2026-02-05", "effective_to": None, "department": "Engineering", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "engineering staff", "conditions": ["development-environment deployments"]}, "source_id": "ENG_02", "authority": {"issuer": "Release Manager", "source_class": "sop", "approval_status": "approved"}, "event_type": "ADDED", "title": "Development Deployment SOP v1.0"},
    {"document_id": "D503", "policy_family": "deployment", "version": "Production 1.1", "publication_date": "2026-03-20", "ingested_at": "2026-03-21T00:00:00Z", "effective_from": "2026-04-01", "effective_to": None, "department": "Engineering", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "engineering staff", "conditions": ["production deployments"]}, "source_id": "ENG_03", "authority": {"issuer": "Release Manager", "source_class": "sop", "approval_status": "approved"}, "event_type": "ADDED", "title": "Production Deployment SOP v1.1"},
    {"document_id": "D504", "policy_family": "deployment", "version": "Production 1.1 reissue", "publication_date": "2026-04-10", "ingested_at": "2026-04-11T00:00:00Z", "effective_from": "2026-04-01", "effective_to": None, "department": "Engineering", "scope": {"locations": ["Pune", "Mumbai"], "employee_category": "engineering staff", "conditions": ["production deployments"]}, "source_id": "ENG_04", "authority": {"issuer": "Release Manager", "source_class": "sop", "approval_status": "approved"}, "event_type": "ADDED", "title": "Production Deployment SOP Accessible Reissue"}
]

with open(META_DIR / "metadata.jsonl", "w", encoding="utf-8") as f:
    for m in METADATA:
        f.write(json.dumps(m) + "\n")

# 4. Questions
QUESTIONS = [
    {"question_id": "Q001", "policy_family": "remote_work", "question": "How many remote days per week are allowed today, and is a change already announced?", "snapshot_at": "2026-06-20T23:59:59Z", "asked_at": "2026-06-20T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "permanent", "conditions": []}},
    {"question_id": "Q002", "policy_family": "remote_work", "question": "What was the remote-work allowance throughout 2025?", "snapshot_at": "2026-08-01T23:59:59Z", "asked_at": "2026-08-01T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "permanent", "conditions": []}},
    {"question_id": "Q003", "policy_family": "remote_work", "question": "What is the current weekly remote-work maximum?", "snapshot_at": "2026-08-01T23:59:59Z", "asked_at": "2026-08-01T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "permanent", "conditions": []}},
    {"question_id": "Q004", "policy_family": "remote_work", "question": "Give the sequence of remote-work allowances and their start dates since January 2025.", "snapshot_at": "2026-08-01T23:59:59Z", "asked_at": "2026-08-01T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "permanent", "conditions": []}},
    {"question_id": "Q005", "policy_family": "remote_work", "question": "Does the July FAQ create an additional remote-work allowance?", "snapshot_at": "2026-08-01T23:59:59Z", "asked_at": "2026-08-01T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "permanent", "conditions": []}},
    {"question_id": "Q006", "policy_family": "office_meals", "question": "What is the meal ceiling for a Pune employee working after 21:00?", "snapshot_at": "2026-07-01T23:59:59Z", "asked_at": "2026-07-01T23:59:59Z", "user_context": {"locations": ["Pune"], "employee_category": "all", "conditions": ["work after 21:00", "one meal per eligible working day"]}},
    {"question_id": "Q007", "policy_family": "office_meals", "question": "What can a Mumbai employee claim for one meal after working beyond 21:00?", "snapshot_at": "2026-07-01T23:59:59Z", "asked_at": "2026-07-01T23:59:59Z", "user_context": {"locations": ["Mumbai"], "employee_category": "all", "conditions": ["work after 21:00", "one meal per eligible working day"]}},
    {"question_id": "Q008", "policy_family": "office_meals", "question": "Do the current Pune and Mumbai meal ceilings contradict each other?", "snapshot_at": "2026-07-01T23:59:59Z", "asked_at": "2026-07-01T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "all", "conditions": ["work after 21:00", "one meal per eligible working day"]}},
    {"question_id": "Q009", "policy_family": "office_meals", "question": "What was the Mumbai late-working meal ceiling on 1 March 2026?", "snapshot_at": "2026-07-01T23:59:59Z", "asked_at": "2026-07-01T23:59:59Z", "user_context": {"locations": ["Mumbai"], "employee_category": "all", "conditions": ["work after 21:00", "one meal per eligible working day"]}},
    {"question_id": "Q010", "policy_family": "office_meals", "question": "Did the February Pune knowledge-base wording increase the meal ceiling?", "snapshot_at": "2026-03-01T23:59:59Z", "asked_at": "2026-03-01T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "all", "conditions": ["work after 21:00", "one meal per eligible working day"]}},
    {"question_id": "Q011", "policy_family": "support_callback", "question": "What callback deadline currently governs standard-plan incidents received during business hours?", "snapshot_at": "2026-04-10T23:59:59Z", "asked_at": "2026-04-10T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "support staff", "conditions": ["standard-plan incidents", "received during business hours"]}},
    {"question_id": "Q012", "policy_family": "support_callback", "question": "Which two callback deadlines are recorded for the same incident scope?", "snapshot_at": "2026-04-10T23:59:59Z", "asked_at": "2026-04-10T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "support staff", "conditions": ["standard-plan incidents", "received during business hours"]}},
    {"question_id": "Q013", "policy_family": "support_callback", "question": "Does the approved 3 April notice settle the disputed callback deadline?", "snapshot_at": "2026-04-10T23:59:59Z", "asked_at": "2026-04-10T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "support staff", "conditions": ["standard-plan incidents", "received during business hours"]}},
    {"question_id": "Q014", "policy_family": "support_callback", "question": "Must standard-plan callbacks be recorded in the service-desk ticket?", "snapshot_at": "2026-04-10T23:59:59Z", "asked_at": "2026-04-10T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "support staff", "conditions": ["standard-plan incidents", "received during business hours"]}},
    {"question_id": "Q015", "policy_family": "support_callback", "question": "Does the accessible copy add a new callback-recording requirement?", "snapshot_at": "2026-04-10T23:59:59Z", "asked_at": "2026-04-10T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "support staff", "conditions": ["standard-plan incidents", "received during business hours"]}},
    {"question_id": "Q016", "policy_family": "backup_retention", "question": "What production-backup retention applies today?", "snapshot_at": "2026-03-10T23:59:59Z", "asked_at": "2026-03-10T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "IT staff", "conditions": ["production application backups"]}},
    {"question_id": "Q017", "policy_family": "backup_retention", "question": "Did the March discussion change production-backup retention to seven days?", "snapshot_at": "2026-03-10T23:59:59Z", "asked_at": "2026-03-10T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "IT staff", "conditions": ["production application backups"]}},
    {"question_id": "Q018", "policy_family": "backup_retention", "question": "What backup retention applies today, and what approved change is pending?", "snapshot_at": "2026-04-20T23:59:59Z", "asked_at": "2026-04-20T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "IT staff", "conditions": ["production application backups"]}},
    {"question_id": "Q019", "policy_family": "backup_retention", "question": "What is the current production-backup retention period?", "snapshot_at": "2026-05-10T23:59:59Z", "asked_at": "2026-05-10T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "IT staff", "conditions": ["production application backups"]}},
    {"question_id": "Q020", "policy_family": "backup_retention", "question": "What production-backup retention governed on 1 February 2026?", "snapshot_at": "2026-05-10T23:59:59Z", "asked_at": "2026-05-10T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "IT staff", "conditions": ["production application backups"]}},
    {"question_id": "Q021", "policy_family": "deployment", "question": "How many reviewers does a production deployment need after version 1.1?", "snapshot_at": "2026-04-20T23:59:59Z", "asked_at": "2026-04-20T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "engineering staff", "conditions": ["production deployments"]}},
    {"question_id": "Q022", "policy_family": "deployment", "question": "How long must production deployment health checks run now?", "snapshot_at": "2026-04-20T23:59:59Z", "asked_at": "2026-04-20T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "engineering staff", "conditions": ["production deployments"]}},
    {"question_id": "Q023", "policy_family": "deployment", "question": "What changed between production deployment versions 1.0 and 1.1, and what stayed the same?", "snapshot_at": "2026-04-20T23:59:59Z", "asked_at": "2026-04-20T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "engineering staff", "conditions": ["production deployments"]}},
    {"question_id": "Q024", "policy_family": "deployment", "question": "Do development deployments need the same reviewer count as production?", "snapshot_at": "2026-04-20T23:59:59Z", "asked_at": "2026-04-20T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "engineering staff", "conditions": ["production deployments"]}},
    {"question_id": "Q025", "policy_family": "deployment", "question": "How long should development-environment deployment logs be retained?", "snapshot_at": "2026-02-10T23:59:59Z", "asked_at": "2026-02-10T23:59:59Z", "user_context": {"locations": ["Pune", "Mumbai"], "employee_category": "engineering staff", "conditions": ["development-environment deployments"]}}
]

with open(EVAL_DIR / "questions.jsonl", "w", encoding="utf-8") as f:
    for q in QUESTIONS:
        f.write(json.dumps(q) + "\n")

# 5. Ground Truth Answers
ANSWERS = [
    {
        "question_id": "Q001",
        "policy_family": "remote_work",
        "gold_answer": "Three days per week currently; an approved change to four days per week has been announced starting 1 July 2026.",
        "required_claims": ["three days", "four days", "1 July"],
        "forbidden_claims": ["four days currently", "two days currently"],
        "acceptable_evidence_sets": [["D102#C1", "D103#C1"], ["D102#C1", "D104#C1"]],
        "question_category": "future_effective",
        "expected_difficulty": "medium"
    },
    {
        "question_id": "Q002",
        "policy_family": "remote_work",
        "gold_answer": "Two days per week throughout 2025.",
        "required_claims": ["two days"],
        "forbidden_claims": ["three days", "four days"],
        "acceptable_evidence_sets": [["D101#C1", "D101#C2"]],
        "question_category": "historical",
        "expected_difficulty": "easy"
    },
    {
        "question_id": "Q003",
        "policy_family": "remote_work",
        "gold_answer": "Four days per week.",
        "required_claims": ["four days"],
        "forbidden_claims": ["two days", "three days"],
        "acceptable_evidence_sets": [["D103#C1"], ["D104#C1"]],
        "question_category": "current",
        "expected_difficulty": "easy"
    },
    {
        "question_id": "Q004",
        "policy_family": "remote_work",
        "gold_answer": "Two days per week from 1 January 2025; three days per week from 1 February 2026; four days per week from 1 July 2026.",
        "required_claims": ["two days", "three days", "four days"],
        "forbidden_claims": [],
        "acceptable_evidence_sets": [["D101#C1", "D102#C1", "D103#C1"]],
        "question_category": "history_chain",
        "expected_difficulty": "hard"
    },
    {
        "question_id": "Q005",
        "policy_family": "remote_work",
        "gold_answer": "No, the FAQ does not create an additional allowance; it restates the existing four-day allowance from version 3.0.",
        "required_claims": ["no", "restates"],
        "forbidden_claims": ["creates an additional", "yes"],
        "acceptable_evidence_sets": [["D104#C1", "D104#C2"]],
        "question_category": "duplicate",
        "expected_difficulty": "medium"
    },
    {
        "question_id": "Q006",
        "policy_family": "office_meals",
        "gold_answer": "INR 250 for one meal per eligible working day.",
        "required_claims": ["250"],
        "forbidden_claims": ["350", "400"],
        "acceptable_evidence_sets": [["D201#C1"], ["D203#C1"]],
        "question_category": "current",
        "expected_difficulty": "easy"
    },
    {
        "question_id": "Q007",
        "policy_family": "office_meals",
        "gold_answer": "INR 400 for one meal per eligible working day.",
        "required_claims": ["400"],
        "forbidden_claims": ["350", "250"],
        "acceptable_evidence_sets": [["D204#C1"]],
        "question_category": "current",
        "expected_difficulty": "easy"
    },
    {
        "question_id": "Q008",
        "policy_family": "office_meals",
        "gold_answer": "No, they do not contradict each other; they apply to different office locations (Pune vs Mumbai).",
        "required_claims": ["no", "different"],
        "forbidden_claims": ["yes", "contradict"],
        "acceptable_evidence_sets": [["D201#C1", "D204#C1"], ["D203#C1", "D204#C1"]],
        "question_category": "scope_difference",
        "expected_difficulty": "medium"
    },
    {
        "question_id": "Q009",
        "policy_family": "office_meals",
        "gold_answer": "INR 350 for one meal per eligible working day.",
        "required_claims": ["350"],
        "forbidden_claims": ["400", "250"],
        "acceptable_evidence_sets": [["D202#C1"]],
        "question_category": "historical",
        "expected_difficulty": "medium"
    },
    {
        "question_id": "Q010",
        "policy_family": "office_meals",
        "gold_answer": "No, the February knowledge-base wording did not increase the ceiling; it restated the existing INR 250 limit.",
        "required_claims": ["no", "250"],
        "forbidden_claims": ["yes", "increased"],
        "acceptable_evidence_sets": [["D203#C1", "D203#C2"]],
        "question_category": "duplicate",
        "expected_difficulty": "easy"
    },
    {
        "question_id": "Q011",
        "policy_family": "support_callback",
        "gold_answer": "Cannot establish one definitive deadline because available sources conflict between four business hours and eight business hours with no authoritative resolution.",
        "required_claims": ["conflict", "four", "eight"],
        "forbidden_claims": ["governs four hours definitely", "governs eight hours definitely"],
        "acceptable_evidence_sets": [["D301#C1", "D302#C1"]],
        "question_category": "unresolved_conflict",
        "expected_difficulty": "hard"
    },
    {
        "question_id": "Q012",
        "policy_family": "support_callback",
        "gold_answer": "Four business hours and eight business hours.",
        "required_claims": ["four", "eight"],
        "forbidden_claims": [],
        "acceptable_evidence_sets": [["D301#C1", "D302#C1"]],
        "question_category": "conflict_identification",
        "expected_difficulty": "easy"
    },
    {
        "question_id": "Q013",
        "policy_family": "support_callback",
        "gold_answer": "No, the 3 April notice establishes recordkeeping requirements only and explicitly does not define a callback deadline.",
        "required_claims": ["no", "does not define"],
        "forbidden_claims": ["yes", "settles the deadline"],
        "acceptable_evidence_sets": [["D303#C2"]],
        "question_category": "scope_difference",
        "expected_difficulty": "medium"
    },
    {
        "question_id": "Q014",
        "policy_family": "support_callback",
        "gold_answer": "Yes, every callback for a standard-plan incident received during business hours must be recorded in the service-desk ticket.",
        "required_claims": ["yes", "recorded"],
        "forbidden_claims": ["no"],
        "acceptable_evidence_sets": [["D303#C1"], ["D304#C1"]],
        "question_category": "straightforward",
        "expected_difficulty": "easy"
    },
    {
        "question_id": "Q015",
        "policy_family": "support_callback",
        "gold_answer": "No, the accessible copy is an equivalent restatement of the existing ticket-recording requirement.",
        "required_claims": ["no", "restatement"],
        "forbidden_claims": ["yes", "adds a new requirement"],
        "acceptable_evidence_sets": [["D304#C1", "D304#C2"]],
        "question_category": "duplicate",
        "expected_difficulty": "easy"
    },
    {
        "question_id": "Q016",
        "policy_family": "backup_retention",
        "gold_answer": "30 days.",
        "required_claims": ["30 days"],
        "forbidden_claims": ["7 days", "45 days"],
        "acceptable_evidence_sets": [["D401#C1"]],
        "question_category": "current",
        "expected_difficulty": "easy"
    },
    {
        "question_id": "Q017",
        "policy_family": "backup_retention",
        "gold_answer": "No, the March discussion was an unapproved team suggestion and did not change the approved standard.",
        "required_claims": ["no", "unapproved"],
        "forbidden_claims": ["yes", "changed to 7 days"],
        "acceptable_evidence_sets": [["D402#C1", "D402#C2", "D401#C2"]],
        "question_category": "lower_authority",
        "expected_difficulty": "medium"
    },
    {
        "question_id": "Q018",
        "policy_family": "backup_retention",
        "gold_answer": "30 days applies today; an approved revision to 45 days is pending and takes effect on 1 May 2026.",
        "required_claims": ["30 days", "45 days", "1 May"],
        "forbidden_claims": ["45 days applies today"],
        "acceptable_evidence_sets": [["D401#C1", "D403#C1"]],
        "question_category": "future_effective",
        "expected_difficulty": "medium"
    },
    {
        "question_id": "Q019",
        "policy_family": "backup_retention",
        "gold_answer": "45 days.",
        "required_claims": ["45 days"],
        "forbidden_claims": ["30 days", "7 days"],
        "acceptable_evidence_sets": [["D403#C1"], ["D404#C1"]],
        "question_category": "current",
        "expected_difficulty": "easy"
    },
    {
        "question_id": "Q020",
        "policy_family": "backup_retention",
        "gold_answer": "30 days.",
        "required_claims": ["30 days"],
        "forbidden_claims": ["45 days"],
        "acceptable_evidence_sets": [["D401#C1"]],
        "question_category": "historical",
        "expected_difficulty": "medium"
    },
    {
        "question_id": "Q021",
        "policy_family": "deployment",
        "gold_answer": "Two reviewers.",
        "required_claims": ["two"],
        "forbidden_claims": ["one"],
        "acceptable_evidence_sets": [["D503#C1"], ["D504#C1"], ["D501#C1"]],
        "question_category": "unchanged",
        "expected_difficulty": "easy"
    },
    {
        "question_id": "Q022",
        "policy_family": "deployment",
        "gold_answer": "30 minutes.",
        "required_claims": ["30 minutes"],
        "forbidden_claims": ["15 minutes"],
        "acceptable_evidence_sets": [["D503#C3"], ["D504#C3"]],
        "question_category": "partial_update",
        "expected_difficulty": "medium"
    },
    {
        "question_id": "Q023",
        "policy_family": "deployment",
        "gold_answer": "The post-deployment health check duration increased from 15 minutes to 30 minutes; reviewer approval (two reviewers) and log retention (90 days) remained unchanged.",
        "required_claims": ["health check", "30 minutes", "unchanged"],
        "forbidden_claims": [],
        "acceptable_evidence_sets": [["D501#C1", "D501#C2", "D501#C3", "D503#C3", "D503#C4"]],
        "question_category": "unchanged",
        "expected_difficulty": "hard"
    },
    {
        "question_id": "Q024",
        "policy_family": "deployment",
        "gold_answer": "No, development-environment deployments require one reviewer, while production deployments require two reviewers.",
        "required_claims": ["no", "one", "two"],
        "forbidden_claims": ["yes"],
        "acceptable_evidence_sets": [["D502#C1", "D501#C1"], ["D502#C1", "D503#C1"]],
        "question_category": "scope_difference",
        "expected_difficulty": "medium"
    },
    {
        "question_id": "Q025",
        "policy_family": "deployment",
        "gold_answer": "14 days.",
        "required_claims": ["14 days"],
        "forbidden_claims": ["90 days"],
        "acceptable_evidence_sets": [["D502#C2"]],
        "question_category": "straightforward",
        "expected_difficulty": "easy"
    }
]

with open(GT_DIR / "answers.jsonl", "w", encoding="utf-8") as f:
    for a in ANSWERS:
        f.write(json.dumps(a) + "\n")

# 6. Splits
SPLITS = {
    "split_unit": "policy_family",
    "train": [],
    "development": [
        "remote_work",
        "office_meals",
        "support_callback",
        "backup_retention",
        "deployment"
    ],
    "test": [
        "procurement",
        "visitor_access",
        "training_budget",
        "equipment_service",
        "invoice_disputes"
    ],
    "reason": "Development partition contains 5 families; test partition reserved for frozen confirmatory run."
}
(BASE_DIR / "splits.json").write_text(json.dumps(SPLITS, indent=2), encoding="utf-8")

# 7. Runner Snapshots
SNAPSHOTS = [
    {"snapshot_at": "2026-02-10T23:59:59Z", "eligible_doc_ids": ["D101", "D102", "D201", "D202", "D203", "D401", "D501", "D502"]},
    {"snapshot_at": "2026-03-01T23:59:59Z", "eligible_doc_ids": ["D101", "D102", "D201", "D202", "D203", "D401", "D501", "D502"]},
    {"snapshot_at": "2026-03-10T23:59:59Z", "eligible_doc_ids": ["D101", "D102", "D201", "D202", "D203", "D401", "D402", "D501", "D502"]},
    {"snapshot_at": "2026-04-10T23:59:59Z", "eligible_doc_ids": ["D101", "D102", "D201", "D202", "D203", "D301", "D302", "D303", "D304", "D401", "D402", "D501", "D502", "D503"]},
    {"snapshot_at": "2026-04-20T23:59:59Z", "eligible_doc_ids": ["D101", "D102", "D201", "D202", "D203", "D301", "D302", "D303", "D304", "D401", "D402", "D403", "D404", "D501", "D502", "D503", "D504"]},
    {"snapshot_at": "2026-05-10T23:59:59Z", "eligible_doc_ids": ["D101", "D102", "D201", "D202", "D203", "D301", "D302", "D303", "D304", "D401", "D402", "D403", "D404", "D501", "D502", "D503", "D504"]},
    {"snapshot_at": "2026-06-20T23:59:59Z", "eligible_doc_ids": ["D101", "D102", "D103", "D104", "D201", "D202", "D203", "D204", "D301", "D302", "D303", "D304", "D401", "D402", "D403", "D404", "D501", "D502", "D503", "D504"]},
    {"snapshot_at": "2026-07-01T23:59:59Z", "eligible_doc_ids": ["D101", "D102", "D103", "D104", "D201", "D202", "D203", "D204", "D301", "D302", "D303", "D304", "D401", "D402", "D403", "D404", "D501", "D502", "D503", "D504"]},
    {"snapshot_at": "2026-08-01T23:59:59Z", "eligible_doc_ids": ["D101", "D102", "D103", "D104", "D201", "D202", "D203", "D204", "D301", "D302", "D303", "D304", "D401", "D402", "D403", "D404", "D501", "D502", "D503", "D504"]}
]

with open(RUNNER_DIR / "snapshots.jsonl", "w", encoding="utf-8") as f:
    for s in SNAPSHOTS:
        f.write(json.dumps(s) + "\n")

print(f"Dataset populated successfully: {len(DOCUMENTS)} docs, {len(METADATA)} metadata, {len(QUESTIONS)} questions, {len(ANSWERS)} gold answers.")

"""Behaviour tests on independent uploaded company documents, not canned answers."""
import json
from unittest.mock import patch
from enterprise.api import create_app,ROOT
from .conftest import ADMIN,EMPLOYEE,add,ask


def test_default_workspace_is_empty(tmp_path):
    app=create_app(tmp_path)
    assert app.state.store.documents(ADMIN)==[]


def test_unseen_collection_and_policy_evolution(store,engine):
    first=add(store,'# Solvent disposal\n\nThe solvent storage limit is 140 litres.',family='laboratory_safety',document_key='chemical_storage',rule_key='storage',version='1')
    second=add(store,'# Solvent disposal\n\nThe solvent storage limit is 95 litres.',family='laboratory_safety',document_key='chemical_storage',rule_key='storage',version='2',effective_from='2026-07-01')
    result=ask(engine,'What is the solvent storage limit?')
    assert {c['document_id'] for c in result['evidence']}=={second}
    previous=ask(engine,'What was the solvent storage limit on 2026-03-01?')
    assert {c['document_id'] for c in previous['evidence']}=={first}


def test_explanation_keeps_proposal_as_supporting(store,engine):
    policy=add(store,'The solvent storage limit is 140 litres.',family='laboratory_safety',document_key='chemical_storage',rule_key='storage')
    proposal=add(store,'The solvent storage limit should be 200 litres.',family='laboratory_safety',document_key='proposal',rule_key='storage',status='draft')
    current=ask(engine,'What is the solvent storage limit?')
    assert {c['document_id'] for c in current['evidence']}=={policy}
    explanation=ask(engine,'Did the proposal change the solvent storage limit?')
    support=[c for c in explanation['evidence'] if c['document_id']==proposal]
    assert support and support[0]['evidence_role']=='supporting'
    assert any(c['document_id']==policy and c['evidence_role']=='governing' for c in explanation['evidence'])


def test_comparison_retains_both_environments(store,engine):
    a=add(store,'The deployment review needs one reviewer.',conditions=['sandbox deployments'],family='release_control',document_key='sandbox',rule_key='review')
    b=add(store,'The deployment review needs three reviewers.',conditions=['live deployments'],family='release_control',document_key='live',rule_key='review')
    out=ask(engine,'Do sandbox and live deployments need the same review?',conditions=['sandbox deployments'])
    assert {c['document_id'] for c in out['evidence']}=={a,b}
    assert not out['resolution']['conflicts']


def test_references_explain_without_overriding(store,engine):
    policy=add(store)
    reference=add(store,'This FAQ restates the remote work allowance and adds no allowance.',document_key='faq',document_type='reference',authority=100)
    current=ask(engine)
    assert {c['document_id'] for c in current['evidence']}=={policy}
    explanation=ask(engine,'Does this FAQ create an additional remote work allowance?')
    assert reference in {c['document_id'] for c in explanation['evidence']}
    assert policy in {c['document_id'] for c in explanation['evidence'] if c['evidence_role']=='governing'}


def test_shared_clause_numbers_do_not_merge_unrelated_rules(store,engine):
    add(store,'[C1] Solvent storage must not exceed 100 litres.',family='laboratory',document_key='solvent',rule_key='')
    add(store,'[C1] Emergency evacuation drills occur every quarter.',family='laboratory',document_key='evacuation',rule_key='')
    out=ask(engine,'Explain the laboratory procedures.')
    assert not out['resolution']['conflicts']


def test_warm_cache_bypasses_corpus_embedding_and_search(store,engine):
    add(store)
    first=ask(engine,agent='amo')
    with patch.object(store,'corpus_snapshot',side_effect=AssertionError('Corpus loaded on hit')),patch.object(engine.retriever,'vector',side_effect=AssertionError('Embedded on hit')),patch.object(engine.retriever,'search',side_effect=AssertionError('Searched on hit')):
        warm=ask(engine,agent='amo')
    assert warm['cache']['context_hit']
    assert warm['citations']==first['citations']
    assert warm['timing']['corpus_load_ms']==0 and warm['timing']['embedding_ms']==0


def test_warm_cache_does_not_commit_usage(store,engine):
    add(store)
    ask(engine,agent='amo')
    # Only the lightweight revision read is needed on a normal cache hit.
    with patch.object(store,'flush_usage',side_effect=AssertionError('Usage flushed on query')):
        ask(engine,agent='amo')
    assert store._pending_usage


def test_amo_preserves_cake_evidence(store,engine):
    add(store)
    add(store,'Remote work maximum is five days.',document_key='independent')
    cake=ask(engine,agent='cake');amo=ask(engine,agent='amo')
    assert {c['id'] for c in cake['evidence']}=={c['id'] for c in amo['evidence']}
    assert cake['resolution']['conflicts']==amo['resolution']['conflicts']


def test_metadata_approval_change_invalidates_cached_context(store,engine):
    ident=add(store,status='draft')
    assert not ask(engine,agent='amo')['evidence']
    store.update_metadata(ident,{'status':'approved'},ADMIN)
    out=ask(engine,agent='amo')
    assert out['evidence'] and not out['cache']['context_hit']


def test_customer_evaluation_import_and_real_scores(client):
    from .test_workflows import upload
    ident=upload(client).json()['id']
    clauses=client.get('/api/documents/'+ident+'/clauses').json()['clauses']
    source=clauses[0]['id']
    dataset=client.post('/api/evaluations',json={'name':'Company verification','questions':[{'id':'custom','question':'What is the remote work maximum?','as_of':'2026-08-01','acceptable_evidence_sets':[[source]]}]}).json()
    result=client.post('/api/benchmark',json={'dataset_id':dataset['id'],'generator':'evidence','repeats':2}).json()
    assert all(r['evidence_set_coverage']==1 for r in result['runs'])
    assert all(r['exact_claim_accuracy'] is None for r in result['runs'])
    assert result['runs'][-1]['context_cache_hit_rate']==1
    assert all(row['same_evidence_as_cold'] for row in result['runs'][-1]['rows'])


def test_missing_research_corpus_is_not_scored_as_bad_system(client):
    result=client.post('/api/benchmark',json={'dataset_id':'development'})
    assert result.status_code==400 and 'optional research' in result.json()['detail']


def test_streaming_endpoint_completes_with_valid_sources(client):
    from .test_workflows import upload
    upload(client)
    r=client.post('/api/chat/stream',json={'question':'What is the remote work maximum?','as_of':'2026-08-01'})
    events=[json.loads(line) for line in r.text.splitlines()]
    assert events[0]['event']=='status'
    final=next(x for x in events if x['event']=='complete')
    assert final['data']['results'][0]['citations']


def test_conversation_followup_and_ownership(client):
    from .test_workflows import upload
    upload(client,locations=['North'])
    upload(client,content=b'# Allowance\n\nRemote work maximum is six days.',document_key='south',locations=['South'])
    first=client.post('/api/chat',json={'question':'What is the remote work maximum in North?','as_of':'2026-08-01'}).json()
    ident=first['conversation_id']
    second=client.post('/api/chat',json={'question':'What about South?','as_of':'2026-08-01','conversation_id':ident}).json()
    assert second['results'][0]['followup_context_used']
    assert 'six' in second['results'][0]['answer'] and 'three' not in second['results'][0]['answer']
    assert len(client.get('/api/conversations/'+ident).json()['runs'])==2
    client.post('/api/logout');client.post('/api/login',json={'username':'employee','password':'EvoLLM-Employee-2026!'})
    assert client.get('/api/conversations/'+ident).status_code==404


def test_no_runtime_policy_family_or_document_id_branches():
    text=(ROOT/'enterprise/resolution.py').read_text()+(ROOT/'enterprise/planning.py').read_text()+(ROOT/'enterprise/engine.py').read_text()
    for name in ('support_callback','remote_work','D101','D502'):
        assert name not in text


def test_diagnosed_research_cases_retain_needed_evidence(tmp_path):
    app=create_app(tmp_path,seed=True);engine=app.state.engine
    questions=[json.loads(x) for x in (ROOT/'data/evaluation/development/questions.jsonl').read_text().splitlines()]
    gold={g['question_id']:g for g in map(json.loads,(ROOT/'data/ground_truth/development/answers.jsonl').read_text().splitlines())}
    for q in questions:
        if q['question_id'] in ('Q005','Q008','Q010','Q017','Q024'):
            out=engine.run(q['question'],'cake',ADMIN,q['snapshot_at'],q['user_context'])
            ids={c['id'] for c in out['evidence']}
            assert any(set(s)<=ids for s in gold[q['question_id']]['acceptable_evidence_sets'])


def test_custom_department_account_and_document_access(client):
    from .test_workflows import upload
    r=client.post('/api/users',json={'username':'legal_user','password':'Company-Legal-2026!','department':'Legal','role':'employee'})
    assert r.status_code==200
    ident=upload(client,audience='Legal').json()['id']
    client.post('/api/logout');client.post('/api/login',json={'username':'legal_user','password':'Company-Legal-2026!'})
    assert ident in {d['id'] for d in client.get('/api/documents').json()}
    assert client.post('/api/users',json={}).status_code==403


def test_employee_cannot_edit_source_authority(client):
    from .test_workflows import upload
    ident=upload(client).json()['id']
    client.post('/api/logout');client.post('/api/login',json={'username':'employee','password':'EvoLLM-Employee-2026!'})
    assert client.patch('/api/documents/'+ident+'/metadata',json={'authority':100}).status_code==403


def test_password_change_checks_current_secret(client):
    assert client.post('/api/account/password',json={'current_password':'wrong','new_password':'New-Workspace-2026!'}).status_code==400
    assert client.post('/api/account/password',json={'current_password':'EvoLLM-Admin-2026!','new_password':'New-Workspace-2026!'}).status_code==200
    client.post('/api/logout')
    assert client.post('/api/login',json={'username':'admin','password':'EvoLLM-Admin-2026!'}).status_code==401
    assert client.post('/api/login',json={'username':'admin','password':'New-Workspace-2026!'}).status_code==200


def test_legacy_history_migrates_to_owned_conversation(store):
    from enterprise.store import Store
    payload={'results':[{'question':'An earlier question','answer':'Earlier evidence'}]}
    store.save_history(ADMIN,payload)
    upgraded=Store(store.root)
    row=upgraded.history(ADMIN)[0]
    assert row['conversation_id']
    assert upgraded.conversation_runs(ADMIN,row['conversation_id'])[0]['results'][0]['question']=='An earlier question'

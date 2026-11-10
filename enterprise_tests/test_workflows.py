import io
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from enterprise.engine import target_date, BoundedCache
from enterprise.parsing import dynamic_chunks, parse_file
from enterprise.store import stamp
from .conftest import ADMIN, EMPLOYEE, add, ask


def test_revision_supersedes_only_same_lineage(store, engine):
    old = add(store)
    new = add(store, 'Remote work maximum is four days per week.', version='2', effective_from='2026-07-01')
    out = ask(engine)
    assert new in {c['document_id'] for c in out['evidence']}
    assert old not in {c['document_id'] for c in out['evidence']}
    assert any('Superseded' in x['reason'] for x in out['resolution']['excluded'])

def test_historical_day_is_exact(store, engine):
    old = add(store)
    add(store, 'Remote work maximum is four days.', version='2', effective_from='2026-03-15')
    out = ask(engine, 'What remote work maximum applied on 1 March 2026?')
    assert out['resolution']['target_date'] == '2026-03-01'
    assert {c['document_id'] for c in out['evidence']} == {old}

@pytest.mark.parametrize('text,expected', [('on 2026-03-01','2026-03-01'),('on 1 March 2026','2026-03-01'),('throughout 2025','2025-12-31')])
def test_target_dates(text, expected):
    assert target_date(text, '2026-08-01') == expected

def test_future_announcements_are_separate(store, engine):
    old = add(store)
    new = add(store, 'Remote maximum is four days.', version='2', effective_from='2026-09-01')
    current = ask(engine)
    assert {c['document_id'] for c in current['evidence']} == {old}
    announced = ask(engine, 'What remote work maximum applies and what change is announced?')
    assert {c['document_id'] for c in announced['evidence']} == {old,new}
    assert announced['resolution']['future'][0]['document_id'] == new

def test_independent_sources_conflict_even_with_different_dates(store, engine):
    add(store)
    add(store, 'Remote work maximum is five days.', document_key='other_hr', effective_from='2026-04-01')
    out = ask(engine)
    assert out['resolution']['conflicts']
    assert 'Unresolved conflict' in out['answer']

def test_same_date_conflict_not_silently_selected(store, engine):
    add(store)
    add(store, 'Remote work maximum is seven days.', version='2')
    assert ask(engine)['resolution']['conflicts']

def test_draft_cannot_supersede_approved(store, engine):
    old=add(store)
    add(store,'Remote work maximum is seven days.',version='2',status='draft',effective_from='2026-07-01')
    assert {c['document_id'] for c in ask(engine)['evidence']} == {old}

def test_withdrawn_and_expired_not_used(store, engine):
    add(store,status='withdrawn')
    add(store,version='2',effective_to='2026-02-01')
    assert not ask(engine)['evidence']

def test_unknown_approval_requires_review(store, engine):
    add(store,status='unknown')
    assert ask(engine)['resolution']['conflicts']

def test_snapshot_excludes_not_yet_ingested(store, engine):
    add(store,ingested_at='2026-09-01T00:00:00Z')
    assert not ask(engine)['evidence']

def test_timezone_snapshot_canonicalization():
    assert stamp('2026-01-01T05:30:00+05:30') == stamp('2026-01-01T00:00:00Z')

def test_scopes_do_not_create_false_conflicts(store, engine):
    pune=add(store,locations=['Pune'])
    add(store,'Remote work maximum is four days.',document_key='mumbai',locations=['Mumbai'])
    out=ask(engine,'What remote work maximum applies in Pune?',locations=['Pune'])
    assert {c['document_id'] for c in out['evidence']} == {pune}
    assert not out['resolution']['conflicts']

def test_employee_permissions_precede_retrieval(store, engine):
    add(store,'Secret remote work maximum is ninety days.',audience='HR')
    for agent in ('baseline','cake','amo'):
        out=ask(engine,agent=agent,user=EMPLOYEE)
        assert not out['evidence'] and not out['retrieved']
        assert 'ninety' not in out['answer']

def test_tenant_isolation(store, engine):
    add(store)
    other=dict(ADMIN,tenant='other')
    assert not ask(engine,user=other)['evidence']

def test_cache_hit_and_invalidation_on_upload(store, engine):
    add(store)
    assert not ask(engine,agent='amo')['cache']['context_hit']
    assert ask(engine,agent='amo')['cache']['context_hit']
    add(store,'Remote work maximum is four days.',version='2',effective_from='2026-07-01')
    out=ask(engine,agent='amo')
    assert not out['cache']['context_hit'] and 'four' in out['answer']

def test_cache_invalidates_on_delete(store, engine):
    ident=add(store)
    ask(engine,agent='amo')
    store.remove(ident,ADMIN)
    out=ask(engine,agent='amo')
    assert not out['cache']['context_hit'] and not out['evidence']

def test_cached_admin_context_cannot_leak(store, engine):
    add(store,audience='HR')
    ask(engine,agent='amo')
    assert not ask(engine,agent='amo',user=EMPLOYEE)['evidence']

def test_cache_separates_dates_and_scopes(store, engine):
    add(store)
    ask(engine,agent='amo',locations=['Pune'])
    assert not ask(engine,agent='amo',locations=['Mumbai'])['cache']['context_hit']
    assert not ask(engine,agent='amo',as_of='2026-08-02',locations=['Pune'])['cache']['context_hit']

def test_compare_uses_same_snapshot_and_corpus(store, engine):
    add(store)
    out=engine.compare('What is the remote work maximum?',ADMIN,'2026-08-01',{})
    assert len(out['results'])==3
    assert len({r['revision'] for r in out['results']})==1
    assert all(r['timing']['total_ms']>=r['timing']['generation_ms'] for r in out['results'])

def test_dynamic_chunking_keeps_clause_boundaries():
    chunks=dynamic_chunks([{'text':'# Policy\n\n[C1] Three remote days.\n[C2] Four reviewers.','page':2}])
    assert len(chunks)==2
    assert chunks[0]['clause']=='C1' and 'reviewers' not in chunks[0]['text']
    assert chunks[1]['page']==2

def test_dynamic_chunk_size_and_parent_identity():
    chunks=dynamic_chunks([{'text':'# Long rule\n\n[C1] '+('A sentence with several words. '*100),'page':3}],max_words=30)
    assert len(chunks)>2 and all(len(c['text'].split())<=30 for c in chunks)
    assert {c['rule_key'] for c in chunks}=={'C1'}

def test_oversized_clause_parts_not_false_conflict(store, engine):
    add(store,'# Remote\n\n[C1] '+('Remote work has a maximum of three days. '*100))
    assert not ask(engine)['resolution']['conflicts']

@pytest.mark.parametrize('filename,data', [('bad.exe',b'abc'),('x.txt',b'\xff\xfe'),('empty.txt',b'')])
def test_parser_rejects_bad_inputs(filename,data):
    with pytest.raises(ValueError):parse_file(filename,data)

def test_csv_preserves_column_names():
    chunks=dynamic_chunks(parse_file('policy.csv',b'Office,Limit\nPune,250\nMumbai,300'))
    assert 'Office: Pune' in chunks[0]['text'] and 'Limit: 250' in chunks[0]['text']

def test_docx_tables_in_document_order():
    from docx import Document
    doc=Document();doc.add_heading('First',1);doc.add_paragraph('Before table')
    table=doc.add_table(rows=1,cols=2);table.cell(0,0).text='Rule';table.cell(0,1).text='Three days'
    doc.add_paragraph('After table');buffer=io.BytesIO();doc.save(buffer)
    text=parse_file('test.docx',buffer.getvalue())[0]['text']
    assert text.index('Before table')<text.index('Three days')<text.index('After table')

def test_pdf_retains_page_reference():
    # A minimal valid one-page PDF, so this test needs no additional PDF authoring package.
    stream=b'BT /F1 12 Tf 50 700 Td (Remote work maximum is three days.) Tj ET'
    objects=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
        b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream']
    data=b'%PDF-1.4\n';offsets=[0]
    for i,obj in enumerate(objects,1):
        offsets.append(len(data));data+=str(i).encode()+b' 0 obj\n'+obj+b'\nendobj\n'
    xref=len(data);data+=b'xref\n0 6\n0000000000 65535 f \n'
    data+=b''.join(f'{offset:010d} 00000 n \n'.encode() for offset in offsets[1:])
    data+=b'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n'+str(xref).encode()+b'\n%%EOF'
    assert parse_file('policy.pdf',data)[0]['page']==1

def test_cache_eviction_and_thread_safety():
    cache=BoundedCache(3)
    with ThreadPoolExecutor(4) as pool:list(pool.map(lambda n:cache.set(n,{'n':n}),range(50)))
    assert len(cache.data)==3
    # Submission order is not completion order: any worker key may be evicted.
    # Insert a known entry after all workers finish to test defensive copying.
    cache.set('copy_check',{'n':49})
    value=cache.get('copy_check')
    assert value is not None
    value['n']=0
    assert cache.get('copy_check')['n']==49
    assert len(cache.data)==3

    # Force reversed completion order instead of depending on OS scheduling.
    from threading import Event
    earlier_write_done=Event()
    ordered_cache=BoundedCache(1)
    def submitted_first():
        assert earlier_write_done.wait(timeout=5)
        ordered_cache.set(0,{'n':0})
    def submitted_last():
        ordered_cache.set(49,{'n':49})
        earlier_write_done.set()
    with ThreadPoolExecutor(2) as pool:
        first=pool.submit(submitted_first)
        last=pool.submit(submitted_last)
        first.result(timeout=10)
        last.result(timeout=10)
    assert ordered_cache.get(49) is None
    assert ordered_cache.get(0)=={'n':0}

def test_upload_retry_idempotent(store):
    ident=add(store)
    assert add(store)==ident and store.revision()==1

def test_duplicate_version_edit_rejected(store):
    add(store)
    with pytest.raises(ValueError,match='already exists'):add(store,'Changed remote maximum.')

def test_persistence_after_restart(store):
    ident=add(store)
    from enterprise.store import Store
    again=Store(store.root)
    assert again.documents(ADMIN)[0]['id']==ident

def test_api_login_required(client):
    client.post('/api/logout')
    assert client.get('/api/documents').status_code==401

def upload(client,content=b'# Allowance\n\nRemote work maximum is three days.',**changes):
    meta={'family':'remote_work','document_key':'remote','version':'1','effective_from':'2026-01-01','ingested_at':'2026-01-02T00:00:00Z'}
    meta.update(changes)
    return client.post('/api/documents',files={'file':('policy.md',content,'text/markdown')},data={'metadata':json.dumps(meta)})

def test_api_upload_chat_source_and_delete(client):
    r=upload(client);assert r.status_code==200
    ident=r.json()['id']
    result=client.post('/api/chat',json={'question':'What is the remote work maximum?','as_of':'2026-08-01','agent':'compare'})
    assert result.status_code==200 and len(result.json()['results'])==3
    assert client.get('/api/documents/'+ident+'/source').status_code==200
    assert client.get('/api/documents/'+ident+'/clauses').json()['clauses']
    assert client.get('/api/history').json()
    assert client.delete('/api/documents/'+ident).status_code==200
    assert client.get('/api/documents/'+ident+'/source').status_code==404

def test_api_employee_cannot_manage_or_read_hr(client):
    r=upload(client,audience='HR');ident=r.json()['id']
    client.post('/api/logout');client.post('/api/login',json={'username':'employee','password':'EvoLLM-Employee-2026!'})
    assert client.get('/api/documents').json()==[]
    assert client.get('/api/documents/'+ident+'/source').status_code==404
    assert client.delete('/api/documents/'+ident).status_code==403
    assert upload(client,version='2').status_code==403
    assert client.get('/api/audit').status_code==403

def test_api_invalid_metadata_and_files(client):
    assert upload(client,effective_from='not-a-date').status_code==400
    assert client.post('/api/documents',files={'file':('x.exe',b'123')},data={'metadata':'{}'}).status_code==400
    assert client.post('/api/chat',json={'question':'hi','agent':'invalid'}).status_code==422

def test_api_cross_origin_writes_rejected(client):
    assert client.post('/api/chat',json={'question':'Remote work?'},headers={'Origin':'https://evil.example'}).status_code==403

def test_static_assets_and_security_headers(client):
    for route in ('/','/static/app.js','/static/style.css'):
        r=client.get(route);assert r.status_code==200
        assert r.headers['x-content-type-options']=='nosniff'

def test_login_failure_does_not_set_session(client):
    client.post('/api/logout')
    r=client.post('/api/login',json={'username':'admin','password':'wrong'})
    assert r.status_code==401 and 'evollm_session' not in r.cookies

def test_no_hardcoded_metrics_in_benchmark(client):
    dataset=client.post('/api/evaluations',json={'name':'Empty corpus check','questions':[{'question':'What is the travel limit?', 'acceptable_evidence_sets':[['missing#C1']]}]}).json()
    result=client.post('/api/benchmark',json={'generator':'evidence','dataset_id':dataset['id'],'repeats':1})
    assert result.status_code==200
    assert len(result.json()['runs'])==4
    assert all(r['exact_claim_accuracy'] is None and r['evidence_set_coverage']==0 and r['n']==1 for r in result.json()['runs'])

def test_missing_qwen_returns_actionable_error(client):
    upload(client)
    import sys
    from unittest.mock import patch
    with patch.dict(sys.modules,{'torch':None}):
        r=client.post('/api/chat',json={'question':'What is the remote work maximum?','as_of':'2026-08-01','generator':'qwen'})
    assert r.status_code==503 and 'requirements-ml' in r.json()['detail']


def test_memory_usage_and_archival_preserve_history(store, engine):
    old=add(store)
    add(store,'Remote work maximum is four days.',version='2',effective_from='2026-07-01')
    ask(engine,agent='amo')
    memory=store.memory(ADMIN)
    assert any(r['state']=='archived' and r['document_id']==old for r in memory)
    assert any(r['hits']>0 and r['utility']>0 for r in memory)
    assert old in {c['document_id'] for c in ask(engine,'What was the remote work maximum on 2026-02-01?')['evidence']}

def test_lineage_records_actual_changes(store):
    add(store)
    new=add(store,'Remote work maximum is four days.',version='2',effective_from='2026-07-01')
    lineage=store.lineage(new,ADMIN)
    assert len(lineage['versions'])==2
    assert 'three' in lineage['changes'][0]['before'] and 'four' in lineage['changes'][0]['after']

def test_api_memory_and_lineage_access(client):
    ident=upload(client).json()['id']
    assert client.get('/api/memory').status_code==200
    assert client.get('/api/documents/'+ident+'/lineage').json()['versions']
    client.post('/api/logout');client.post('/api/login',json={'username':'employee','password':'EvoLLM-Employee-2026!'})
    assert client.get('/api/memory').status_code==403

def test_metadata_rank_and_scope_validation(client):
    assert upload(client,authority='highest').status_code==400
    assert upload(client,locations='Pune').status_code==400

def test_parent_metadata_not_indexed_as_policy():
    text='# Remote Policy\nVersion: 1.0\nDocument ID: D101\n\n[C1] Three remote days.\n[C2] Pune only.'
    chunks=dynamic_chunks([{'text':text,'page':None}])
    assert len(chunks)==2 and all('Document ID' not in c['text'] for c in chunks)

def test_history_since_year_keeps_later_versions(store, engine):
    old=add(store,effective_from='2025-01-01')
    new=add(store,'Remote work maximum is four days.',version='2',effective_from='2026-07-01')
    out=ask(engine,'Give the sequence of remote-work allowances since January 2025.')
    assert out['resolution']['mode']=='history'
    assert {c['document_id'] for c in out['evidence']}=={old,new}

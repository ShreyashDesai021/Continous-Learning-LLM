import pytest
from fastapi.testclient import TestClient
from enterprise.api import create_app
from enterprise.store import Store
from enterprise.engine import Engine

ADMIN = {'username': 'admin', 'role': 'admin', 'department': 'HR', 'tenant': 'demo'}
EMPLOYEE = {'username': 'employee', 'role': 'employee', 'department': 'Engineering', 'tenant': 'demo'}

@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    monkeypatch.setenv('EVOLLM_ADMIN_PASSWORD', 'EvoLLM-Admin-2026!')
    monkeypatch.setenv('EVOLLM_EMPLOYEE_PASSWORD', 'EvoLLM-Employee-2026!')
    monkeypatch.setenv('EVOLLM_RETRIEVAL', 'lexical')
    monkeypatch.setenv('EVOLLM_DEMO','0')

@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)

@pytest.fixture
def engine(store):
    return Engine(store, 'lexical')

@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path, seed=False)) as c:
        c.post('/api/login', json={'username': 'admin', 'password': 'EvoLLM-Admin-2026!'})
        yield c

def add(store, text='Remote work maximum is three days per week.', **changes):
    m = {'family': 'remote_work', 'document_key': 'remote', 'version': '1',
         'effective_from': '2026-01-01', 'ingested_at': '2026-01-02T00:00:00Z',
         'rule_key': 'allowance', 'title': 'Remote work', 'authority': 80}
    m.update(changes)
    return store.ingest('policy.txt', text.encode(), m, ADMIN)['id']

def ask(engine, question='What is the remote work maximum?', agent='cake', user=ADMIN, as_of='2026-08-01', **context):
    return engine.run(question, agent, user, as_of, context)

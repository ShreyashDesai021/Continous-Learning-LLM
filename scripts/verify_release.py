"""Self-contained release verifier. Uses temporary data, never your runtime database."""
import json
import hashlib
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from enterprise.api import create_app

def main():
    os.environ['EVOLLM_ADMIN_PASSWORD']='EvoLLM-Admin-2026!'
    os.environ['EVOLLM_EMPLOYEE_PASSWORD']='EvoLLM-Employee-2026!'
    os.environ['EVOLLM_RETRIEVAL']='lexical'
    project=Path(__file__).resolve().parent.parent
    manifest=project/'RELEASE_MANIFEST.json'
    if manifest.exists():
        for item in json.loads(manifest.read_text())['files']:
            assert hashlib.sha256((project/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    with tempfile.TemporaryDirectory() as root:
        with TestClient(create_app(root,seed=True)) as c:
            assert c.get('/api/health').json()['status']=='ok'
            assert c.post('/api/login',json={'username':'admin','password':'EvoLLM-Admin-2026!'}).status_code==200
            assert len(c.get('/api/documents').json())==20
            request={'question':'What is the current weekly remote-work maximum?','agent':'compare','as_of':'2026-08-01'}
            cold=c.post('/api/chat',json=request).json()
            assert len(cold['results'])==3
            assert 'four' in cold['results'][1]['answer']
            warm=c.post('/api/chat',json=request).json()
            assert warm['results'][2]['cache']['context_hit']
            benchmark=c.post('/api/benchmark',json={'generator':'evidence'})
            assert benchmark.status_code==200
            assert len(benchmark.json()['runs'])==4
            assert all(r['n']==25 for r in benchmark.json()['runs'])
            print(json.dumps({'status':'passed','seed_documents':20,'comparison_pipelines':3,
                              'warm_context_hit':True,'benchmark_runs':4,'questions_per_run':25},indent=2))

if __name__=='__main__':main()

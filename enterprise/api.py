"""Local authenticated FastAPI service. Start via python run.py."""
import json
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request, Response, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import Engine
from .store import Store, now, stamp
from .parsing import MAX_BYTES

ROOT = Path(__file__).resolve().parent.parent


class Login(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=200)


class Chat(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    agent: Literal['baseline', 'cake', 'amo', 'compare'] = 'amo'
    as_of: str | None = None
    locations: list[str] = Field(default_factory=list, max_length=10)
    employee_category: str = Field(default='all', max_length=80)
    conditions: list[str] = Field(default_factory=list, max_length=10)
    generator: Literal['evidence', 'qwen'] = 'evidence'
    cache_state: Literal['cold', 'warm'] = 'warm'
    conversation_id: str | None = Field(default=None,max_length=100)


def create_app(data_root=None, seed=False):
    store = Store(data_root or os.getenv('EVOLLM_DATA_DIR', str(ROOT / 'runtime')))
    if seed or os.getenv('EVOLLM_DEMO')=='1':
        store.seed(ROOT)
    engine = Engine(store)
    @asynccontextmanager
    async def lifespan(app):
        yield
        store.flush_usage()
    app = FastAPI(title='EvoLLM Enterprise', version='3.0.0',lifespan=lifespan)
    app.state.store, app.state.engine = store, engine
    attempts = defaultdict(deque)
    rate_lock = threading.Lock()
    # Serialize model calls and cache resets within a process. Compare mode uses
    # one frozen read snapshot, so uploads cannot change its evidence mid-run.
    work_lock = threading.RLock()

    @app.middleware('http')
    async def local_guard(request, call_next):
        origin = request.headers.get('origin')
        host = request.headers.get('host', '')
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and origin:
            from urllib.parse import urlsplit
            if urlsplit(origin).netloc != host:
                return JSONResponse({'detail': 'Cross-origin writes are disabled.'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        return response

    def user(request: Request):
        current = store.user(request.cookies.get('evollm_session', ''))
        if not current:
            raise HTTPException(401, 'Please sign in.')
        return current

    def admin(current=Depends(user)):
        if current['role'] != 'admin':
            raise HTTPException(403, 'Administrator account required.')
        return current

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'version': '3.0.0'}

    @app.post('/api/login')
    def login(body: Login, request: Request, response: Response):
        peer = request.client.host if request.client else 'local'
        with rate_lock:
            attempts[peer] = deque(t for t in attempts[peer] if t > time.monotonic()-60)
            if len(attempts[peer]) >= 10:
                raise HTTPException(429, 'Too many sign-in attempts. Try again in one minute.')
            attempts[peer].append(time.monotonic())
        token = store.login(body.username, body.password)
        if not token:
            raise HTTPException(401, 'Invalid username or password.')
        response.set_cookie('evollm_session', token, httponly=True, samesite='strict', max_age=43200,
                            secure=os.getenv('EVOLLM_SECURE_COOKIE') == '1')
        return store.user(token)

    @app.post('/api/logout')
    def logout(request: Request, response: Response):
        store.logout(request.cookies.get('evollm_session', ''))
        response.delete_cookie('evollm_session')
        return {'ok': True}

    @app.get('/api/me')
    def me(current=Depends(user)):
        return {'user': current, 'retrieval': engine.retriever.backend,
                'llm': engine.generator.model_name, 'revision': store.revision(current['tenant']),
                'workspace_name':os.getenv('EVOLLM_WORKSPACE_NAME','Company workspace'),
                'default_generator':os.getenv('EVOLLM_GENERATOR','evidence')}

    @app.get('/api/users')
    def users(current=Depends(admin)):
        with store.connect() as db:
            return [dict(r) for r in db.execute('SELECT username,role,department FROM users WHERE tenant=? ORDER BY username',(current['tenant'],))]

    @app.post('/api/account/password')
    def change_password(body: dict,request: Request,current=Depends(user)):
        from .store import password_hash
        import secrets,hmac
        old=body.get('current_password','');new=body.get('new_password','')
        if not isinstance(old,str) or not isinstance(new,str) or not 10<=len(new)<=200:
            raise HTTPException(400,'New passwords must contain 10–200 characters.')
        with store.connect() as db:
            account=db.execute('SELECT salt,password FROM users WHERE username=?',(current['username'],)).fetchone()
            if not hmac.compare_digest(account['password'],password_hash(old,account['salt'])):
                raise HTTPException(400,'Current password is incorrect.')
            salt=secrets.token_hex(16)
            db.execute('UPDATE users SET salt=?,password=? WHERE username=?',(salt,password_hash(new,salt),current['username']))
            db.execute('DELETE FROM sessions WHERE username=? AND token<>?',(current['username'],request.cookies.get('evollm_session','')))
        store.audit(current,'password_changed','Own account password updated')
        return {'ok':True}

    @app.post('/api/users')
    def create_user(body: dict,current=Depends(admin)):
        import re
        username=body.get('username','');password=body.get('password','');department=body.get('department','');role=body.get('role','employee')
        if not isinstance(username,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{3,60}',username) or not isinstance(password,str) or not 10<=len(password)<=200:
            raise HTTPException(400,'Use a 3–60 character username and a password of 10–200 characters.')
        if role not in ('admin','employee') or not isinstance(department,str) or not 1<=len(department)<=100:
            raise HTTPException(400,'Choose an account role and a department name.')
        with store.connect() as db:
            if db.execute('SELECT 1 FROM users WHERE username=?',(username,)).fetchone():
                raise HTTPException(409,'Username already exists.')
        store.ensure_user(username,password,role,department,current['tenant'])
        store.audit(current,'account_created',username)
        return {'ok':True}

    @app.post('/api/demo/load')
    def demo(current=Depends(admin)):
        if store.documents(current):
            raise HTTPException(409,'Demo import is available only in an empty workspace.')
        store.seed(ROOT)
        return {'ok':True}

    @app.get('/api/documents')
    def documents(current=Depends(user)):
        return store.documents(current)

    @app.post('/api/documents')
    async def upload(file: UploadFile = File(...), metadata: str = Form(...), current=Depends(admin)):
        try:
            meta = json.loads(metadata)
            if not isinstance(meta, dict):
                raise ValueError('Metadata must be a JSON object.')
            body = await file.read(MAX_BYTES+1)
            result = store.ingest(file.filename or 'upload.txt', body, meta, current)
            engine.context_cache.clear()
            return result
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(400, 'Document could not be parsed. Check the file format and contents.') from exc
        finally:
            await file.close()

    @app.get('/api/documents/{ident}/source')
    def source(ident: str, current=Depends(user)):
        try:
            row = store.source(ident, current)
        except KeyError as exc:
            raise HTTPException(404, 'Document not found.') from exc
        from urllib.parse import quote
        return Response(row['content'], media_type='application/octet-stream',
                        headers={'Content-Disposition': "attachment; filename*=UTF-8''"+quote(row['filename'])})

    @app.get('/api/documents/{ident}/clauses')
    def clauses(ident: str, current=Depends(user)):
        try:
            row = store.source(ident, current)
        except KeyError as exc:
            raise HTTPException(404, 'Document not found.') from exc
        _, corpus = store.corpus_snapshot(current, '9999-12-31')
        return {'metadata': json.loads(row['meta']), 'clauses': [c for c in corpus if c['document_id'] == ident]}

    @app.delete('/api/documents/{ident}')
    def delete(ident: str, current=Depends(admin)):
        try:
            store.remove(ident, current)
        except KeyError as exc:
            raise HTTPException(404, 'Document not found.') from exc
        engine.context_cache.clear()
        return {'ok': True}

    @app.patch('/api/documents/{ident}/metadata')
    def update_metadata(ident: str,body: dict,current=Depends(admin)):
        try:
            store.update_metadata(ident,body,current)
        except KeyError as exc:
            raise HTTPException(404,'Document not found') from exc
        except ValueError as exc:
            raise HTTPException(400,str(exc)) from exc
        engine.context_cache.clear()
        return {'ok':True}

    def execute_chat(body, current,on_token=None):
        try:
            as_of = stamp(body.as_of)
            context = {'locations': body.locations, 'employee_category': body.employee_category, 'conditions': body.conditions}
            try:
                conversation=store.conversation(current,body.conversation_id,title=body.question)
            except KeyError as exc:
                raise HTTPException(404,'Conversation not found') from exc
            previous=store.conversation_runs(current,conversation['id'])
            if previous:
                context['_previous_question']=previous[-1]['results'][0]['question']
            with work_lock:
                if body.agent == 'compare':
                    payload = engine.compare(body.question, current, as_of, context, body.generator, body.cache_state,on_token=on_token)
                else:
                    if body.cache_state == 'cold':
                        engine.clear()
                    payload = {'results': [engine.run(body.question, body.agent, current, as_of, context, body.generator,on_token=on_token)]}
            payload['conversation_id']=conversation['id']
            store.save_history(current, payload,conversation['id'])
            store.audit(current, 'query', f'{body.agent}; revision={payload["results"][0]["revision"]}')
            return payload
        except (ImportError, OSError) as exc:
            if engine.generator.backend=='ollama':
                raise HTTPException(503,'A neural component is unavailable. BGE needs the existing neural dependencies/model; Local Qwen needs Ollama and setup-local-qwen.bat. Details: '+str(exc)[:300]) from exc
            raise HTTPException(503, 'Selected ML engine is unavailable. Install requirements-ml.txt and download '
                                'the models with python scripts/download_models.py. Details: '+str(exc)[:300]) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/chat')
    def chat(body: Chat,current=Depends(user)):
        return execute_chat(body,current)

    @app.post('/api/chat/stream')
    def stream_chat(body: Chat,current=Depends(user)):
        import queue
        events=queue.Queue()
        def work():
            try:
                result=execute_chat(body,current,on_token=lambda text,agent:events.put({'event':'token','text':text,'agent':agent}))
                events.put({'event':'complete','data':result})
            except HTTPException as exc:
                events.put({'event':'error','message':exc.detail})
            except Exception:
                events.put({'event':'error','message':'The local engine could not finish this response. Check the server terminal for model or memory errors.'})
            finally:
                events.put(None)
        worker=threading.Thread(target=work,daemon=True);worker.start()
        def chunks():
            yield json.dumps({'event':'status','message':'Checking accessible documents'})+'\n'
            while True:
                try:
                    event=events.get(timeout=1)
                except queue.Empty:
                    yield json.dumps({'event':'status','message':'Working'})+'\n';continue
                if event is None:
                    break
                yield json.dumps(event)+'\n'
        return StreamingResponse(chunks(),media_type='application/x-ndjson',headers={'X-Accel-Buffering':'no'})

    @app.get('/api/history')
    def history(current=Depends(user)):
        return store.history(current)

    @app.get('/api/conversations/{ident}')
    def conversation(ident: str,current=Depends(user)):
        try:
            return {'runs':store.conversation_runs(current,ident),'conversation_id':ident}
        except KeyError as exc:
            raise HTTPException(404,'Conversation not found') from exc

    @app.get('/api/audit')
    def audit(current=Depends(admin)):
        with store.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM audit WHERE tenant=? ORDER BY id DESC LIMIT 100', (current['tenant'],))]

    @app.get('/api/memory')
    def memory(current=Depends(admin)):
        return {'items':store.memory(current), 'formula':'0.5 × authority + 0.3 × capped usage frequency + 0.2 × use recency',
                'note':'Archival and consolidation are logical tiers. Originals remain available for historical questions; no automatic deletion.'}

    @app.get('/api/documents/{ident}/lineage')
    def lineage(ident: str, current=Depends(user)):
        try:
            return store.lineage(ident,current)
        except KeyError as exc:
            raise HTTPException(404,'Document not found') from exc

    @app.post('/api/cache/clear')
    def clear(current=Depends(admin)):
        with work_lock:
            engine.clear()
        return {'ok': True}

    @app.get('/api/models/status')
    def models_status(current=Depends(user)):
        if engine.generator.backend=='ollama':
            from .ollama_backend import model_status
            try:
                ready=model_status(engine.generator.model_name)
                return {'ready':ready,'message':('Local quantized Qwen model is available. Run a question to validate inference.' if ready else 'Qwen model is missing. Run setup-local-qwen.bat to import the tested model.')}
            except (ValueError,OSError):
                return {'ready':False,'message':'Local Ollama is not reachable. Start Ollama and run setup-local-qwen.bat.'}
        import importlib.util
        from pathlib import Path
        deps=all(importlib.util.find_spec(x) is not None for x in ('torch','transformers','sentence_transformers','qdrant_client'))
        if not deps:
            return {'ready':False,'message':'Neural dependencies are not installed. Run setup-ml.bat; source excerpts remain available.'}
        from huggingface_hub import try_to_load_from_cache
        def local_model(path):
            return Path(path).is_dir() or isinstance(try_to_load_from_cache(path,'config.json'),str)
        configured=local_model(engine.generator.model_name) and local_model(engine.retriever.embedding_model)
        return {'ready':configured,'message':'Model configuration files found locally. Run a Qwen question to verify weights and inference.' if configured else 'Model files are not cached. Run setup-ml.bat to download them.'}

    @app.get('/api/evaluations')
    def evaluations(current=Depends(admin)):
        with store.connect() as db:
            return [{'id':r['id'],'name':r['name'],'count':len(json.loads(r['payload'])['questions'])} for r in db.execute('SELECT * FROM evaluation_sets WHERE tenant=?',(current['tenant'],))]

    @app.post('/api/evaluations')
    def import_evaluation(body: dict,current=Depends(admin)):
        from .evaluation import validate_dataset
        import uuid
        try:
            data=validate_dataset(body)
        except (ValueError,TypeError) as exc:
            raise HTTPException(400,str(exc)) from exc
        ident=uuid.uuid4().hex
        with store.connect() as db:
            db.execute('INSERT INTO evaluation_sets VALUES(?,?,?,?)',(ident,current['tenant'],data['name'],json.dumps(data)))
        return {'id':ident,'name':data['name']}

    @app.post('/api/benchmark')
    def benchmark(body: dict,current=Depends(admin)):
        from .evaluation import evaluate,validate_dataset
        generator=body.get('generator','evidence')
        if generator not in ('evidence','qwen'):
            raise HTTPException(400,'Invalid generator')
        ident=body.get('dataset_id','development')
        if ident=='development':
            qs=[json.loads(x) for x in (ROOT/'data/evaluation/development/questions.jsonl').read_text().splitlines()]
            gold={g['question_id']:g for g in (json.loads(x) for x in (ROOT/'data/ground_truth/development/answers.jsonl').read_text().splitlines())}
            data=validate_dataset({'name':'Original research development set','questions':[
                {'id':q['question_id'],'question':q['question'],'as_of':q['snapshot_at'],'context':q['user_context'],
                 'required_claims':gold[q['question_id']]['required_claims'],
                 'forbidden_claims':gold[q['question_id']]['forbidden_claims'],
                 'acceptable_evidence_sets':gold[q['question_id']]['acceptable_evidence_sets']} for q in qs]})
            expected={cid.split('#')[0] for q in data['questions'] for ev in q['acceptable_evidence_sets'] for cid in ev}
            if not expected & {d['id'] for d in store.documents(current)}:
                raise HTTPException(400,'This dataset labels the optional research documents. Import your own labelled questions, or load the demo in an empty workspace.')
        else:
            with store.connect() as db:
                row=db.execute('SELECT payload FROM evaluation_sets WHERE id=? AND tenant=?',(ident,current['tenant'])).fetchone()
            if not row:
                raise HTTPException(404,'Evaluation dataset not found')
            data=json.loads(row['payload'])
        try:
            with work_lock:
                return evaluate(engine,current,data,generator,body.get('repeats',3))
        except (ImportError,OSError) as exc:
            raise HTTPException(503,'Selected neural engine is unavailable. Run setup-ml.bat. '+str(exc)[:200]) from exc
        except (ValueError,TypeError) as exc:
            raise HTTPException(400,str(exc)) from exc

    app.mount('/static', StaticFiles(directory=ROOT/'enterprise/static'), name='static')

    @app.get('/')
    def home():
        return FileResponse(ROOT/'enterprise/static/index.html')

    return app

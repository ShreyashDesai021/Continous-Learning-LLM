"""Transactional document/version storage and local account isolation."""
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
import uuid
import threading
from contextlib import contextmanager
from datetime import datetime, timezone, date
from pathlib import Path

from .parsing import parse_file, dynamic_chunks


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    """Canonical UTC instant; date-only values mean the end of that UTC day."""
    if not value:
        return now()
    if len(value) == 10:
        date.fromisoformat(value)
        value += 'T23:59:59+00:00'
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc).isoformat()


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 240000).hex()


def can_read(user, meta):
    return (user['tenant'] == meta.get('tenant', 'demo') and
            (user['role'] == 'admin' or meta.get('audience', 'all') == 'all' or
             (meta.get('audience')!='admin' and meta.get('audience') == user['department'])))


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'evollm.sqlite3'
        self._usage_lock=threading.RLock()
        self._pending_usage={}
        with self.connect() as db:
            db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, meta TEXT NOT NULL,
                content BLOB NOT NULL, filename TEXT NOT NULL, fingerprint TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS state(tenant TEXT PRIMARY KEY, revision INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, salt TEXT, password TEXT,
                role TEXT, department TEXT, tenant TEXT);
            CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, username TEXT, expires REAL);
            CREATE TABLE IF NOT EXISTS history(id TEXT PRIMARY KEY, username TEXT, created TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, created TEXT, username TEXT,
                tenant TEXT, action TEXT, detail TEXT);
            CREATE TABLE IF NOT EXISTS memory_usage(tenant TEXT, chunk_id TEXT, hits INTEGER,
                last_used TEXT, PRIMARY KEY(tenant,chunk_id));
            CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, username TEXT, title TEXT, created TEXT);
            CREATE TABLE IF NOT EXISTS evaluation_sets(id TEXT PRIMARY KEY, tenant TEXT, name TEXT, payload TEXT);
            ''')
            if 'conversation_id' not in {r[1] for r in db.execute('PRAGMA table_info(history)')}:
                db.execute('ALTER TABLE history ADD COLUMN conversation_id TEXT')
            for row in db.execute('SELECT id,username,created,payload FROM history WHERE conversation_id IS NULL').fetchall():
                payload=json.loads(row['payload'])
                title=payload.get('results',[{}])[0].get('question','Imported conversation')[:100]
                conversation_id=uuid.uuid4().hex
                db.execute('INSERT INTO conversations VALUES(?,?,?,?)',(conversation_id,row['username'],title,row['created']))
                db.execute('UPDATE history SET conversation_id=? WHERE id=?',(conversation_id,row['id']))
        self.ensure_user('admin', os.getenv('EVOLLM_ADMIN_PASSWORD', 'EvoLLM-Admin-2026!'), 'admin', 'HR')
        self.ensure_user('employee', os.getenv('EVOLLM_EMPLOYEE_PASSWORD', 'EvoLLM-Employee-2026!'), 'employee', 'Engineering')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def ensure_user(self, username, password, role, department, tenant='demo'):
        salt = secrets.token_hex(16)
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO users VALUES(?,?,?,?,?,?)',
                       (username, salt, password_hash(password, salt), role, department, tenant))

    def login(self, username, password):
        with self.connect() as db:
            row = db.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
            if row is None or not hmac.compare_digest(row['password'], password_hash(password, row['salt'])):
                return None
            token = secrets.token_urlsafe(40)
            db.execute('DELETE FROM sessions WHERE expires < ?', (time.time(),))
            db.execute('INSERT INTO sessions VALUES(?,?,?)', (token, username, time.time() + 12 * 3600))
        return token

    def user(self, token):
        with self.connect() as db:
            row = db.execute('SELECT u.username,u.role,u.department,u.tenant FROM users u JOIN sessions s '
                             'ON u.username=s.username WHERE s.token=? AND s.expires>?',
                             (token, time.time())).fetchone()
            return dict(row) if row else None

    def logout(self, token):
        with self.connect() as db:
            db.execute('DELETE FROM sessions WHERE token=?', (token,))

    def revision(self, tenant='demo'):
        with self.connect() as db:
            row = db.execute('SELECT revision FROM state WHERE tenant=?', (tenant,)).fetchone()
            return row[0] if row else 0

    def _bump(self, db, tenant):
        db.execute('INSERT INTO state VALUES(?,1) ON CONFLICT(tenant) DO UPDATE SET revision=revision+1', (tenant,))

    def audit(self, user, action, detail):
        with self.connect() as db:
            db.execute('INSERT INTO audit(created,username,tenant,action,detail) VALUES(?,?,?,?,?)',
                       (now(), user['username'], user['tenant'], action, detail))

    def ingest(self, filename, content, meta, user, *, legacy_id=None):
        if user['role'] != 'admin':
            raise PermissionError('Administrator account required to manage documents.')
        meta = dict(meta)
        for field in ('family', 'document_key', 'version'):
            if not str(meta.get(field, '')).strip():
                raise ValueError(f'{field} is required')
            if not isinstance(meta[field], str) or len(meta[field]) > 200:
                raise ValueError(f'{field} must be text of at most 200 characters')
        if not isinstance(meta.get('authority', 50), (int, float)) or not 0 <= meta.get('authority', 50) <= 100:
            raise ValueError('Authority must be a number between 0 and 100')
        for field in ('locations', 'conditions'):
            if not isinstance(meta.get(field, []), list) or any(not isinstance(s, str) or len(s) > 200 for s in meta.get(field, [])):
                raise ValueError(f'{field} must be a list of short strings')
        if meta.get('status', 'approved') not in ('approved', 'unknown', 'draft', 'withdrawn'):
            raise ValueError('Invalid approval status')
        if not isinstance(meta.get('audience','all'),str) or not 1<=len(meta.get('audience','all'))<=100:
            raise ValueError('Read access must be all, admin, or a department name of at most 100 characters.')
        for field in ('effective_from', 'effective_to', 'published_at'):
            if meta.get(field):
                date.fromisoformat(meta[field])
        if meta.get('effective_from') and meta.get('effective_to') and meta['effective_to'] <= meta['effective_from']:
            raise ValueError('Effective-to must be after effective-from (exclusive).')
        meta['ingested_at'] = stamp(meta.get('ingested_at'))
        meta['tenant'] = user['tenant']
        meta.setdefault('audience', 'all')
        meta.setdefault('status', 'approved')
        meta.setdefault('authority', 50)
        meta.setdefault('issuer', 'Document administrator')
        meta.setdefault('locations', [])
        meta.setdefault('conditions', [])
        meta.setdefault('employee_category', 'all')
        meta.setdefault('title', filename)
        meta.setdefault('document_type','policy')
        if meta['document_type'] not in ('policy','reference'):
            raise ValueError('Document type must be policy or reference')
        if not isinstance(meta.get('rule_keys',{}),dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in meta.get('rule_keys',{}).items()):
            raise ValueError('Rule mappings must be a JSON object of clause IDs to rule names')
        parsed = parse_file(filename, content)
        pieces = dynamic_chunks(parsed)
        fingerprint = hashlib.sha256(content).hexdigest()
        ident = legacy_id or uuid.uuid4().hex[:16]
        with self.connect() as db:
            for row in db.execute('SELECT id,meta,fingerprint FROM documents'):
                old = json.loads(row['meta'])
                if old['tenant'] != user['tenant']:
                    continue
                if old['document_key'] == meta['document_key'] and old['version'] == meta['version']:
                    # Exact retry is idempotent; edits need a new version identifier.
                    comparable_old = {k: v for k, v in old.items() if k != 'ingested_at'}
                    comparable_new = {k: v for k, v in meta.items() if k != 'ingested_at'}
                    if row['fingerprint'] == fingerprint and comparable_old == comparable_new:
                        return {'id': row['id'], 'duplicate': True, 'chunks': 0}
                    raise ValueError('This document key/version already exists. Use a new version.')
            db.execute('INSERT INTO documents VALUES(?,?,?,?,?)',
                       (ident, json.dumps(meta), content, Path(filename).name, fingerprint))
            for p in pieces:
                chunk_id = f"{ident}#{p['clause'] or 'S'+str(p['ordinal'])}"
                if p['part'] > 1:
                    chunk_id += f".p{p['part']}"
                # Several unnumbered paragraphs can belong to one heading; unique source IDs.
                if db.execute('SELECT 1 FROM chunks WHERE id=?', (chunk_id,)).fetchone():
                    chunk_id += f".{p['ordinal']}"
                payload = {**p, 'id': chunk_id, 'document_id': ident, 'filename': Path(filename).name,
                           'meta': meta, 'rule_key': meta.get('rule_key') or p['rule_key']}
                db.execute('INSERT INTO chunks VALUES(?,?,?)', (chunk_id, ident, json.dumps(payload)))
            self._bump(db, user['tenant'])
            db.execute('INSERT INTO audit(created,username,tenant,action,detail) VALUES(?,?,?,?,?)',
                       (now(), user['username'], user['tenant'], 'upload', ident))
        return {'id': ident, 'duplicate': False, 'chunks': len(pieces)}

    def documents(self, user):
        with self.connect() as db:
            result = []
            for row in db.execute('SELECT id,meta,filename FROM documents ORDER BY rowid DESC'):
                m = json.loads(row['meta'])
                if can_read(user, m):
                    count = db.execute('SELECT COUNT(*) FROM chunks WHERE document_id=?', (row['id'],)).fetchone()[0]
                    result.append({'id': row['id'], 'filename': row['filename'], 'chunks': count, **m})
            return result

    def corpus_snapshot(self, user, snapshot):
        # Revision and corpus are read within the same SQLite read transaction.
        cutoff = stamp(snapshot)
        with self.connect() as db:
            db.execute('BEGIN')
            rev = db.execute('SELECT revision FROM state WHERE tenant=?', (user['tenant'],)).fetchone()
            chunks = [json.loads(r[0]) for r in db.execute('SELECT payload FROM chunks')]
            visible = [c for c in chunks if can_read(user, c['meta']) and stamp(c['meta']['ingested_at']) <= cutoff]
            return (rev[0] if rev else 0), visible

    def source(self, ident, user):
        with self.connect() as db:
            row = db.execute('SELECT * FROM documents WHERE id=?', (ident,)).fetchone()
            if not row or not can_read(user, json.loads(row['meta'])):
                raise KeyError('Document not found')
            return dict(row)

    def remove(self, ident, user):
        if user['role'] != 'admin':
            raise PermissionError('Administrator account required.')
        self.source(ident, user)
        with self.connect() as db:
            db.execute('DELETE FROM chunks WHERE document_id=?', (ident,))
            db.execute('DELETE FROM documents WHERE id=?', (ident,))
            self._bump(db, user['tenant'])
            db.execute('INSERT INTO audit(created,username,tenant,action,detail) VALUES(?,?,?,?,?)',
                       (now(), user['username'], user['tenant'], 'delete', ident))

    def update_metadata(self,ident,changes,user):
        if user['role']!='admin':
            raise PermissionError('Administrator account required.')
        source=self.source(ident,user)
        metadata=json.loads(source['meta'])
        allowed={'status','document_type','authority','rule_keys'}
        if set(changes)-allowed:
            raise ValueError('Only approval, source type, authority and rule mappings can be edited here.')
        metadata.update(changes)
        if metadata.get('status') not in ('approved','draft','unknown','withdrawn') or metadata.get('document_type','policy') not in ('policy','reference'):
            raise ValueError('Invalid approval or source type.')
        if not isinstance(metadata.get('authority'),(int,float)) or not 0<=metadata['authority']<=100:
            raise ValueError('Authority must be between 0 and 100.')
        mappings=metadata.get('rule_keys',{})
        if not isinstance(mappings,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in mappings.items()):
            raise ValueError('Rule mappings must map clause IDs to rule names.')
        with self.connect() as db:
            db.execute('UPDATE documents SET meta=? WHERE id=?',(json.dumps(metadata),ident))
            for row in db.execute('SELECT id,payload FROM chunks WHERE document_id=?',(ident,)).fetchall():
                payload=json.loads(row['payload']);payload['meta']=metadata
                db.execute('UPDATE chunks SET payload=? WHERE id=?',(json.dumps(payload),row['id']))
            self._bump(db,user['tenant'])
            db.execute('INSERT INTO audit(created,username,tenant,action,detail) VALUES(?,?,?,?,?)',(now(),user['username'],user['tenant'],'metadata_update',ident))

    def conversation(self, user, ident=None, title=None):
        with self.connect() as db:
            if ident:
                row=db.execute('SELECT * FROM conversations WHERE id=? AND username=?',(ident,user['username'])).fetchone()
                if not row:
                    raise KeyError('Conversation not found')
                return dict(row)
            ident=uuid.uuid4().hex
            db.execute('INSERT INTO conversations VALUES(?,?,?,?)',(ident,user['username'],(title or 'New conversation')[:100],now()))
            return {'id':ident,'username':user['username'],'title':title}

    def conversation_runs(self,user,ident):
        self.conversation(user,ident)
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT payload FROM history WHERE username=? AND conversation_id=? ORDER BY created',(user['username'],ident))]

    def save_history(self, user, payload, conversation_id=None):
        with self.connect() as db:
            db.execute('INSERT INTO history(id,username,created,payload,conversation_id) VALUES(?,?,?,?,?)',
                       (uuid.uuid4().hex, user['username'], now(), json.dumps(payload),conversation_id))

    def history(self, user):
        with self.connect() as db:
            return [dict(r) | {'payload': json.loads(r['payload'])} for r in db.execute(
                'SELECT id,created,payload,conversation_id FROM history WHERE username=? ORDER BY created DESC LIMIT 30',
                (user['username'],))]

    def note_usage(self, user, evidence):
        with self._usage_lock:
            for c in evidence:
                key=(user['tenant'],c['id'])
                hits,_=self._pending_usage.get(key,(0,None))
                self._pending_usage[key]=(hits+1,now())

    def flush_usage(self):
        with self._usage_lock:
            if not self._pending_usage:
                return
            entries=self._pending_usage
            self._pending_usage={}
        with self.connect() as db:
            for (tenant,ident),(hits,used) in entries.items():
                db.execute('INSERT INTO memory_usage VALUES(?,?,?,?) ON CONFLICT(tenant,chunk_id) '
                           'DO UPDATE SET hits=hits+excluded.hits,last_used=excluded.last_used',
                           (tenant,ident,hits,used))

    def memory(self, user):
        """Explainable utility tiers. Archival is logical: historical data is retained."""
        self.flush_usage()
        _, chunks = self.corpus_snapshot(user, now())
        with self.connect() as db:
            usage = {r['chunk_id']: dict(r) for r in db.execute('SELECT * FROM memory_usage WHERE tenant=?', (user['tenant'],))}
        newest = {}
        for c in chunks:
            m = c['meta']
            if m.get('status') == 'approved' and (m.get('effective_from') or '') <= now()[:10]:
                key = (m['document_key'], c['rule_key'])
                newest[key] = max(newest.get(key, ''), m.get('effective_from') or '')
        result, canonical = [], {}
        for c in chunks:
            m = c['meta']; u = usage.get(c['id'], {})
            norm = ' '.join(c['text'].lower().split())
            identity = (m['family'], tuple(m.get('locations', [])), tuple(m.get('conditions', [])),
                        m.get('effective_from'), m.get('employee_category'), norm)
            duplicate_of = canonical.get(identity)
            canonical.setdefault(identity, c['id'])
            expired = m.get('effective_to') and m['effective_to'] <= now()[:10]
            superseded = (m.get('effective_from') or '') < newest.get((m['document_key'], c['rule_key']), '')
            future = (m.get('effective_from') or '') > now()[:10]
            state = 'archived' if expired or superseded or m['status'] == 'withdrawn' else ('draft' if m['status']=='draft' else ('review' if m['status']=='unknown' else ('consolidated' if duplicate_of else ('scheduled' if future else 'active'))))
            recency = 0.0
            if u.get('last_used'):
                age = (datetime.now(timezone.utc)-datetime.fromisoformat(u['last_used'])).total_seconds()/86400
                recency = 1/(1+max(age,0))
            frequency = min(u.get('hits',0)/10,1)
            utility = round(.5*m.get('authority',50)/100 + .3*frequency + .2*recency, 3)
            result.append({'chunk_id':c['id'],'document_id':c['document_id'],'title':m['title'],
                           'version':m['version'],'family':m['family'],'state':state,'utility':utility,
                           'hits':u.get('hits',0),'last_used':u.get('last_used'),'duplicate_of':duplicate_of})
        return sorted(result,key=lambda r:-r['utility'])

    def lineage(self, ident, user):
        import difflib
        source = self.source(ident, user)
        meta = json.loads(source['meta'])
        docs = [d for d in self.documents(user) if d['document_key'] == meta['document_key']]
        docs.sort(key=lambda d:(d.get('effective_from') or '',d['version']))
        _, corpus = self.corpus_snapshot(user, '9999-12-31')
        changes = []
        previous = None
        for d in docs:
            current = {}
            for c in sorted((c for c in corpus if c['document_id']==d['id']),key=lambda c:c['ordinal']):
                key=c['rule_key']
                current[key] = (current.get(key,'')+'\n\n'+c['text']).strip()
            if previous:
                old_doc, old = previous
                for key in sorted(set(old)|set(current)):
                    if old.get(key) != current.get(key):
                        changes.append({'from':old_doc['id'],'to':d['id'],'rule':key,
                                        'before':old.get(key),'after':current.get(key),
                                        'diff':'\n'.join(difflib.unified_diff((old.get(key) or '').splitlines(),
                                                (current.get(key) or '').splitlines(),lineterm=''))})
            previous = (d,current)
        return {'document_key':meta['document_key'],'versions':docs,'changes':changes,
                'note':'Lineage is explicitly assigned by the administrator. Changes do not imply automatic authority.'}

    def seed(self, project_root):
        """Import the supplied controlled corpus, retaining source IDs and timestamps."""
        admin = {'username': 'admin', 'role': 'admin', 'tenant': 'demo', 'department': 'HR'}
        if self.documents(admin):
            return
        base = Path(project_root) / 'data/inference/development'
        for line in (base / 'metadata.jsonl').read_text(encoding='utf-8').splitlines():
            m = json.loads(line)
            fam = m['policy_family']
            source_class=m['authority']['source_class']
            reference=(source_class in ('guide','knowledge_base') or
                       any(w in (m.get('version') or '').lower() for w in ('accessible','reissue','reminder')))
            lineage_payload=[fam,m['authority']['issuer'],m['scope'],source_class if reference else 'policy']
            lineage='import_'+hashlib.sha256(json.dumps(lineage_payload,sort_keys=True).encode()).hexdigest()[:14]
            meta = {'title': m['title'], 'family': fam, 'document_key': lineage,
                    'document_type':'reference' if reference else 'policy',
                    'version': m.get('version') or m['document_id'],
                    'effective_from': m.get('effective_from'), 'effective_to': m.get('effective_to'),
                    'published_at': m.get('publication_date'), 'ingested_at': m['ingested_at'],
                    'status': {'approved': 'approved', 'unknown': 'unknown'}.get(m['authority']['approval_status'], 'draft'),
                    'authority': 80 if m['authority']['source_class'] == 'corporate_policy' else 50,
                    'issuer': m['authority']['issuer'], 'locations': m['scope']['locations'],
                    'employee_category': m['scope']['employee_category'], 'conditions': m['scope']['conditions']}
            # Reuploads can retain a version number; keep distinct identities to show duplicates.
            if any(d['document_key'] == lineage and d['version'] == meta['version'] for d in self.documents(admin)):
                meta['version'] += '-' + m['document_id']
            self.ingest(m['document_id']+'.txt', (base/'documents'/f"{m['document_id']}.txt").read_bytes(),
                        meta, admin, legacy_id=m['document_id'])

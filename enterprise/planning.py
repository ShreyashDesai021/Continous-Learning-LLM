"""Corpus-independent query intent and scope planning.

Rules describe language patterns, never policy families, document IDs or answers.
"""
import re
from datetime import date
from .store import stamp

STOP = set('the a an is are was were what which how many does do did for to of in on at and or can i we it be this that today current currently policy rule please tell me allowed have has from after before applies apply all its their our per about would should than each'.split())


def tokens(text):
    result = []
    for word in re.findall(r'[a-z0-9]+', text.lower()):
        if word in STOP:
            continue
        # Generic inflection normalization, not a domain synonym dictionary.
        if len(word)>5 and word.endswith('ies'):
            word=word[:-3]+'y'
        elif len(word)>5 and word.endswith('ing'):
            word=word[:-3]
        elif len(word)>4 and word.endswith('s') and not word.endswith(('ss','us')):
            word=word[:-1]
        result.append(word)
    return result


def normalized(text):
    return ' '.join(re.findall(r'[a-z0-9]+',text.lower()))


def explicit_dates(question):
    """Return all explicit dates in text order; never infer an unstated interval."""
    months={m:i for i,m in enumerate('january february march april may june july august september october november december'.split(),1)}
    found=[]
    for m in re.finditer(r'\b(\d{4}-\d{2}-\d{2})\b',question):
        found.append((m.start(),date.fromisoformat(m[1]).isoformat()))
    for m in re.finditer(r'\b(\d{1,2})\s+('+'|'.join(months)+r')\s+(\d{4})\b',question.lower()):
        found.append((m.start(),date(int(m[3]),months[m[2]],int(m[1])).isoformat()))
    for m in re.finditer(r'\b('+'|'.join(months)+r')\s+(\d{1,2}),?\s+(\d{4})\b',question.lower()):
        found.append((m.start(),date(int(m[3]),months[m[1]],int(m[2])).isoformat()))
    return list(dict.fromkeys(value for _,value in sorted(found)))


def target_date(question, default):
    dates=explicit_dates(question)
    if dates:
        return dates[0]
    year=re.search(r'\b((?:19|20)\d{2})\b',question)
    return f'{year[1]}-12-31' if year else stamp(default)[:10]


def mode_for(question):
    q=question.lower()
    if any(w in q for w in ('history','timeline','sequence','throughout')):
        return 'history'
    if any(w in q for w in ('compare','contradict','difference','same','versus',' vs ')):
        return 'comparison'
    if any(w in q for w in ('what changed','what stayed','did ','does ','why ','explain','wording','clarif')):
        return 'explanation'
    if any(w in q for w in ('announced','pending','future','upcoming','starting','scheduled')):
        return 'announcement'
    return 'governing'


def query_plan(question, as_of):
    mode=mode_for(question)
    dates=explicit_dates(question)
    if len(dates)>1 and mode!='history':
        mode='comparison'
    year=re.search(r'\b((?:19|20)\d{2})\b',question)
    history_end=(f'{year[1]}-12-31' if year and 'throughout' in question.lower() else stamp(as_of)[:10])
    return {'mode':mode,'target_date':history_end if mode=='history' else target_date(question,as_of),
            'target_dates':dates if dates else [history_end if mode=='history' else target_date(question,as_of)],
            'range_start':f'{year[1]}-01-01' if year and mode=='history' else None,
            'include_supporting':mode in ('explanation','comparison'),
            'include_future':mode=='announcement'}


def rule(chunk):
    return chunk['meta'].get('rule_keys',{}).get(chunk.get('clause'),chunk['rule_key'])


def scoped(chunk, question, context):
    meta=chunk['meta'];qt=set(tokens(question))
    locations={s.lower() for s in context.get('locations',[])}
    mentioned={s.lower() for s in context.get('_known_locations',[]) if re.search(r'\b'+re.escape(s.lower())+r'\b',question.lower())}
    if mentioned:
        locations=mentioned
    doc_locations={s.lower() for s in meta.get('locations',[])}
    if locations and doc_locations and not locations & doc_locations:
        return False
    category=context.get('employee_category','all')
    if category!='all' and meta.get('employee_category','all') not in ('all',category):
        return False
    conditions=meta.get('conditions',[])
    requested=context.get('conditions',[])
    known=context.get('_known_conditions',[])
    # Derive distinguishing environment terms from the corpus, not fixed Dev/Prod names.
    term_sets=[set(tokens(s)) for s in known]
    common=set.intersection(*term_sets) if len(term_sets)>1 else set()
    discriminating=set.union(*term_sets)-common if term_sets else set()
    mentioned_terms=qt & discriminating
    if mentioned_terms:
        return not conditions or any(set(tokens(s)) & mentioned_terms for s in conditions)
    if requested and conditions:
        return any(set(tokens(r)) & set(tokens(c)) for r in requested for c in conditions)
    return True

"""User-supplied labelled evaluation. Never feeds expected answers to the engine."""
import math
import statistics
from .store import stamp


def validate_dataset(body):
    if not isinstance(body,dict) or not isinstance(body.get('questions'),list) or not 1<=len(body['questions'])<=100:
        raise ValueError('Provide a JSON object with 1–100 questions.')
    name=body.get('name','Imported evaluation')
    if not isinstance(name,str) or len(name)>150:
        raise ValueError('Dataset name must be short text.')
    questions=[]
    for i,q in enumerate(body['questions']):
        if not isinstance(q,dict) or not isinstance(q.get('question'),str) or not 3<=len(q['question'])<=2000:
            raise ValueError(f'Question {i+1} requires 3–2000 characters.')
        sets=q.get('acceptable_evidence_sets',[])
        if not isinstance(sets,list) or any(not isinstance(s,list) or not s or any(not isinstance(x,str) for x in s) for s in sets):
            raise ValueError('Acceptable evidence sets must contain nonempty lists of source IDs.')
        for field in ('required_claims','forbidden_claims'):
            if not isinstance(q.get(field,[]),list) or any(not isinstance(x,str) for x in q.get(field,[])):
                raise ValueError(f'{field} must be a list of strings.')
        context=q.get('context',{})
        if not isinstance(context,dict) or any(not isinstance(context.get(k,[]),list) for k in ('locations','conditions')):
            raise ValueError('Context must contain location and condition lists.')
        questions.append({**q,'id':str(q.get('id',i+1)),'as_of':stamp(q.get('as_of')),
                          'context':context,'acceptable_evidence_sets':sets,
                          'required_claims':q.get('required_claims',[]),'forbidden_claims':q.get('forbidden_claims',[])})
    return {'name':name,'questions':questions}


def evidence_metrics(ids,sets):
    if not sets:
        return None,None,None
    acceptable=set().union(*map(set,sets))
    recall=max(len(ids & set(s))/len(s) for s in sets)
    precision=len(ids & acceptable)/len(ids) if ids else 0.0
    full=any(set(s)<=ids for s in sets)
    return recall,precision,full


def mean_present(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def evaluate(engine,user,dataset,generator='evidence',repeats=3):
    repeats=int(repeats)
    if not 1<=repeats<=5:
        raise ValueError('Timing repetitions must be between 1 and 5.')
    runs=[];cold_contexts={}
    for agent in ('baseline','cake','amo'):
        engine.clear()
        for pass_name in (('cold','warm') if agent=='amo' else ('cold',)):
            if pass_name=='warm':
                # Prime every question explicitly. Priming is excluded from warm timing.
                for q in dataset['questions']:
                    engine.run(q['question'],agent,user,q['as_of'],q['context'],generator)
            rows=[];times=[];hits=[]
            for q in dataset['questions']:
                samples=[];out=None
                for _ in range(repeats):
                    if pass_name=='cold':
                        engine.clear()
                    out=engine.run(q['question'],agent,user,q['as_of'],q['context'],generator)
                    samples.append(out['timing']['total_ms']);hits.append(out['cache']['context_hit'])
                times.extend(samples)
                ids={c['id'] for c in out['evidence']}
                recall,precision,full=evidence_metrics(ids,q['acceptable_evidence_sets'])
                raw_recall,_,_=evidence_metrics({c['id'] for c in out['retrieved']},q['acceptable_evidence_sets'])
                # Excerpted drafts/retired rules are evidence, not generated claims.
                exact=None
                if generator=='qwen' and q['required_claims']:
                    text=out['answer'].lower()
                    exact=all(x.lower() in text for x in q['required_claims']) and not any(x.lower() in text for x in q['forbidden_claims'])
                fingerprint=tuple(sorted(ids))
                if agent=='amo' and pass_name=='cold':
                    cold_contexts[q['id']]=fingerprint
                rows.append({'question_id':q['id'],'question':q['question'],'exact_claim_match':exact,
                             'evidence_set_covered':full,'precision':precision,'recall':recall,'raw_recall':raw_recall,
                             'latency_samples_ms':samples,'answer':out['answer'],'citations':out['citations'],
                             'selected_evidence_ids':sorted(ids),'same_evidence_as_cold':fingerprint==cold_contexts.get(q['id']) if pass_name=='warm' else None,
                             'timing':out['timing'],'cache':out['cache']})
            runs.append({'agent':agent,'pass':pass_name,'n':len(rows),'generator':generator,'repeats':repeats,
                         'exact_claim_accuracy':mean_present([r['exact_claim_match'] for r in rows]),
                         'evidence_set_coverage':mean_present([r['evidence_set_covered'] for r in rows]),
                         'mean_precision':mean_present([r['precision'] for r in rows]),
                         'mean_recall':mean_present([r['recall'] for r in rows]),
                         'raw_recall':mean_present([r['raw_recall'] for r in rows]),
                         'mean_latency_ms':statistics.mean(times),'median_latency_ms':statistics.median(times),
                         'p95_latency_ms':sorted(times)[max(0,math.ceil(.95*len(times))-1)],
                         'context_cache_hit_rate':statistics.mean(hits),'rows':rows})
    return {'runs':runs,'dataset':dataset['name'],
            'note':'Measured engine times, including generation. Cold clears query/context caches; model/index initialization may affect early runs. '
                   'Warm timings exclude priming. Claim phrase checks apply only to generated answers with labels; — means not measured. '
                   'Evidence coverage is separate from factual answer accuracy.'}

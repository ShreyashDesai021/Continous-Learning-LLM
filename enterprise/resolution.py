"""Question-aware CAKE: normative decisions and explanatory evidence are distinct."""
from collections import defaultdict
from .planning import query_plan, scoped, rule, tokens, normalized


def _scope(c):
    m=c['meta']
    return tuple(sorted(m.get('locations',[]))),tuple(sorted(m.get('conditions',[]))),m.get('employee_category','all')


def _same_predicate(a,b):
    if a['meta']['family']!=b['meta']['family'] or _scope(a)!=_scope(b):
        return False
    explicit_a=a['meta'].get('rule_key') or a['meta'].get('rule_keys',{}).get(a.get('clause'))
    explicit_b=b['meta'].get('rule_key') or b['meta'].get('rule_keys',{}).get(b.get('clause'))
    if explicit_a and explicit_b:
        return explicit_a==explicit_b
    # Repeated C1 labels in independent documents do not establish equivalence.
    ta=set(tokens(a['text']));tb=set(tokens(b['text']))
    union=ta|tb
    return bool(union) and len(ta&tb)/len(union)>=.62


def resolve(candidates, question, as_of, context, *, _target_date=None):
    plan=query_plan(question,as_of)
    if _target_date is None and plan['mode']=='comparison' and len(plan['target_dates'])>1:
        # Resolve each point separately. Never compare old/new versions as if
        # both had to govern the first date in the question.
        combined={};excluded=[];future={};conflicts=[]
        priority={'supporting':0,'announced':1,'historical':2,'governing':3,'contested':4}
        for point in plan['target_dates']:
            leg=resolve(candidates,question,as_of,context,_target_date=point)
            for c in leg['selected']:
                role=c['evidence_role']
                if c['id'] not in combined:
                    combined[c['id']]=dict(c,date_roles={},applicable_dates=[])
                item=combined[c['id']]
                item['date_roles'][point]=role
                if role in ('governing','historical','contested'):
                    item['applicable_dates'].append(point)
                if priority.get(role,0)>priority.get(item['evidence_role'],0):
                    item['evidence_role']=role
                item['decision_reason']='; '.join(f'{d}: {r}' for d,r in item['date_roles'].items())
            excluded.extend(dict(x,target_date=point,reason=point+': '+x['reason']) for x in leg['excluded'])
            future.update({c['id']:c for c in leg['future']})
            conflicts.extend(dict(x,target_date=point) for x in leg['conflicts'])
        selected=list(combined.values())
        return {'selected':selected,'excluded':excluded,'future':list(future.values()),
                'supporting':[c for c in selected if c['evidence_role']=='supporting'],
                'conflicts':conflicts,'target_date':plan['target_date'],'target_dates':plan['target_dates'],
                'mode':plan['mode'],'plan':plan}
    target=_target_date or plan['target_date']
    selected=[];excluded=[];future=[];supporting=[];conflicts=[];lineages=defaultdict(list)
    context=dict(context,_known_locations=sorted({x for c in candidates for x in c['meta'].get('locations',[])}),
                 _known_conditions=sorted({x for c in candidates for x in c['meta'].get('conditions',[])}))
    def reject(c,reason,explain=False):
        excluded.append({'chunk':c,'reason':reason})
        if explain and plan['include_supporting']:
            supporting.append(dict(c,evidence_role='supporting',decision_reason=reason))
    for c in candidates:
        m=c['meta']
        if not scoped(c,question,context):
            reject(c,'Outside requested scope');continue
        if m.get('status') in ('draft','withdrawn'):
            reject(c,'Draft or withdrawn; cannot establish the governing rule',True);continue
        if m.get('document_type','policy')=='reference':
            reject(c,'Reference or clarification; not an independent policy override',True);continue
        if m.get('effective_from') and m['effective_from']>target:
            future.append(dict(c,evidence_role='announced'))
            reject(c,'Announced but not yet effective',plan['include_supporting']);continue
        if m.get('effective_to') and m['effective_to']<=target and plan['mode']!='history':
            reject(c,'Expired at the requested date',True);continue
        if plan['mode']=='history' and plan['range_start'] and m.get('effective_to') and m['effective_to']<=plan['range_start']:
            reject(c,'Outside requested historical interval');continue
        lineages[(m['family'],m['document_key'],_scope(c),rule(c))].append(c)
    winners=[]
    for key,versions in lineages.items():
        if plan['mode']=='history':
            selected.extend(dict(c,evidence_role='historical') for c in sorted(versions,key=lambda c:(c['meta'].get('effective_from') or '',c['ordinal'])))
            continue
        latest=max(c['meta'].get('effective_from') or '' for c in versions)
        for c in versions:
            if (c['meta'].get('effective_from') or '')==latest:
                winners.append(c)
            else:
                reject(c,'Superseded within the same document lineage and rule',True)
    # Assemble parts of each parent rule before aligning independent sources.
    parents=defaultdict(list)
    for c in winners:
        parents[(c['document_id'],rule(c),_scope(c))].append(c)
    parent_groups=[]
    for parts in parents.values():
        body=dict(parts[0],text=' '.join(c['text'] for c in sorted(parts,key=lambda c:c['ordinal'])))
        match=next((g for g in parent_groups if _same_predicate(g[0]['body'],body)),None)
        entry={'body':body,'parts':parts}
        if match is None:
            parent_groups.append([entry])
        else:
            match.append(entry)
    for group in parent_groups:
        highest=max(x['body']['meta'].get('authority',50) for x in group)
        chosen=[];seen={}
        for entry in group:
            body=entry['body'];parts=entry['parts']
            if body['meta'].get('authority',50)<highest:
                for c in parts:reject(c,'Lower explicit authority for this same rule',True)
                continue
            identity=normalized(body['text'])
            if identity in seen:
                for c in parts:reject(c,'Exact duplicate of canonical evidence',True)
                continue
            seen[identity]=True;chosen.extend(parts)
        contested=len(seen)>1 or any(c['meta'].get('status')=='unknown' for c in chosen)
        if contested:
            conflicts.append({'rule':rule(chosen[0]),'clauses':[c['id'] for c in chosen],
                              'reason':'Competing equal-authority statements or unverified approval; review required'})
        selected.extend(dict(c,evidence_role='contested' if contested else 'governing') for c in chosen)
    selected.extend(supporting)
    if plan['include_future']:
        selected.extend(future)
    # Keep one occurrence per source; evidence roles remain explicit.
    selected=[dict(c,applicable_dates=[target] if c['evidence_role'] in ('governing','historical','contested') else [],
                   date_roles={target:c['evidence_role']}) for c in {c['id']:c for c in selected}.values()]
    return {'selected':selected,'excluded':excluded,'future':future,'supporting':supporting,
            'conflicts':conflicts,'target_date':target,'target_dates':[target],'mode':plan['mode'],'plan':plan}

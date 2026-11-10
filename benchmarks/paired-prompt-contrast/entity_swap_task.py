"""A balanced assignment x query x presentation-order task, with pair-disjoint splits."""
from collections import Counter
import hashlib
import itertools
import math

from shared_prefix_study import require

ENTITIES = (32,36,33,39,35,38,34,37)
ON, IDLE = 2,16
ANSWERS = (ON,IDLE)
EPS, EFFECT = 5e-6,.01
STEPS = 64
STATES = ("initial","subset_rank")
ARMS = ("retention","direct_ce","direct_matched","direct_rotated","safe_ce","safe_matched","safe_rotated")


def render(entities, order, idle, query):
    l,r = order,1-order
    cue = lambda j: 26 if j==idle else 24
    return [60,entities[l],cue(l),46,entities[r],cue(r),47,entities[query],23]


def identifier(prompt):
    return hashlib.sha256(bytes(prompt)).hexdigest()


def deck(split):
    require(split in ("train","test"),"unknown split")
    offset = 1 if split=="train" else 3
    rows=[]
    for edge,a in enumerate(ENTITIES):
        es=[a,ENTITIES[(edge+offset)%8]]
        for order,idle,query in itertools.product(range(2),repeat=3):
            prompt=render(es,order,idle,query)
            rows.append(dict(id=identifier(prompt),prompt=prompt,entities=es,edge=edge,order=order,idle=idle,
                             query=query,gold=IDLE if query==idle else ON))
    return sorted(rows,key=lambda r:r["id"])


def queries(rows):
    return [{k:r[k] for k in ("id","prompt")} for r in rows]


def key(rows):
    return [{k:v for k,v in r.items() if k!="prompt"} for r in rows]


def design(train,test):
    edges=[]
    for split,rows in (("train",train),("test",test)):
        require(rows==deck(split) and len({r["id"] for r in rows})==64,"unexpected or duplicate cells")
        pairs={tuple(sorted(r["entities"])) for r in rows}
        require(len(pairs)==8 and Counter(x for pair in pairs for x in pair)==Counter({e:2 for e in ENTITIES}),
                "unbalanced entity coverage")
        require(Counter((r["entities"][r["query"]],r["order"]^r["query"],r["gold"]) for r in rows)==
                Counter({(e,i,k):2 for e in ENTITIES for i in (0,1) for k in ANSWERS}),"unbalanced position or label")
        edges.append(pairs)
    require(edges[0].isdisjoint(edges[1]) and {r["id"] for r in train}.isdisjoint(r["id"] for r in test),"pair or prompt leakage")


def batch(rows,t):
    query_ids={ENTITIES[t%4],ENTITIES[t%4+4]}
    chosen=[r for r in rows if r["entities"][r["query"]] in query_ids]
    require(len(chosen)==16 and Counter(r["gold"] for r in chosen)==Counter({ON:8,IDLE:8}),"invalid minibatch")
    return chosen


def negatives(rows):
    """Keep queried identity, position and both cue positions. Rotation changes only the distractor."""
    result={"matched":[],"rotated":[]}
    for i,r in enumerate(rows):
        q=r["entities"][r["query"]];pos=r["order"]^r["query"]
        options=[(j,x) for j,x in enumerate(rows) if x["entities"][x["query"]]==q and x["order"]^x["query"]==pos and x["gold"]!=r["gold"]]
        require(len(options)==2,"expected two negative partners for the queried identity")
        (a,x),=[v for v in options if v[1]["edge"]==r["edge"]]
        (b,y),=[v for v in options if v[1]["edge"]!=r["edge"]]
        require(a!=i and sorted(x["prompt"])==sorted(r["prompt"]) and
                [k for k,(u,v) in enumerate(zip(x["prompt"],y["prompt"],strict=True)) if u!=v]==[4 if pos==0 else 1],
                "unmatched negative control")
        result["matched"].append(a);result["rotated"].append(b)
    for ns in result.values():
        require(sorted(ns)==list(range(len(rows))) and all(ns[j]==i for i,j in enumerate(ns)),"non-bijective negative reuse")
    return result


def mean(values):
    return math.fsum(values)/len(values)


def metrics(labels, logits):
    """Only join answers after all logits are sealed. Edge blocks, not individual views, are the units."""
    require(len(labels)==len(logits)==64 and {r["id"] for r in labels}==set(logits),"incomplete scored deck")
    cells={}; strict=ties=full=0; costs=[]; binary=[]
    for r in labels:
        i,o,a,q=(r[k] for k in ("edge","order","idle","query"))
        z=logits[r["id"]];gold=r["gold"]
        require((i,o,a,q) not in cells and len(z)==64 and all(map(math.isfinite,z)) and gold==(IDLE if a==q else ON) and
                identifier(render(r["entities"],o,a,q))==r["id"],"bad cell, score, or answer")
        g=float(z[IDLE])-float(z[ON]);sg=g if gold==IDLE else -g
        cells[i,o,a,q]=(g,sg>EPS)
        strict+=sg>EPS;ties+=abs(g)<=EPS;full+=z[gold]>max(x for j,x in enumerate(z) if j!=gold)
        mx=max(z);costs.append((mx-z[gold])+math.log(math.fsum(math.exp(x-mx) for x in z)))
        binary.append(max(0.,-sg)+math.log1p(math.exp(-abs(sg))))
    edge_values=[];orders=[];whole=quartets=rank_quartets=assignment_pairs=query_pairs=0
    for i in range(8):
        iscore=[];exact=[]
        for o in range(2):
            (aa,ca),(ab,cb),(ba,cc),(bb,cd)=[cells[i,o,a,q] for a,q in itertools.product(range(2),repeat=2)]
            iscore.append(math.fsum((aa,-ab,-ba,bb))/4)
            assignment_pairs+=int(ca and cc)+int(cb and cd); query_pairs+=int(ca and cb)+int(cc and cd)
            exact.append(ca and cb and cc and cd)
            rank_quartets+=min(aa,bb)>max(ab,ba)+EPS
        orders.append(iscore);edge_values.append(mean(iscore));quartets+=sum(exact);whole+=all(exact)
    return dict(interaction=mean(edge_values),by_edge=edge_values,by_order=[mean(list(v)) for v in zip(*orders,strict=True)],
                correct_choices=strict,ties=ties,full_vocab_strict_top1=full,assignment_pairs=assignment_pairs,query_pairs=query_pairs,
                exact_quartets=quartets,rank_quartets=rank_quartets,complete_edges=whole,mean_gold_nll=mean(costs),binary_nll=mean(binary))


def advantage(target, reference):
    if target is None or reference is None:return None
    d=[x-y for x,y in zip(target["by_edge"],reference["by_edge"],strict=True)]
    orders=[x-y for x,y in zip(target["by_order"],reference["by_order"],strict=True)]
    return dict(delta=mean(d),by_edge=d,by_order=orders,positive_edges=sum(x>EPS for x in d),
                passed=mean(d)>EFFECT and sum(x>EPS for x in d)>=6 and min(orders)>EPS)


def decisions(scores):
    out={}
    for route,refs in (("safe",("origin","retention","safe_ce","safe_rotated")),("direct",("origin","direct_ce","direct_rotated"))):
        val=scores[route+"_matched"];vs={k:advantage(val,scores[k]) for k in refs}
        out[route]=dict(contrasts=vs,advantage=all(v is not None and v["passed"] for v in vs.values()),
                        competence=val is not None and val["complete_edges"]>=6 and val["complete_edges"]>scores["origin"]["complete_edges"])
    return out

"""Position and frozen-logit controls. Fit on the train split only."""
import math


def features(kind, rows, origin):
    if kind == "logits":
        return [origin[r["key"]] for r in rows]
    if kind not in ("position","bag"):
        raise ValueError("unknown features")
    out=[]
    for r in rows:
        ids=r["prompt"]
        if kind=="bag":
            out.append([ids.count(i) for i in range(11)])
        else:
            out.append([float(i==t) for t in ids for i in range(11)])
    return out


def standardize(train):
    mean=[math.fsum(col)/len(train) for col in zip(*train)]
    std=[max(1e-4,math.sqrt(math.fsum((x-m)**2 for x in col)/len(train))) for m,col in zip(mean,zip(*train))]
    return mean,std


def apply(rows,mean,std):
    return [[(x-m)/s for x,m,s in zip(row,mean,std)]+[1.] for row in rows]


def wins(z,y):
    return dict(wins=sum(row[g]>row[1-g] for row,g in zip(z,y)),ties=sum(row[0]==row[1] for row in z),n=len(y))


def fit(st,kind,sp,origin,role):
    train=features(kind,sp["train"],origin)
    mean,std=standardize(train)
    x=st.tensor(apply(train,mean,std))
    y=[r[role] for r in sp["train"]]
    xt=x.transpose()
    w=st.Tensor.zeros(len(mean)+1,2)
    seed=st.tensor([[1.]])
    for _ in range(64):
        logits=x.matmul(w,backend="cpu")
        g=logits.cross_entropy_with_logits_backward(y,seed,reduction="mean")
        w=w.sub(xt.matmul(g,backend="cpu").add(w.scale(.01)).scale(.1))
    # Evaluation features and labels are never supplied to the optimizer.
    test=st.tensor(apply(features(kind,sp["test"],origin),mean,std))
    train_z=x.matmul(w,backend="cpu").tolist()
    test_z=test.matmul(w,backend="cpu").tolist()
    return dict(mean=mean,std=std,weights=w.tolist(),updates=64,train=train_z,test=test_z,
                train_counts=wins(train_z,y),test_counts=wins(test_z,[r[role] for r in sp["test"]]))


def verify(kind,sp,origin,role,result):
    train=features(kind,sp["train"],origin)
    mean,std=standardize(train)
    if (result["mean"],result["std"],result["updates"])!=(mean,std,64):
        raise ValueError("probe normalization or budget")
    x=apply(train,mean,std)
    y=[r[role] for r in sp["train"]]
    # Independent f64 reference for the two-class CE step (one column suffices by symmetry).
    w=[0.]*len(x[0])
    for _ in range(64):
        pred=[sum(a*b for a,b in zip(row,w)) for row in x]
        errors=[1/(1+math.exp(-2*z))-(label==0) for z,label in zip(pred,y)]
        w=[a-.1*(math.fsum(e*xi for e,xi in zip(errors,col))/len(x)+.01*a) for a,col in zip(w,zip(*x))]
    stored=result["weights"]
    if len(stored)!=len(w) or any(len(row)!=2 or max(abs(row[0]-a),abs(row[1]+a))>3e-5 for a,row in zip(w,stored)):
        raise ValueError("different learned linear map")
    for split in ("train","test"):
        xs=apply(features(kind,sp[split],origin),mean,std)
        got=result[split]
        expected=[[math.fsum(a*w[j] for a,w in zip(row,stored)) for j in (0,1)] for row in xs]
        if (len(got)!=len(xs) or any(len(pair)!=2 or not all(math.isclose(a,b,rel_tol=3e-6,abs_tol=3e-5)
                for a,b in zip(pair,ref)) for pair,ref in zip(got,expected)) or
                wins(got,[r[role] for r in sp[split]])!=result[split+"_counts"]):
            raise ValueError("probe readback mismatch")
    n=len(sp["test"])
    w=result["test_counts"]["wins"]
    if (kind=="position" and w < n-(1 if sp["formal"] else 0)) or (kind=="bag" and 4*w>3*n):
        raise ValueError("positional capability control or bag symmetry failed")

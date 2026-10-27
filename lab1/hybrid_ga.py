"""Optional hybrid assignment GA: inversion + RSS + elitism remain intact.

Rank-scaled RSS, adaptive inversion, cost-aware 2-swap and true
multiprocess island evolution are independent, opt-in improvements.
"""
from __future__ import annotations
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Config:
    population: int = 120
    generations: int = 40
    mutation_prob: float = 1.0
    inversion_pct: float = 2.0
    elitism: float = .2
    selection: str = "raw"
    rank_power: float = 2.0
    adaptive: bool = False
    adaptive_long_pct: float = 20.0
    adaptive_long_prob: float = .05
    local_every: int = 0
    local_top: int = 1
    local_passes: int = 2
    local_candidates: int = 8
    islands: int = 1
    migration_interval: int = 25
    migration_pct: float = .1

def validate(A, c):
    a=np.asarray(A)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or len(a)<2 or np.any(a<0):
        raise ValueError("nonnegative square matrix with n>=2 required")
    if not 1<=c.islands<=c.population or not 1<=c.population or not 1<=c.generations:
        raise ValueError("invalid population/island/generation budget")
    if not 0<=c.mutation_prob<=1 or not 0<=c.elitism<=1 or not 0<c.inversion_pct<=100:
        raise ValueError("mutation/elitism/inversion outside valid range")
    if c.selection not in ("raw","rank") or c.rank_power<=0:
        raise ValueError("selection must be raw or positive-power rank RSS")
    if not 0<=c.adaptive_long_prob<=1 or not 0<c.adaptive_long_pct<=100:
        raise ValueError("invalid adaptive mutation controls")
    if c.local_every<0 or c.local_top<0 or c.local_top>c.population or c.local_passes<1 or c.local_candidates<1:
        raise ValueError("invalid local search controls")
    if c.migration_interval<1 or not 0<=c.migration_pct<=1:
        raise ValueError("invalid migration controls")
    return np.asarray(a,dtype=np.int32)

def costs(a,p):
    p=np.asarray(p,dtype=np.int32)
    if p.ndim==1:p=p[None,:]
    return a[np.arange(a.shape[0])[None,:],p].sum(axis=1,dtype=np.int64)

def rss_indices(cost,size,elitism,selection,power,rng):
    cost=np.asarray(cost,dtype=np.int64)
    count=len(cost)
    if count<size:raise ValueError("candidate pool smaller than target population")
    k=int(size*elitism)
    order=np.argsort(cost,kind="stable")
    chosen=order[:k].tolist()
    remainder=size-k
    if remainder:
        if selection=="raw":
            if np.any(cost==0):
                scores=(cost==0).astype(float)
            else:
                scores=1.0/cost.astype(float)
        elif selection=="rank":
            ranks=np.empty(count,dtype=float)
            ranks[order]=np.arange(count,0,-1,dtype=float)/count
            scores=ranks**power
        else:
            raise ValueError(selection)
        e=remainder*scores/scores.sum()
        nwhole=np.floor(e).astype(int)
        chosen.extend(np.repeat(np.arange(count),nwhole).tolist())
        missing=size-len(chosen)
        if missing:
            fractional=e-nwhole
            if fractional.sum()>0:
                probabilities=fractional/fractional.sum()
            else:
                probabilities=np.full(count,1/count)
            chosen.extend(rng.choice(count,missing,replace=True,p=probabilities).tolist())
    assert len(chosen)==size
    return np.asarray(chosen,dtype=int)

def mutation_length(n,pct,rng):
    return int(rng.integers(2,min(n,max(2,math.ceil(n*pct/100)))+1))

def mutate(parents,c,iteration,rng):
    n=parents.shape[1]
    output=[]
    for v in parents:
        if rng.random()>=c.mutation_prob:continue
        pct=c.inversion_pct
        if c.adaptive:
            progress=iteration/max(1,c.generations)
            pct=max(.2,c.inversion_pct*(.1+.9*(1-progress)**2))
            if rng.random()<c.adaptive_long_prob:pct=c.adaptive_long_pct
        length=mutation_length(n,pct,rng)
        start=int(rng.integers(0,n-length+1))
        vnew=v.copy()
        vnew[start:start+length]=vnew[start:start+length][::-1]
        output.append(vnew)
    return np.asarray(output,dtype=np.int32).reshape((-1,n))

def improve_swaps(a,perm,topcols,passes=2):
    """Accept only decreasing exact two-worker swap deltas."""
    v=np.asarray(perm,dtype=np.int32).copy()
    n=len(v)
    inverse=np.empty(n,dtype=np.int32)
    inverse[v]=np.arange(n)
    nswaps=0
    for _ in range(passes):
        changed=0
        for i in range(n):
            current=int(v[i])
            best_delta=0
            best_j=-1
            for target in topcols[i]:
                j=int(inverse[target])
                other=int(v[j])
                if i==j:continue
                d=int(a[i,other])+int(a[j,current])-int(a[i,current])-int(a[j,other])
                if d<best_delta:
                    best_delta=d
                    best_j=j
            if best_j>=0:
                col_j=int(v[best_j])
                v[i],v[best_j]=col_j,current
                inverse[current]=best_j
                inverse[col_j]=i
                changed+=1
        nswaps+=changed
        if changed==0:break
    return v,nswaps

def best_columns(a,k):
    n=len(a)
    k=min(k,n)
    cand=np.argpartition(a,k-1,axis=1)[:,:k]
    ordered=np.argsort(np.take_along_axis(a,cand,axis=1),axis=1,kind="stable")
    return np.take_along_axis(cand,ordered,axis=1)

def _evolve(state):
    a,pop,c,begin,steps,rng_state,columns,best,island_id=state
    local_calls=0
    rng=np.random.default_rng()
    rng.bit_generator.state=rng_state
    pc=costs(a,pop)
    for gen in range(begin,begin+steps):
        mutants=mutate(pop,c,gen,rng)
        pool=np.concatenate((pop,mutants),axis=0)
        pc=costs(a,pool)
        arg=int(np.argmin(pc))
        if int(pc[arg])<best[0]:
            best=(int(pc[arg]),pool[arg].copy())
        if c.local_every and (gen+1)%c.local_every==0:
            # Global budget regardless of islands; rotate the remainder.
            cycle=(gen+1)//c.local_every-1
            common,extra=divmod(c.local_top,c.islands)
            quota=common+int((island_id-cycle)%c.islands<extra)
            selected=np.argsort(pc)[:min(quota,len(pc))]
            local_calls+=len(selected)
            better=[]
            for index in selected:
                candidate,_=improve_swaps(a,pool[index],columns,c.local_passes)
                better.append(candidate)
            if better:
                pool=np.concatenate((pool,np.asarray(better,dtype=np.int32)),axis=0)
                pc=costs(a,pool)
                ib=int(np.argmin(pc))
                if int(pc[ib])<best[0]:
                    best=(int(pc[ib]),pool[ib].copy())
        selected=rss_indices(pc,len(pop),c.elitism,c.selection,c.rank_power,rng)
        pop=pool[selected].copy()
    return pop,rng.bit_generator.state,best,local_calls

def run(a,c=Config(),seed=42,workers=2):
    a=validate(a,c)
    n=len(a)
    t0=time.perf_counter()
    rng=np.random.default_rng(seed)
    initial=np.asarray([rng.permutation(n) for _ in range(c.population)],dtype=np.int32)
    sizes=[c.population//c.islands+int(i<c.population%c.islands) for i in range(c.islands)]
    columns=best_columns(a,c.local_candidates) if c.local_every else None
    offsets=np.cumsum([0]+sizes)
    cc=costs(a,initial)
    ib=int(np.argmin(cc))
    best=(int(cc[ib]),initial[ib].copy())
    states=[]
    for i,size in enumerate(sizes):
        rg=np.random.default_rng(seed+1+i*107)
        states.append((initial[offsets[i]:offsets[i+1]].copy(),rg.bit_generator.state,best))
    total_local_calls=0
    if c.islands==1:
        _,_,best,local_calls=_evolve((a,states[0][0],c,0,c.generations,
                          states[0][1],columns,best,0))
        total_local_calls=local_calls
    else:
        actual_workers=max(1,min(workers,c.islands))
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            for beginning in range(0,c.generations,c.migration_interval):
                steps=min(c.migration_interval,c.generations-beginning)
                tasks=[(a,pop,c,beginning,steps,rng_state,columns,island_best,i)
                       for i,(pop,rng_state,island_best) in enumerate(states)]
                outcomes=list(executor.map(_evolve,tasks))
                for _,_,local_best,local_calls in outcomes:
                    total_local_calls+=local_calls
                    if local_best[0]<best[0]:best=local_best
                states=[(p,r,b) for p,r,b,_ in outcomes]
                if beginning+steps<c.generations and c.migration_pct:
                    migrants=[]
                    for pop,_,_ in states:
                        number=min(len(pop),math.ceil(len(pop)*c.migration_pct))
                        migrants.append(pop[np.argsort(costs(a,pop))[:number]].copy())
                    following=[]
                    for i,(pop,rng_state,local_best) in enumerate(states):
                        augmented=np.concatenate((pop,migrants[i-1]),axis=0)
                        rg=np.random.default_rng()
                        rg.bit_generator.state=rng_state
                        indices=rss_indices(costs(a,augmented),sizes[i],c.elitism,
                                            c.selection,c.rank_power,rg)
                        following.append((augmented[indices].copy(),rg.bit_generator.state,local_best))
                    states=following
    elapsed=time.perf_counter()-t0
    return {"cost":int(best[0]),"seconds":float(elapsed),"best":best[1],
            "generations":c.generations,"total_population":sum(sizes),
            "num_islands":c.islands,"local_calls":total_local_calls}

"""Deterministic multi-start open tour with edge-delta local improvement."""
import math
import numpy as np

def route(start,nodes,starts=12,end=None):
    keys=sorted(nodes); n=len(keys)
    if not n: return []
    points=np.array([nodes[k] for k in keys]+[start],dtype=float)
    d=np.sqrt(((points[:,None]-points[None,:])**2).sum(axis=2)).tolist()
    ends=[0.]*n if end is None else [math.dist(nodes[k],end) for k in keys]
    def edge(a,c): return (0. if a==n else ends[a]) if c is None else d[a][c]
    def cost(r): return d[n][r[0]]+sum(d[a][c] for a,c in zip(r,r[1:]))+ends[r[-1]]
    def improve(r):
        for _ in range(100):
            changed=False
            for i in range(n-1):
                a=n if i==0 else r[i-1]; c=r[i]
                for j in range(i+1,n):
                    e=r[j]; f=None if j==n-1 else r[j+1]
                    delta=d[a][e]+edge(c,f)-d[a][c]-edge(e,f)
                    if delta < -1e-7:
                        r[i:j+1]=reversed(r[i:j+1]); changed=True; break
                if changed: break
            if changed: continue
            for length in (1,2,3):
                for i in range(n-length+1):
                    piece=r[i:i+length]; rest=r[:i]+r[i+length:]
                    a=n if i==0 else r[i-1]; f=None if i+length==n else r[i+length]
                    remove=edge(a,f)-d[a][piece[0]]-edge(piece[-1],f)
                    for j in range(len(rest)+1):
                        if j==i: continue
                        p=n if j==0 else rest[j-1]; q=None if j==len(rest) else rest[j]
                        for seg in (piece,piece[::-1]):
                            delta=remove+d[p][seg[0]]+edge(seg[-1],q)-edge(p,q)
                            if delta < -1e-7:
                                r=rest[:j]+seg+rest[j:]; changed=True; break
                        if changed: break
                    if changed: break
                if changed: break
            if not changed: break
        return r
    firsts=sorted(range(n),key=lambda k:math.atan2(points[k,1]-start[1],points[k,0]-start[0]))
    firsts=list(dict.fromkeys([min(range(n),key=lambda k:d[n][k])]+[firsts[i] for i in np.linspace(0,n-1,min(starts,n),dtype=int)]))
    candidates=[]
    for first in firsts:
        r=[first]; remaining=set(range(n))-{first}
        while remaining:
            nxt=min(remaining,key=lambda k:(d[r[-1]][k],k));r.append(nxt);remaining.remove(nxt)
        candidates.append(improve(r))
    return [keys[i] for i in min(candidates,key=cost)]

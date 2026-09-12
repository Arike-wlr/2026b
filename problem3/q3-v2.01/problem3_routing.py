"""Deterministic multistart open-tour local search; a heuristic, not an optimality proof."""
import numpy as np


def optimize_route(start, points):
    n=len(points)
    if not n:return []
    xy=np.vstack([points,start])
    d0=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
    d=np.zeros((n+2,n+2));d[:n+1,:n+1]=d0
    starts=[]
    for first in range(n):
        order=[first];left=set(range(n));left.remove(first)
        while left:
            q=min(left,key=lambda j:(d[order[-1],j],j))
            order.append(q);left.remove(q)
        starts.append(order)
    best=None;score=float('inf')
    for initial in starts:
        r=np.array(initial,dtype=int)
        for _ in range(100):
            prev=np.r_[n,r[:-1]];nxt=np.r_[r[1:],n+1]
            delta=d[prev[:,None],r[None,:]]+d[r[:,None],nxt[None,:]]-d[prev,r][:,None]-d[r,nxt][None,:]
            delta[np.tril_indices(n)]=np.inf
            i,j=np.unravel_index(np.argmin(delta),delta.shape)
            if delta[i,j]>=-1e-8:break
            r[i:j+1]=r[i:j+1][::-1]
        cost=float(d[n,r[0]]+d[r[:-1],r[1:]].sum())
        if cost<score-1e-8:score=cost;best=r.tolist()
    return best


def route_cost(start,points,order):
    xy=np.vstack([start]+[points[j] for j in order])
    return float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())

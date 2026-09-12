"""问题3 v2.01：开发集冻结的新候选。原v2.00保存在problem3_v200.py。"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import problem3_legacy as legacy
from problem3_v200 import OptimizedStrategy as V200, _TYPES
from problem3_legacy import DEFAULT_HEX_RING_RADIUS_M,DEFAULT_PIGGYBACK_GAIN_M2
from coverage_strategy import Task,reception
from problem3_geometry import min_enclosing_circle,clip_wedge,clip_halfplane
from problem3_routing import optimize_route,route_cost


def quick_route_cost(start,points):
    n=len(points)
    if not n:return 0.
    points=np.asarray(points);xy=np.vstack([points,start])
    d0=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
    d=np.zeros((n+2,n+2));d[:n+1,:n+1]=d0
    left=set(range(n));order=[];last=n
    while left:
        q=min(left,key=lambda j:(d[last,j],j));order.append(q);left.remove(q);last=q
    r=np.array(order,dtype=int)
    for _ in range(50):
        prev=np.r_[n,r[:-1]];nxt=np.r_[r[1:],n+1]
        delta=d[prev[:,None],r[None,:]]+d[r[:,None],nxt[None,:]]-d[prev,r][:,None]-d[r,nxt][None,:]
        delta[np.tril_indices(n)]=np.inf
        i,j=np.unravel_index(np.argmin(delta),delta.shape)
        if delta[i,j]>=-1e-8:break
        r[i:j+1]=r[i:j+1][::-1]
    return float(d[n,r[0]]+d[r[:-1],r[1:]].sum())

class ContinuousProbe(V200):
    max_evals=42
    starts=2
    def choose(self):
        tasks=self.target_tasks();tasks+=self.planned_searches(tasks)
        if not tasks:return None
        before=np.asarray(self.robot.current_position)
        order=optimize_route(before,[t.point for t in tasks]);first=tasks[order[0]]
        if first.kind!='probe':return first
        cs=self.channels[first.channel]
        after=tasks[order[1]].point if len(order)>1 else None
        candidates=self.candidates(cs,before,after)
        if not candidates:return first
        values=[self.probe_score(cs,p,before,after) for p in candidates]
        best=min(range(len(values)),key=lambda i:values[i]);point=candidates[best];value=values[best]
        from scipy.optimize import minimize
        def objective(p):
            if np.linalg.norm(p)>2400:return 1e5+np.linalg.norm(p)
            if any(np.linalg.norm(p-q)<2 for q in self.used[cs.channel]):return 1e5
            return self.probe_score(cs,p,before,after)
        for idx in np.argsort(values)[:self.starts]:
            x=candidates[idx]
            size=min(100.,max(20.,cs.radius*.15))
            result=minimize(objective,x,method='Nelder-Mead',options=dict(maxfev=self.max_evals,xatol=2.,fatol=.2,
                initial_simplex=np.array([x,x+[size,0],x+[0,size]])))
            if result.fun<value-.1:value=float(result.fun);point=result.x.copy()
        first.point=point.copy();return first

def reset_unexecuted_hex(strategy):
    strategy._hex_points=None;strategy._hex_disks=None;strategy._hex_orientation_rad=None
    strategy._cover_patches=None;strategy._cover_allowed=None
    strategy._cover_neighborhood_points=None

class SeedRouteValue(V200):
    residual_weight=.3
    initial_scenarios=5
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.initial_information_done=False;self.initial_measure_pending=False
    def choose(self):
        t=super().choose()
        if t is None or self.initial_information_done:return t
        self.initial_information_done=True
        detected=[cs for cs in self.channels.values() if cs.status=='DETECTED']
        if not detected:return t
        heading=math.atan2(t.point[1],t.point[0])
        candidates=[t.point.copy()]
        for radius in (200.,300.,450.):
            for off in (-60.,-30.,0.,30.,60.):
                angle=heading+math.radians(off);candidates.append(radius*np.array([math.cos(angle),math.sin(angle)]))
        hypothetical=[]
        for i in range(self.initial_scenarios):
            gg=[]
            for cs in detected:
                options=self.particles(cs);gg.append(options[(i+cs.channel)%len(options)])
            hypothetical.append(gg)
        ring=self._hex_points or []
        values=[]
        for p in candidates:
            total=0.
            for k,scene in enumerate(hypothetical):
                centers=[];penalty=0.
                for cs,(g,rc) in zip(detected,scene):
                    poly=cs.feasible
                    if np.linalg.norm(g-p)<=rc:
                        error=(-.7,.0,.7)[k%3]
                        angle=math.atan2(g[1]-p[1],g[0]-p[0])+math.radians(error)
                        poly=clip_wedge(poly,p,angle,math.radians(self.cfg.bearing_error_deg))
                    else:
                        first=cs.first_direction[0];v=p-first;rhs=float(p@p-first@first)
                        poly=clip_halfplane(poly,-2*v[0],-2*v[1],rhs)
                    if len(poly):
                        c,r=min_enclosing_circle(poly);centers.append(c)
                        penalty+=self.residual_weight*max(0,r-20)
                    else:centers.append(cs.center);penalty+=1000
                total+=np.linalg.norm(p)+quick_route_cost(p,centers+ring)+penalty
            values.append(total/self.initial_scenarios)
        p=candidates[int(np.argmin(values))]
        primary=min(detected,key=lambda cs:np.linalg.norm(cs.center-p))
        self.initial_measure_pending=True
        return Task('probe',p.copy(),primary.channel)
    def execute(self,t):
        super().execute(t)
        if self.initial_measure_pending:
            self.initial_measure_pending=False
            p=np.asarray(self.robot.current_position)
            for ch,cs in self.channels.items():
                if cs.status=='DETECTED' and cs.radius>58 and all(np.linalg.norm(p-q)>10 for q in self.used[ch]):
                    self.sense(ch,p,'INITIAL_ROUTE_INFORMATION')

class SeedRouteContinuous(SeedRouteValue,ContinuousProbe):pass

OptimizedStrategy=SeedRouteContinuous

def build_strategy():
    return OptimizedStrategy,_TYPES

def main(argv=None):
    p=argparse.ArgumentParser(description="问题3 v2.01 离线测试")
    p.add_argument('--cases',type=int,default=20)
    p.add_argument('--random-state',type=int,default=None)
    p.add_argument('--source-count',type=int,default=None)
    p.add_argument('--output-prefix',type=Path,default=None)
    a=p.parse_args(argv)
    if a.cases<1:p.error('--cases must be positive')
    if a.source_count is not None and not 10<=a.source_count<=16:
        p.error('--source-count must be between 10 and 16')
    seed=legacy.resolve_random_state(a.random_state)
    output=a.output_prefix or Path(__file__).resolve().parent/'outputs'/f'local_{seed}'
    print(f'随机种子 {seed}，案例 {a.cases}，策略 v2.01',flush=True)
    results=[]
    for i in range(a.cases):
        r=legacy.run_case(i+1,seed+i,OptimizedStrategy,num_sources=a.source_count)
        results.append(r)
        print(f"[{i+1}/{a.cases}] {r['source_cleared']}/{r['source_total']}，"
              f"{r['seconds_per_source']:.2f} 秒/源，error={r['error']}",flush=True)
    summary=legacy.summarize(results,seed)
    summary.update(strategy='q3_v2_01',seeds=list(range(seed,seed+a.cases)))
    legacy.write_results(results,summary,output)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if all(r['success'] for r in results) else 1


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())

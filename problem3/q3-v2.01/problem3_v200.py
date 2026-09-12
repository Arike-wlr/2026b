"""问题3 v2.00：六扇区残余覆盖 + 多初值开放路径。

默认使用开发集冻结的 HexMultiNoScan 方案。旧模型与通信层保存在
problem3_legacy.py；本文件可离线运行，problem3_main.py 为演练入口。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Point
import problem3_legacy as legacy
from problem3_legacy import DEFAULT_HEX_RING_RADIUS_M, DEFAULT_PIGGYBACK_GAIN_M2
from problem3_routing import optimize_route, route_cost

_TYPES=legacy.load_implementation()
from lookahead_strategy import LookaheadConfig
from coverage_strategy import reception

_HexBase=legacy.build_hex_cover_strategy(
    _TYPES[1], 1e30, always_finish_channel=False, ring_radius_m=1125.,
    enable_neighborhood_cover=True)


class OptimizedStrategy(_HexBase):
    def __init__(self,robot,config=None):
        super().__init__(robot,config,options=LookaheadConfig(history=True,full_regions=True))

    def _optimized_neighborhood_searches(self,tasks):
        need=self.remaining_shape()
        original_allowed=self._cover_allowed
        original_patches=self._cover_patches
        updated=[];allowed=[]
        for old,region in zip(original_patches,original_allowed):
            piece=old.intersection(need)
            if piece.is_empty or piece.area<1e-8:
                updated.append(old);allowed.append(region);continue
            hull=piece.convex_hull
            if hull.geom_type!='Polygon':
                updated.append(old);allowed.append(region);continue
            vertices=np.asarray(hull.exterior.coords)[:-1]
            a=None
            for v in vertices:
                disk=Point(*v).buffer(999.5,quad_segs=24)
                a=disk if a is None else a.intersection(disk)
            if a.is_empty:
                updated.append(old);allowed.append(region)
            else:
                updated.append(hull);allowed.append(a)
        self._cover_patches=updated;self._cover_allowed=allowed
        try:return super()._optimized_neighborhood_searches(tasks)
        finally:self._cover_patches=original_patches;self._cover_allowed=original_allowed

    def _route_distance(self,points):
        if not points:return 0.
        start=self.robot.current_position
        return route_cost(start,points,optimize_route(start,points))

    def choose(self):
        tasks=self.target_tasks();tasks+=self.planned_searches(tasks)
        if not tasks:return None
        p=np.asarray(self.robot.current_position)
        order=optimize_route(p,[t.point for t in tasks])
        first=tasks[order[0]]
        if first.kind=='probe':
            cs=self.channels[first.channel]
            after=tasks[order[1]].point if len(order)>1 else None
            candidates=self.candidates(cs,p,after)
            if candidates:first.point=min(candidates,key=lambda q:self.probe_score(cs,q,p,after))
        return first

    def scan(self,p,force=False):
        if force:return super().scan(p,True)
        return False


def build_strategy():
    """Return the frozen v2.00 policy and the unchanged client/config types."""
    return OptimizedStrategy,_TYPES


def main(argv=None):
    p=argparse.ArgumentParser(description="问题3 v2.00 离线测试")
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
    print(f'随机种子 {seed}，案例 {a.cases}，策略 v2.00',flush=True)
    results=[]
    for i in range(a.cases):
        r=legacy.run_case(i+1,seed+i,OptimizedStrategy,num_sources=a.source_count)
        results.append(r)
        print(f"[{i+1}/{a.cases}] {r['source_cleared']}/{r['source_total']}，"
              f"{r['seconds_per_source']:.2f} 秒/源，error={r['error']}",flush=True)
    summary=legacy.summarize(results,seed)
    summary.update(strategy='q3_v2_00',seeds=list(range(seed,seed+a.cases)))
    legacy.write_results(results,summary,output)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if all(r['success'] for r in results) else 1


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())

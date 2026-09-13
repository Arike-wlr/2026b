"""第四问 21 点框架改进版。离线运行本文件；在线入口见 problem4_main.py。

保留 21 点确定性覆盖和原有观测几何，在行走途中补测，并以经过半径检查的
凸分块覆盖执行定向源兜底清除。只使用 RobotClient 公共接口。
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import problem4_baseline as b
import routing

class EfficientClear:
    """Convex pieces covered by verified <=20m disks; independent of emission orientation."""
    def _grid_clear(self,ch):
        from problem3_geometry import clip_halfplane
        self.grid_fallback_count+=1
        stack=[self._polygon(ch)]; cells=[]
        while stack:
            poly=stack.pop(); circle=b.minimum_enclosing_circle(poly)
            if circle.radius<=b.CLEAR_RADIUS-1e-5:
                cells.append((circle.center,poly));continue
            distances=((poly[:,None,:]-poly[None,:,:])**2).sum(axis=2)
            i,j=np.unravel_index(distances.argmax(),distances.shape)
            axis=poly[j]-poly[i];axis=axis/np.linalg.norm(axis)
            proj=poly@axis;mid=float((proj.min()+proj.max())/2)
            stack.extend([clip_halfplane(poly,*axis,-mid),clip_halfplane(poly,*(-axis),mid)])
        while cells:
            if not self._time_left():raise b.TimeBudgetExceeded('覆盖清除时间不足')
            index=min(range(len(cells)),key=lambda i:b.distance(self.robot.current_position,cells[i][0]))
            point,poly=cells.pop(index)
            if self._clear(*point,ch).cleared:self.cleared.add(ch);return
        raise RuntimeError('Certified cover exhausted without clearing')

class Cached(b.Problem4Strategy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._polygon_cache={}
        self._circle_cache={}
    def _polygon(self,ch):
        key=(ch,len(self.observations[ch]))
        if key not in self._polygon_cache: self._polygon_cache[key]=super()._polygon(ch)
        return self._polygon_cache[key]
    def _circle(self,ch):
        key=(ch,len(self.observations[ch]))
        if key not in self._circle_cache: self._circle_cache[key]=b.minimum_enclosing_circle(self._polygon(ch))
        return self._circle_cache[key]

class Joint(Cached):
    SHARE=True
    ROUTE_STARTS=12
    PROVISIONAL=False
    EARLY_CLEAR_RADIUS=58.
    ENROUTE=False
    ENROUTE_CLOSE=900.
    ENROUTE_LIMIT=3
    PHANTOMS=False
    ROTATE=False
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.enroute_counts={ch:0 for ch in self.channels}
        self.diagnostics={'enroute_measures':0}
    def _enroute(self,destination,exclude=None):
        if not self.ENROUTE:return False
        start=np.array(self.robot.current_position);end=np.array(destination)
        length=float(np.linalg.norm(end-start))
        if length<200:return False
        options=[]
        for ch in sorted(self.detected-self.cleared):
            if ch==exclude or self.enroute_counts[ch]>=self.ENROUTE_LIMIT:continue
            circle=self._circle(ch)
            if circle.radius<=58:continue
            center=np.array(circle.center)
            projection=float(np.clip(np.dot(center-start,end-start)/length**2,.15,.85))
            fractions=sorted(set([.25,.5,.75,projection]))
            for f in fractions:
                point=tuple(start+f*(end-start))
                near=b.distance(point,circle.center)
                if near>self.ENROUTE_CLOSE or near<40:continue
                hypo=self._hypothetical_circle_after_measure(ch,point,circle)
                if hypo.radius>58:continue
                # Prioritize short approaches and stronger intersection; this is only a value heuristic.
                if max(self._maximum_crossing_angle(ch,point,circle),0)<12:continue
                options.append((near+.15*f*length+2*hypo.radius,ch,point))
        if not options:return False
        _,ch,point=min(options)
        self.enroute_counts[ch]+=1;self.diagnostics['enroute_measures']+=1
        result=self._measure(*point,ch,is_probe=True)
        self._record_measurement(ch,point,result.result,result.svd_deg)
        if self.SHARE:self._finish_shared_measure_at(self.robot.current_position,ch)
        return True
    def run_survey(self):
        stations=b.twenty_one_point_stations()
        remaining={i:p for i,p in enumerate(stations) if i}
        station=stations[0]; visit=0
        while True:
            self.survey_visited_station_count+=1
            channels=self._unresolved() if len(self.detected)<b.SOURCE_COUNT_MAX else set()
            if self.survey_detected_channels:
                channels.update(ch for ch in self.detected-self.cleared if self._needs_free_survey_measurement(ch,station))
            ordered=sorted(channels,reverse=bool(visit%2))
            current=self.robot.current_channel
            if current in ordered: ordered.remove(current); ordered.insert(0,current)
            for ch in ordered:
                if not self._time_left():
                    self.incomplete_reason='现实时间不足，未完成全部安全巡检点';return self._end_survey()
                r=self._measure(*station,ch);self.survey_measurement_count+=1
                if ch in self.detected: self.stats['survey_remeasures']+=1
                self._record_measurement(ch,station,r.result,r.svd_deg)
                if len(self.detected)>=b.SOURCE_COUNT_MAX:return self._end_survey()
            if not remaining:return self._end_survey()
            if visit==0 and self.ROTATE:
                # Rotation is an isometry of the circular target and the 21-point certificate.
                best=None
                for angle in np.linspace(0,np.pi/2,12,endpoint=False):
                    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                    candidate={i:tuple(rot@p) for i,p in remaining.items()}
                    nodes={1000+i:p for i,p in candidate.items()}
                    nodes.update({ch:self._circle(ch).center for ch in sorted(self.detected-self.cleared)})
                    order=routing.route(self.robot.current_position,nodes,starts=4)
                    score=b.open_route_length(self.robot.current_position,order,nodes)
                    if best is None or score<best[0]-1e-7:best=(score,candidate)
                remaining=best[1]
            while True:
                if not self._time_left(): raise b.TimeBudgetExceeded('清除调度时间不足')
                circles={ch:self._circle(ch) for ch in sorted(self.detected-self.cleared)}
                clear={ch:c for ch,c in circles.items() if c.radius<=self.EARLY_CLEAR_RADIUS}
                nodes={self.SURVEY_NODE_OFFSET+i:p for i,p in remaining.items()}
                nodes.update({ch:c.center for ch,c in clear.items()})
                if self.PROVISIONAL:
                    nodes.update({100+ch:c.center for ch,c in circles.items() if ch not in clear and len(self.observations[ch])>=2})
                if self.PHANTOMS:
                    nodes.update({200+ch:c.center for ch,c in circles.items() if ch not in clear})
                order=routing.route(self.robot.current_position,nodes,starts=self.ROUTE_STARTS)
                if self.PHANTOMS:order=[n for n in order if not 200<n<300]
                nxt=order[0]
                if self._enroute(nodes[nxt],nxt if nxt<100 else None):continue
                if nxt<100:
                    self.stats['survey_inserted_clears']+=1
                    self._following_point=nodes[order[1]] if len(order)>1 else None
                    self._clear_circle(nxt,clear[nxt])
                    self._following_point=None
                    if self.SHARE:self._finish_shared_measure_at(self.robot.current_position,nxt)
                    continue
                if nxt<self.SURVEY_NODE_OFFSET:
                    ch=nxt-100
                    if self._clear(*circles[ch].center,ch).cleared:self.cleared.add(ch)
                    else:
                        # Near the posterior center; a positive bearing can collapse a long range interval.
                        point=self.robot.current_position;r=self._measure(*point,ch,is_probe=True)
                        self._record_measurement(ch,point,r.result,r.svd_deg)
                        if ch not in self.cleared and self._circle(ch).radius>self.EARLY_CLEAR_RADIUS:self._grid_clear(ch)
                    if self.SHARE:self._finish_shared_measure_at(self.robot.current_position,ch)
                    continue
                station=remaining.pop(nxt-self.SURVEY_NODE_OFFSET);visit+=1;break

class Enroute(Joint): ENROUTE=True

class EnrouteOnce(Enroute): ENROUTE_LIMIT=1

class ClearOnce(EfficientClear,EnrouteOnce):pass

class Problem4Strategy(ClearOnce):
    """Frozen v2.0 candidate."""
    pass


def main(argv=None):
    parser=argparse.ArgumentParser(description='第四问21点改进版：本地随机验证')
    parser.add_argument('--cases',type=int,default=20)
    parser.add_argument('--random-state',type=int)
    parser.add_argument('--source-count',type=int,choices=range(10,17))
    parser.add_argument('--directional-probability',type=float,default=.5)
    parser.add_argument('--allow-pure',action='store_true')
    parser.add_argument('--route-end-at-origin',action='store_true',help='仅将原点计入收尾路线评分，不执行返航；默认关闭')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--log-first',action='store_true')
    args=parser.parse_args(argv)
    if args.cases<1:parser.error('--cases must be positive')
    seed=b.resolve_random_state(args.random_state)
    out=args.output_dir or Path(__file__).resolve().parent/'outputs'/datetime.now().strftime('local_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    # Reuse the supplied local generator/evaluator with the frozen strategy class.
    b.Problem4Strategy=Problem4Strategy
    rows=[]
    for i in range(args.cases):
        row=b.run_problem4_case(i+1,seed+i,directional_probability=args.directional_probability,force_mixed=not args.allow_pure,route_end_at_origin=args.route_end_at_origin,source_count=args.source_count,log_path=str(out/'actions.tsv') if i==0 and args.log_first else None)
        rows.append(row)
        print(f'[{i+1}/{args.cases}] seed={seed+i} cleared={row.cleared_count}/{row.source_count} total={row.total_time_s:.2f}s per-source={row.seconds_per_source:.2f}s',flush=True)
    summary=b.summarize_problem4(rows,seed,args.directional_probability,True,args.route_end_at_origin)
    summary['strategy']='problem4_21_v2.0'
    summary['metric_note']='总虚拟秒数包含搜索、移动、测量、切频、清除；不返航。average_clear_time_s 是逐案例 T/N 的平均。'
    b.write_results(rows,summary,out/'results')
    print(f"mean={summary['mean_total_time_s']:.2f}s; mean per-source={summary['average_clear_time_s']:.2f}s; output={out}")
    return 0

if __name__=='__main__':
    raise SystemExit(main())

"""Coverage strategy entry: offline by default; manual PRACTICE only for HTTP."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE))

from problem3_strategy import StrategyConfig

RECOMMENDED='route_joint_regions_selective_approach'
# 'supplied' = 本仓库原文件基线（problem3_strategy.Problem3Strategy），用于和覆盖
# 地图各 policy 在同一批种子上对比；它不需要 coverage_strategy.py（模型）在场。
SUPPLIED='supplied'
POLICIES=[SUPPLIED,'nn_separate','nn_joint_approach','route_joint_approach',
          'route_joint_regions_approach',RECOMMENDED]


def main(argv=None):
    p=argparse.ArgumentParser(description='Q3 coverage-map strategy; OFFLINE default, no formal mode')
    p.add_argument('--mode',choices=['offline','practice'],default='offline')
    p.add_argument('--policy',choices=POLICIES,default=RECOMMENDED)
    p.add_argument('--seed',type=int,default=5000)
    p.add_argument('--robot-id')
    p.add_argument('--base-url',default='http://127.0.0.1:2026')
    p.add_argument('--confirm-problem3-practice',action='store_true')
    args=p.parse_args(argv)
    if args.mode=='offline':
        from research_adapter import run_coverage_case as run_case
        result=run_case(args.policy,args.seed)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result['success'] else 1
    # Existing guard verifies loopback URL and explicit interactive practice label
    # BEFORE creating the HTTP client. Protocol itself cannot detect platform mode.
    from research_adapter import practice_guard
    practice_guard(args)
    from client import RobotClient
    out=HERE/'research'/('practice_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    robot=RobotClient(args.robot_id,base_url=args.base_url,log_dir=str(out/'http'))
    cfg=StrategyConfig(bearing_error_deg=1.01,log_path=str(out/'actions.tsv'))
    if args.policy==SUPPLIED:
        # 原文件基线走本仓库策略，不需要覆盖地图模型在场
        from problem3_strategy import Problem3Strategy
        strategy=Problem3Strategy(robot,cfg)
    else:
        # 模型文件只在真正用到时才导入；本适配层不改动它
        from coverage_strategy import CoverageStrategy, MapConfig
        strategy=CoverageStrategy(robot,cfg,MapConfig(mode=args.policy))
    result={'all_resolved':False,'mode':'practice','policy':args.policy}
    try:
        result.update(strategy.run())
    except Exception as e:
        result['error']=repr(e)
    finally:
        robot.close()
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['all_resolved'] else 1


if __name__=='__main__':raise SystemExit(main())

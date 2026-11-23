"""Run the original experiment scripts without publishing raw or detailed data."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def stages(task,data,output):
    a=str(data);b=str(output);python=sys.executable
    if task in ('housing','all'):
        yield 'housing_baseline',[python,'housing.py',a,str(output/'任务一_房价')]
    if task in ('taxi','all'):
        yield 'taxi_clean_stops_od',[python,'taxi.py',a,str(output/'任务二_轨迹')]
    if task in ('housing','all'):
        yield 'housing_spatial_sensitivity',[python,'-c',
            'from pathlib import Path; import sys; from robustness import housing_check; housing_check(Path(sys.argv[1]))',b]
        for name in ['explain_housing','compare_housing_models','check_gwr','explain_best_model','explain_coordinates']:
            yield name,[python,name+'.py',a,b]
    if task in ('taxi','all'):
        yield 'taxi_same_vehicle_comparison',[python,'-c',
            'from pathlib import Path; import sys; from robustness import taxi_check; taxi_check(Path(sys.argv[1]),Path(sys.argv[2]))',a,b]
        for name in ['diagnose_taxi','map_taxi_od']:
            yield name,[python,name+'.py',a,b]

def main():
    p=argparse.ArgumentParser(description='复现深圳房价与出租车轨迹课程实验')
    p.add_argument('--data',type=Path,default=ROOT/'data/raw',help='已解压的课程数据目录')
    p.add_argument('--output',type=Path,default=ROOT/'local_runs/results',help='默认不被 Git 跟踪')
    p.add_argument('--task',choices=['housing','taxi','all'],default='all')
    p.add_argument('--list',action='store_true',help='只显示步骤，不读取数据或导入分析依赖')
    args=p.parse_args();data=args.data.expanduser().resolve();out=args.output.expanduser().resolve()
    if out==ROOT/'分析结果':
        p.error('请用新的输出目录保留公开的历史汇总，勿覆盖 分析结果。')
    plan=list(stages(args.task,data,out))
    if args.list:
        for i,(name,_) in enumerate(plan,1):print(f'{i:02d} {name}')
        return
    patterns=['*房价*.geojson','*地铁站点*.geojson','*POI*.csv','*行政区划*.geojson'] if args.task in ('housing','all') else ['*行政区划*.geojson']
    if args.task in ('taxi','all'):patterns.append('*Taxi*.csv')
    missing=[pat for pat in patterns if not any(data.rglob(pat))]
    if missing:p.error('缺少已解压数据：'+', '.join(missing))
    out.mkdir(parents=True,exist_ok=True);logs=out/'logs';logs.mkdir(exist_ok=True)
    for i,(name,command) in enumerate(plan,1):
        print(f'[{i}/{len(plan)}] {name}',flush=True)
        log=logs/f'{i:02d}_{name}.log'
        with log.open('w',encoding='utf-8') as stream:
            stream.write(f'Started: {datetime.now().astimezone().isoformat()}\n');stream.flush()
            result=subprocess.run(command,cwd=ROOT/'分析结果',stdout=stream,stderr=subprocess.STDOUT)
            stream.write(f'\nExit code: {result.returncode}\n')
        if result.returncode:raise SystemExit(f'步骤失败，查看 {log}')
    print(f'完成。详细结果与运行日志保留于 {out}')

if __name__=='__main__':main()

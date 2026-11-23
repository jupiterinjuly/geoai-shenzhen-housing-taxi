"""Read raw course data and independently check the stored results.

Usage: python 复核实验.py /path/to/extracted/course/data
Original files are read only; new evidence is written beside this script.
"""
from pathlib import Path
import argparse
import sys
import json
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ORIGINAL = PROJECT / '分析结果'
OUT = HERE / '证据'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ORIGINAL))
from common import districts, features, file_in, lookup_district, projection
from housing import PREDICTORS, CATEGORIES, fitted_models, score
from compare_housing_models import gwr_predict, remove_buffer, choose_parameter
from taxi import quality, segments, stop_table, candidate_trip_table, restrict_to_city

results = {}

def save(frame, name):
    frame.to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')

def metrics(y, p):
    return {'MAE': float(mean_absolute_error(y, p)),
            'RMSE': float(mean_squared_error(y, p) ** .5),
            'R2': float(r2_score(y, p))}

def check_features(data):
    stored = pd.read_csv(ORIGINAL / '任务一_房价/房价特征与标签.csv')
    raw = pd.DataFrame([r['properties'] for r in features(data, '*房价*.geojson')])
    assert len(raw) == len(stored)
    np.testing.assert_allclose(raw.avgprice, stored.avgprice, equal_nan=True)
    assert np.array_equal(raw['count'], stored['count'])
    xy = np.array(projection.transform(raw.centerlon, raw.centerlat)).T
    np.testing.assert_allclose(xy, stored[['x_m', 'y_m']])
    station = pd.DataFrame([(r['properties']['stationnames'], *r['geometry']['coordinates'])
                           for r in features(data, '*地铁站点*.geojson')],
                           columns=['name', 'lon', 'lat']).drop_duplicates()
    station['x'], station['y'] = projection.transform(station.lon, station.lat)
    centers = station.groupby('name')[['x', 'y']].median()
    all_distances = distance.cdist(xy, centers)
    np.testing.assert_allclose(all_distances.min(axis=1) / 1000, stored.metro_km)
    assert np.array_equal(centers.index[all_distances.argmin(axis=1)], stored.nearest_station)
    poi = pd.read_csv(file_in(data, '*POI*.csv'))
    clean = poi.drop_duplicates(subset=['名称', '大类', '中类', '经度', '纬度'])
    valid = clean.经度.between(-180, 180) & clean.纬度.between(-90, 90)
    poi_checks = []
    for cat, col in CATEGORIES.items():
        p = clean.loc[valid & clean.大类.eq(cat)]
        loc = np.array(projection.transform(p.经度, p.纬度)).T
        counts = np.concatenate([(distance.cdist(chunk, loc) <= 1000).sum(axis=1)
                                 for chunk in np.array_split(xy, 28)])
        assert np.array_equal(counts, stored[col])
        poi_checks.append(dict(category=cat, unique_poi=len(p), cells_checked=len(counts),
                               max_count_difference=int(np.max(np.abs(counts-stored[col])))))
    save(pd.DataFrame(poi_checks), '全部房价单元特征独立核验')
    bad = stored.loc[stored.single_inconsistent & stored.avgprice.notna()]
    save(bad, '121个价格汇总口径疑点单元')
    results['housing_data'] = dict(n=len(stored), labeled=int(stored.avgprice.notna().sum()),
        missing=int(stored.avgprice.isna().sum()), inconsistent_single=len(bad),
        avg_outside_minmax=int((stored.avgprice.notna() &
            ((stored.avgprice < stored.minprice) | (stored.avgprice > stored.maxprice))).sum()),
        raw_poi=len(poi), deduplicated_poi=len(clean),
        duplicate_poi_removed=len(poi)-len(clean), metro_station_names=len(centers))
    subset = clean.loc[clean.中类.eq('地铁') & clean.名称.isin(centers.index)]
    pp = subset.groupby('名称')[['经度', '纬度']].median()
    common = centers.index.intersection(pp.index)
    px, py = projection.transform(pp.loc[common, '经度'], pp.loc[common, '纬度'])
    difference = np.hypot(centers.loc[common, 'x']-px, centers.loc[common, 'y']-py)
    save(pd.DataFrame({'station':common, 'difference_m':difference}), '同名站点跨源位置独立核验')
    results['metro_alignment'] = dict(n=len(common), median_m=float(np.median(difference)),
        p90_m=float(np.quantile(difference, .9)), max_m=float(np.max(difference)))
    return stored

def validate_stored_predictions(frame):
    rows = []
    folder = ORIGINAL / '任务一_房价'
    for predfile, fields, ycol, pcol, metricfile in [
        ('房价折外预测', ['variant', 'split', 'model'], 'true', 'predicted', '房价汇总指标'),
        ('扩展模型_逐单元预测', ['axis', 'model'], 'observed', 'predicted', '扩展模型_空间验证')]:
        pred = pd.read_csv(folder / (predfile+'.csv'))
        old = pd.read_csv(folder / (metricfile+'.csv'))
        for key, group in pred.groupby(fields):
            assert not group.cell_id.duplicated().any()
            selected = old.copy()
            for field, value in zip(fields, key):selected = selected.loc[selected[field].eq(value)]
            assert len(selected) == 1
            measured = metrics(group[ycol], group[pcol])
            for m, v in measured.items():
                diff = abs(v-selected.iloc[0][m]); assert diff < 1e-7
                rows.append(dict(source=predfile, group='|'.join(map(str,key)),
                                 metric=m, recomputed=v, difference=diff))
    save(pd.DataFrame(rows), '历史逐单元预测与指标重算')
    shaprows=[]
    for p in folder.glob('*明细.csv'):
        df=pd.read_csv(p)
        cols=[c for c in df if c.startswith('shap_')]
        if cols and 'base' in df and 'predicted' in df:
            err=np.abs(df['base']+df[cols].sum(axis=1)-df.predicted)
            assert err.max() < .25
            shaprows.append(dict(file=p.name, n=len(df), maximum_additivity_residual=float(err.max())))
    save(pd.DataFrame(shaprows), 'SHAP加和性核验')
    print('Saved predictions and SHAP verified',flush=True)

def housing_refit(frame):
    labeled=frame.loc[frame.avgprice.notna()].copy().reset_index(drop=True)
    details=[]; newmetrics=[]; foldchecks=[]; independent=[]
    original=pd.read_csv(ORIGINAL/'任务一_房价/扩展模型_逐单元预测.csv')
    gwr=pd.read_csv(ORIGINAL/'任务一_房价/GWR_局部地铁距离斜率.csv')
    for sample, df in [('原口径复跑',labeled),
                       ('排除121个汇总疑点',labeled.loc[~labeled.single_inconsistent].reset_index(drop=True))]:
        for axis in ['东西','南北']:
            # Keep the original full-sample geographic boundaries for sensitivity comparisons.
            folds=(labeled.space_fold.astype(int) if axis=='东西' else pd.qcut(labeled.y_m,5,labels=False))
            foldmap=pd.Series(folds.to_numpy(),index=labeled.cell_id)
            positions=df.cell_id.map(foldmap).to_numpy()
            x=df[PREDICTORS].to_numpy(); y=df.avgprice.to_numpy(); xy=df[['x_m','y_m']].to_numpy()
            pieces=[]
            for fold,(train,test) in enumerate(PredefinedSplit(positions).split()):
                train=remove_buffer(xy,train,test)
                nearest=cKDTree(xy[test]).query(xy[train])[0].min();assert nearest >= 500
                foldchecks.append(dict(sample=sample,axis=axis,fold=fold+1,n_train=len(train),n_test=len(test),
                                       min_train_test_distance_m=float(nearest)))
                models=fitted_models()
                # Baselines and the two main regressions, plus the positional model.
                prediction={k:m.fit(x[train],y[train]).predict(x[test]) for k,m in models.items()}
                model=fitted_models()['XGBoost']
                prediction['XGBoost＋坐标']=model.fit(np.c_[x[train],xy[train]],y[train]).predict(np.c_[x[test],xy[test]])
                if sample=='原口径复跑':
                    k=int(gwr.loc[gwr.axis.eq(axis)&gwr.fold.eq(fold+1),'neighbors'].iloc[0])
                else:
                    inner=[]
                    for block in np.unique(positions[train]):
                        val=train[positions[train]==block]
                        learn=remove_buffer(xy,train[positions[train]!=block],val)
                        inner.append((learn,val))
                    k=choose_parameter((60,120,240,480,800),inner,
                        lambda q,t,v:gwr_predict(x,y,xy,t,v,q)[0],y)
                p,slopes,radii=gwr_predict(x,y,xy,train,test,k)
                prediction['GWR']=p
                if sample=='原口径复跑':
                    for ii in np.linspace(0,len(test)-1,3,dtype=int):
                        target=test[ii]
                        d,idx=cKDTree(xy[train]).query(xy[target],k=min(k+1,len(train)))
                        bw=np.nextafter(d[-1],np.inf);w=np.maximum(1-(d/bw)**2,0)**2
                        center=x[train].mean(0);scale=x[train].std(0)
                        design=np.c_[np.ones(len(train)),(x[train]-center)/scale]
                        A=design[idx]*np.sqrt(w)[:,None];b=y[train[idx]]*np.sqrt(w)
                        coef=np.linalg.lstsq(A,b,rcond=1e-10)[0]
                        val=np.r_[1,(x[target]-center)/scale]@coef
                        err=abs(val-p[ii]);assert err<1e-5
                        independent.append(dict(axis=axis,fold=fold+1,cell_id=int(df.cell_id.iloc[target]),
                                                max_difference=err,weighted_design_condition=float(np.linalg.cond(A))))
                for name,pred in prediction.items():
                    block=pd.DataFrame({'sample':sample,'axis':axis,'fold':fold+1,'model':name,
                        'cell_id':df.cell_id.iloc[test].to_numpy(),'observed':y[test],'predicted':pred})
                    pieces.append(block)
                    if sample=='原口径复跑' and name in ['线性回归','XGBoost','GWR']:
                        ref=original.loc[original.axis.eq(axis)&original.model.eq(name)].set_index('cell_id')
                        err=np.max(np.abs(ref.loc[block.cell_id,'predicted'].to_numpy()-pred))
                        assert err < .25
                print(sample,axis,fold+1,'k',k,flush=True)
            together=pd.concat(pieces,ignore_index=True);details.append(together)
            for name,g in together.groupby('model'):
                assert len(g)==len(df)==g.cell_id.nunique()
                newmetrics.append(dict(sample=sample,axis=axis,model=name,n=len(g),**metrics(g.observed,g.predicted)))
    save(pd.concat(details,ignore_index=True),'复跑与排除疑点逐单元预测')
    save(pd.DataFrame(newmetrics),'复跑与排除疑点房价指标')
    save(pd.DataFrame(foldchecks),'空间折样本量与隔离距离')
    save(pd.DataFrame(independent),'GWR独立加权最小二乘核验')
    print(pd.DataFrame(newmetrics).to_string(index=False),flush=True)

def taxi_audit(data):
    points, audit, conflicts=quality(data)
    results['taxi_raw']=audit
    specs=[('主规则',60,120,200,1,30),('反向状态',60,120,200,0,30),
           ('严格规则',30,100,200,1,15)]
    stats=[]
    for label,gap,speed,radius,state,eventgap in specs:
        usable,segstats=segments(points,gap,speed)
        od,odstats=candidate_trip_table(usable,state,eventgap)
        row=dict(rule=label,**segstats,**odstats)
        if label!='反向状态':
            stop=stop_table(usable,radius)
            row['停留次数']=len(stop)
            row['停留车辆']=stop.taxi_id.nunique()
        if label=='主规则':
            saved_od=pd.read_csv(ORIGINAL/'任务二_轨迹/候选OD_状态1假设.csv')
            saved_stop=pd.read_csv(ORIGINAL/'任务二_轨迹/停留点_200米5分钟.csv')
            pd.testing.assert_frame_equal(od,saved_od,check_dtype=False,rtol=1e-8,atol=1e-6)
            pd.testing.assert_frame_equal(stop,saved_stop,check_dtype=False,rtol=1e-8,atol=1e-6)
            row['100米停留']=len(stop_table(usable,100))
            city,outside=restrict_to_city(stop,districts(data))
            row['界内停留']=len(city)
            gridchecks=[]
            for period,lo,hi in [('07:00–10:00',25200,36000),('17:00–20:00',61200,72000)]:
                selected=city.loc[city.start_s.ge(lo)&city.start_s.lt(hi)]
                basehits=None;inclusivebase=None
                for size,offset in [(500,0),(500,250),(1000,0)]:
                    keys=pd.DataFrame({'ix':np.floor((selected.x_m-offset)/size).astype(int),
                                       'iy':np.floor((selected.y_m-offset)/size).astype(int)})
                    counts=keys.value_counts().rename('n').reset_index().sort_values(
                        ['n','ix','iy'],ascending=[False,True,True]).reset_index(drop=True)
                    assert counts.n.sum()==len(selected)
                    top=set(map(tuple,counts[['ix','iy']].head(10).to_numpy()))
                    # Also include all cells tied with the tenth cell.
                    tiecut=counts.n.iloc[9]
                    inc=set(map(tuple,counts.loc[counts.n.ge(tiecut),['ix','iy']].to_numpy()))
                    hits=set(keys.index[[tuple(a) in top for a in keys.to_numpy()]])
                    inchits=set(keys.index[[tuple(a) in inc for a in keys.to_numpy()]])
                    if basehits is None:basehits=hits;inclusivebase=inchits
                    gridchecks.append(dict(period=period,grid_m=size,offset_m=offset,n=len(selected),
                        top1_n=int(counts.n.max()),top10_cut_n=int(tiecut),
                        top10_including_tied_cells=len(inc),
                        top10_event_jaccard_pct=100*len(hits&basehits)/len(hits|basehits),
                        tied_event_jaccard_pct=100*len(inchits&inclusivebase)/len(inchits|inclusivebase)))
            save(pd.DataFrame(gridchecks),'热点并列名次敏感性补查')
            legal=points.loc[points.valid]
            profile=legal.groupby('taxi_id').size()
            results['taxi_distribution']=dict(median_points=float(profile.median()),min_points=int(profile.min()),
                max_points=int(profile.max()),max_points_vehicle=int(profile.idxmax()),
                high_density_vehicles=int((profile>20000).sum()))
            # Record all four event-boundary rows to check source-line references.
            sample=saved_od.loc[saved_od.taxi_id.eq(36201)].iloc[0]
            wanted=sample[['before_o_line','after_o_line','before_d_line','after_d_line']].to_numpy()
            source=points.set_index('source_line').loc[wanted,['taxi_id','clock','state','lon','lat']].reset_index()
            save(source,'原始OD四条边界记录复核')
        stats.append(row)
        print('taxi',label,row,flush=True)
    save(pd.DataFrame(stats),'轨迹全量复跑统计')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('data',type=Path)
    parser.add_argument('--part',choices=['housing','taxi','all'],default='all');args=parser.parse_args()
    results['environment']={m:importlib.metadata.version(m) for m in ['numpy','pandas','scipy','scikit-learn','xgboost','pyproj','shapely','numba']}
    if args.part in ['housing','all']:
        frame=check_features(args.data);validate_stored_predictions(frame);housing_refit(frame)
    if args.part in ['taxi','all']:taxi_audit(args.data)
    (OUT/f'复核汇总_{args.part}.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=lambda v:v.item()))
    print('ALL REQUESTED CHECKS PASSED',flush=True)

if __name__=='__main__':main()

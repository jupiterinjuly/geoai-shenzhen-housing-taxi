import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, PredefinedSplit
from xgboost import XGBRegressor

from common import (arguments, districts, draw_boundaries, features, file_in,
                    lookup_district, projection, setup_plot, write_csv)

CATEGORIES = {'购物消费': 'shopping', '科教文化': 'education_culture', '医疗保健': 'health'}
PREDICTORS = ['metro_km', *CATEGORIES.values()]


def make_features(data, outlines, output):
    rows = features(data, '*房价*.geojson')
    frame = pd.DataFrame([r['properties'] for r in rows]).rename(
        columns={'centerlon': 'lon', 'centerlat': 'lat'})
    frame.insert(0, 'cell_id', np.arange(1, len(frame) + 1))
    frame['district'] = lookup_district(frame.lon, frame.lat, outlines)
    frame['x_m'], frame['y_m'] = projection.transform(frame.lon, frame.lat)
    frame['single_inconsistent'] = (frame['count'].eq(1) & frame.avgprice.notna() &
                                     frame.minprice.ne(frame.maxprice))
    points = frame[['x_m', 'y_m']].to_numpy()

    stations = pd.DataFrame([
        (r['properties']['stationnames'], *r['geometry']['coordinates'])
        for r in features(data, '*地铁站点*.geojson')
    ], columns=['station', 'lon', 'lat']).drop_duplicates()
    stations['x_m'], stations['y_m'] = projection.transform(stations.lon, stations.lat)
    # Same station name can appear on several lines. One representative location per name.
    representatives = stations.groupby('station')[['x_m', 'y_m']].median()
    stations_by_name = stations.join(representatives, on='station', rsuffix='_median')
    offsets = np.hypot(stations_by_name.x_m - stations_by_name.x_m_median,
                       stations_by_name.y_m - stations_by_name.y_m_median)
    write_csv(stations_by_name.assign(offset_m=offsets), output, '地铁同名记录核查')

    nearest, indices = cKDTree(representatives).query(points)
    frame['metro_km'] = nearest / 1000
    frame['nearest_station'] = representatives.index[indices]
    anchors = []
    for district in ('福田区', '南山区', '宝安区', '龙岗区', '盐田区'):
        candidates = frame.loc[frame.district.eq(district) & frame.avgprice.notna()]
        anchors.append(candidates.index[len(candidates) // 2])
    independent = pd.DataFrame({'cell_id': frame.loc[anchors, 'cell_id'].to_numpy()})
    direct = distance.cdist(points[anchors], representatives.to_numpy())
    independent['station'] = representatives.index[direct.argmin(axis=1)]
    independent['distance_m'] = direct.min(axis=1)
    assert np.allclose(independent.distance_m, nearest[anchors])

    poi = pd.read_csv(file_in(data, '*POI*.csv'))
    poi = poi.drop_duplicates(subset=['名称', '大类', '中类', '经度', '纬度'])
    valid = poi.经度.between(-180, 180) & poi.纬度.between(-90, 90)
    for category, column in CATEGORIES.items():
        locations = poi.loc[valid & poi.大类.eq(category)]
        xy = np.column_stack(projection.transform(locations.经度, locations.纬度))
        frame[column] = cKDTree(xy).query_ball_point(points, 1000, return_length=True)
        independent[column] = (distance.cdist(points[anchors], xy) <= 1000).sum(axis=1)
        assert np.array_equal(independent[column], frame.loc[anchors, column])
    assert independent.station.to_list() == frame.loc[anchors, 'nearest_station'].to_list()
    assert np.isfinite(frame[PREDICTORS]).all().all()
    write_csv(frame.loc[anchors, ['cell_id','district','lon','lat','nearest_station','metro_km',
                                *CATEGORIES.values()]].merge(independent, on='cell_id',
                                  suffixes=('_indexed','_direct')), output, '五个单元特征复核')
    return frame


def fitted_models():
    return {
        '训练均价': DummyRegressor(strategy='mean'),
        '训练中位价': DummyRegressor(strategy='median'),
        '线性回归': LinearRegression(),
        'XGBoost': XGBRegressor(n_estimators=180, max_depth=3, learning_rate=.04,
                                min_child_weight=8, subsample=.85, colsample_bytree=.9,
                                random_state=42, n_jobs=2, objective='reg:squarederror'),
    }


def score(y, prediction):
    return {'MAE': mean_absolute_error(y, prediction),
            'RMSE': np.sqrt(mean_squared_error(y, prediction)),
            'R2': r2_score(y, prediction),
            'bias': np.mean(prediction - y)}


def evaluate(labeled, variant='全部有均价单元'):
    x = labeled[PREDICTORS].to_numpy()
    y = labeled.avgprice.to_numpy()
    xy = labeled[['x_m', 'y_m']].to_numpy()
    folds = labeled.space_fold.to_numpy()
    schemes = {'空间分块': PredefinedSplit(folds), '随机': KFold(5, shuffle=True, random_state=42)}
    rows, predictions = [], []
    for scheme, splitter in schemes.items():
        for fold, (train, test) in enumerate(splitter.split(x)):
            if scheme == '空间分块':
                # Exclude training points within 500 m of any test cell center.
                buffer_distance = cKDTree(xy[test]).query(xy[train])[0]
                train = train[buffer_distance >= 500]
                assert set(folds[train]).isdisjoint(folds[test])
                assert cKDTree(xy[test]).query(xy[train])[0].min() >= 500
            for name, model in fitted_models().items():
                model.fit(x[train], y[train])
                predicted = model.predict(x[test])
                rows.append(dict(variant=variant, split=scheme, fold=fold + 1, model=name,
                                 n_train=len(train), n_test=len(test), **score(y[test], predicted)))
                predictions.append(pd.DataFrame({'variant': variant, 'split': scheme, 'model': name,
                                                 'fold': fold + 1, 'cell_id': labeled.cell_id.iloc[test],
                                                 'true': y[test], 'predicted': predicted}))
    all_predictions = pd.concat(predictions, ignore_index=True)
    totals = []
    for (split, model), group in all_predictions.groupby(['split', 'model'], sort=False):
        assert len(group) == len(labeled) == group.cell_id.nunique()
        totals.append(dict(variant=variant, split=split, model=model, n=len(group),
                           **score(group.true, group.predicted)))
    return pd.DataFrame(rows), all_predictions, pd.DataFrame(totals)


def district_slopes(labeled):
    sample = labeled.loc[labeled.district.isin(['福田区', '南山区'])].copy()
    ranges = sample.groupby('district').metro_km.quantile([.05, .95]).unstack()
    low, high = ranges[.05].max(), ranges[.95].min()
    sample = sample.loc[sample.metro_km.between(low, high)]
    rows = []
    for variant, data in [('全部有均价单元', sample), ('至少两个价格样本', sample.loc[sample['count'] >= 2])]:
        for district, group in data.groupby('district'):
            x = group[PREDICTORS]
            model = LinearRegression().fit(x, group.avgprice)
            tile = ((group.x_m // 2000).astype(int).astype(str) + '_' +
                    (group.y_m // 2000).astype(int).astype(str))
            estimates = []
            for key in tile.unique():
                keep = tile.ne(key)
                estimates.append(LinearRegression().fit(x[keep], group.avgprice[keep]).coef_[0])
            rows.append(dict(variant=variant, district=district, n=len(group), min_km=low, max_km=high,
                             n_tiles=len(estimates), slope_per_km=model.coef_[0],
                             leave_tile_min=min(estimates), leave_tile_max=max(estimates)))
    return pd.DataFrame(rows)


def charts(frame, predictions, summary, slopes, outlines, output):
    plt = setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax in axes:
        draw_boundaries(ax, outlines)
    labeled = frame.loc[frame.avgprice.notna()]
    axes[0].scatter(labeled.x_m / 1000, labeled.y_m / 1000, c=labeled.space_fold,
                    cmap='Dark2', s=6, linewidths=0)
    axes[0].set_title('沿东西方向切分的五个测试片区')
    best = summary.loc[summary.split.eq('空间分块')].sort_values('MAE').iloc[0].model
    selected = predictions.loc[predictions.split.eq('空间分块') & predictions.model.eq(best)].merge(labeled)
    error = (selected.predicted - selected.true) / 10000
    limit = np.abs(error).quantile(.95)
    dots = axes[1].scatter(selected.x_m / 1000, selected.y_m / 1000, c=error,
                           s=7, cmap='RdBu_r', vmin=-limit, vmax=limit, linewidths=0)
    axes[1].set_title('空间分块测试误差 · ' + best)
    fig.colorbar(dots, ax=axes[1], shrink=.6, extend='both',
                 label='预测减观测（原始价格值 ÷ 10,000）')
    fig.savefig(output / '房价空间分块与误差.png', dpi=165)
    plt.close(fig)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6), layout='constrained')
    summary.pivot(index='model', columns='split', values='MAE').plot.barh(
        ax=ax, color=['#a1b7cb', '#245f75'], width=.7)
    ax.set_xlabel('平均绝对误差（原字段单位）')
    ax.set_ylabel('')
    ax.legend(title=None)
    ax.set_title('同一模型在两种测试方式下的表现')
    for index, row in slopes.reset_index(drop=True).iterrows():
        color = '#1b6675' if row.district == '福田区' else '#be6a37'
        bx.hlines(index, row.leave_tile_min, row.leave_tile_max, color=color)
        bx.scatter(row.slope_per_km, index, color=color, zorder=4)
    bx.set_yticks(range(len(slopes)), [r.district + ' · ' + r.variant for r in slopes.itertuples()])
    bx.axvline(0, color='#9da5aa', lw=1)
    bx.set_xlabel('距离每增加 1 公里的均价变化（原字段单位）')
    bx.set_title('点为分区斜率，线为逐块剔除后的范围')
    fig.savefig(output / '房价评估与分区地铁距离斜率.png', dpi=165)
    plt.close(fig)


def main():
    data, output = arguments()
    outlines = districts(data)
    frame = make_features(data, outlines, output)
    labeled = frame.loc[frame.avgprice.notna() & frame.avgprice.gt(0)].copy()
    labeled['space_fold'] = pd.qcut(labeled.x_m, 5, labels=False)
    frame['space_fold'] = labeled.space_fold
    write_csv(frame.drop(columns=['minlon','maxlon','minlat','maxlat']), output, '房价特征与标签')
    folds, predictions, summary = evaluate(labeled)
    fewer_folds, fewer_predictions, fewer_summary = evaluate(
        labeled.loc[labeled['count'].ge(2)], '至少两个价格样本')
    slopes = district_slopes(labeled)
    for name, content in [('房价逐折指标', pd.concat([folds, fewer_folds])),
                          ('房价折外预测', pd.concat([predictions, fewer_predictions])),
                          ('房价汇总指标', pd.concat([summary, fewer_summary])),
                          ('分区地铁距离斜率', slopes)]:
        write_csv(content, output, name)
    charts(frame, predictions, summary, slopes, outlines, output)
    print(summary.to_string(index=False))
    print(slopes.to_string(index=False))


if __name__ == '__main__':
    main()

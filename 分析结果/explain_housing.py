from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import PredefinedSplit
from xgboost import DMatrix

from common import districts, draw_boundaries, setup_plot, write_csv
from housing import PREDICTORS, fitted_models

NAMES = {'metro_km': '到最近地铁站距离', 'shopping': '购物 POI',
         'education_culture': '科教文化 POI', 'health': '医疗 POI'}


def contributions(frame, coordinates, expected):
    x = frame[PREDICTORS].to_numpy()
    y = frame.avgprice.to_numpy()
    xy = frame[['x_m','y_m']].to_numpy()
    splits = PredefinedSplit(coordinates)
    values = np.empty((len(frame), len(PREDICTORS) + 1))
    for train, test in splits.split():
        train = train[cKDTree(xy[test]).query(xy[train])[0] >= 500]
        model = fitted_models()['XGBoost'].fit(x[train], y[train])
        values[test] = model.get_booster().predict(DMatrix(x[test]), pred_contribs=True)
    residual = np.abs(values.sum(axis=1) - expected)
    assert np.allclose(values.sum(axis=1), expected, rtol=0, atol=.20)
    result = frame[['cell_id', 'district', 'count', 'x_m', 'y_m', *PREDICTORS]].copy()
    result['fold'] = coordinates.to_numpy() + 1
    result['observed'] = y
    result['predicted'] = expected
    result['base'] = values[:, -1]
    result['additivity_residual'] = residual
    for i, feature in enumerate(PREDICTORS):
        result['shap_' + feature] = values[:, i]
    return result


def global_chart(wide, order, output):
    plt = setup_plot()
    from matplotlib.colors import Normalize
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 4.1), layout='constrained',
                                 gridspec_kw={'width_ratios': [1, 1.6]})
    rng = np.random.default_rng(42)
    for rank, feat in enumerate(order):
        attr = wide['shap_' + feat].to_numpy() / 10000
        ax.barh(rank, np.abs(attr).mean(), color='#2b6b79', height=.56)
        bx.scatter(attr, rank + rng.uniform(-.24, .24, len(attr)),
                   c=wide[feat].rank(pct=True), cmap='coolwarm', norm=Normalize(0,1),
                   s=6, alpha=.36, lw=0, rasterized=True)
    ax.set_yticks(range(len(order)), [NAMES[feat] for feat in order])
    ax.invert_yaxis()
    ax.set(xlabel='平均绝对 SHAP 值（原价格值 ÷ 10,000）', title='贡献大小')
    bx.set_yticks(range(len(order)), [NAMES[feat] for feat in order])
    bx.invert_yaxis()
    bx.axvline(0, c='#829198', lw=.8)
    bx.set(xlabel='SHAP 值（原价格值 ÷ 10,000）', title='贡献方向与特征取值')
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1), cmap='coolwarm'), ax=bx,
                 ticks=[0, 1], label='特征值的相对大小', shrink=.73).ax.set_yticklabels(['低', '高'])
    fig.suptitle('XGBoost 的折外 TreeSHAP · 东西空间切分', fontsize=16)
    fig.savefig(output/'房价TreeSHAP_总体与贡献分布.png', dpi=180)
    plt.close(fig)


def metro_chart(frame, outlines, output):
    plt = setup_plot()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.7, 4.5), layout='constrained')
    draw_boundaries(ax, outlines)
    attr = frame.shap_metro_km / 10000
    limit = attr.abs().quantile(.98)
    scatter = ax.scatter(frame.x_m / 1000, frame.y_m / 1000, c=attr, s=9, lw=0,
                         cmap='RdBu_r', vmin=-limit, vmax=limit)
    fig.colorbar(scatter, ax=ax, shrink=.63, label='地铁距离 SHAP（原价格值 ÷ 10,000）',
                 extend='both')
    ax.set_title('地铁距离贡献的空间分布')
    selected = frame.loc[frame.district.isin(['福田区', '南山区'])]
    ranges = selected.groupby('district').metro_km.quantile([.05, .95]).unstack()
    low, high = ranges[.05].max(), ranges[.95].min()
    selected = selected.loc[selected.metro_km.between(low, high)].copy()
    selected['bin'] = pd.cut(selected.metro_km, np.linspace(low, high, 7), include_lowest=True)
    stats = selected.groupby(['district','bin'], observed=True).agg(
        n=('cell_id','size'), x=('metro_km','median'), y=('shap_metro_km','median'),
        low=('shap_metro_km', lambda v: v.quantile(.25)),
        high=('shap_metro_km', lambda v: v.quantile(.75))).reset_index()
    for (district, color) in [('福田区','#1b6675'), ('南山区','#b46335')]:
        part = selected.loc[selected.district.eq(district)]
        bx.scatter(part.metro_km, part.shap_metro_km / 10000, s=7, lw=0, alpha=.16, color=color)
        profile = stats.loc[stats.district.eq(district)]
        bx.plot(profile.x, profile.y/10000, '.-', color=color, label=f'{district}（{len(part)}）')
    bx.axhline(0, color='#829198', lw=.8)
    bx.set(xlabel='距所给地铁站点的直线距离（公里）', ylabel='SHAP 值（原价格值 ÷ 10,000）',
           title='共同距离范围内的折外贡献')
    bx.legend(frameon=False)
    fig.savefig(output/'房价TreeSHAP_地铁贡献地图与分区.png', dpi=180)
    plt.close(fig)
    write_csv(stats, output, '房价TreeSHAP_地铁距离分箱')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    folder = args.output/'任务一_房价'
    labeled = pd.read_csv(folder/'房价特征与标签.csv').dropna(subset=['avgprice'])
    original = pd.read_csv(folder/'房价折外预测.csv')
    original = original.loc[original.variant.eq('全部有均价单元') & original.split.eq('空间分块')]
    north = pd.read_csv(folder/'南北空间分块折外预测.csv')
    shap_tables = []
    ranks = []
    for axis, predictions in [('东西',original), ('南北',north)]:
        reference = predictions.loc[predictions.model.eq('XGBoost')].set_index('cell_id').reindex(labeled.cell_id)
        scores = contributions(labeled, reference.fold - 1, reference.predicted.to_numpy())
        scores['axis'] = axis
        shap_tables.append(scores)
        ranks.extend({'axis': axis, 'feature': NAMES[k],
                      'mean_abs_shap': scores['shap_' + k].abs().mean()} for k in PREDICTORS)
    all_scores = pd.concat(shap_tables)
    write_csv(all_scores, folder, '房价TreeSHAP_折外明细')
    write_csv(pd.DataFrame(ranks), folder, '房价TreeSHAP_切分方向重要性')
    order = sorted(PREDICTORS, key=lambda k: shap_tables[0]['shap_' + k].abs().mean(), reverse=True)
    global_chart(shap_tables[0], order, folder)
    metro_chart(shap_tables[0], districts(args.data), folder)
    print(pd.DataFrame(ranks).to_string(index=False))
    print('Maximum reconstruction error:', all_scores.additivity_residual.max())


if __name__ == '__main__':
    main()

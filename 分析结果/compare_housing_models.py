import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import PredefinedSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import DMatrix

from common import arguments, districts, draw_boundaries, setup_plot, write_csv
from housing import PREDICTORS, fitted_models, score


def scaled(regressor):
    return make_pipeline(StandardScaler(), TransformedTargetRegressor(
        regressor=regressor, transformer=StandardScaler()))


def alternatives():
    return {
        '岭回归': [(v, lambda v=v: scaled(Ridge(alpha=v))) for v in (1, 10, 100)],
        'Lasso': [(v, lambda v=v: scaled(Lasso(alpha=v))) for v in (.003, .01, .04)],
        'KNN 回归': [(v, lambda v=v: scaled(KNeighborsRegressor(n_neighbors=v)))
                     for v in (15, 40, 100)],
        '支持向量回归': [(v, lambda v=v: scaled(SVR(C=v, epsilon=.12))) for v in (.5, 2, 8)],
        '回归树': [(v, lambda v=v: DecisionTreeRegressor(max_depth=v, min_samples_leaf=30,
                    random_state=42)) for v in (3, 5, 8)],
        '随机森林': [(v, lambda v=v: RandomForestRegressor(n_estimators=140, min_samples_leaf=v,
                      max_features=.75, random_state=42, n_jobs=2)) for v in (4, 14, 40)],
        'XGBoost（内层选深度）': [(v, lambda v=v: fitted_models()['XGBoost'].set_params(max_depth=v))
                                  for v in (2, 3, 5)],
    }


def remove_buffer(xy, train, test):
    return train[cKDTree(xy[test]).query(xy[train])[0] >= 500]


def gwr_predict(x, y, xy, train, test, neighbors, return_slopes=False):
    # Adaptive bi-square at each test location, using training observations only.
    factor = x[train].std(axis=0)
    center = x[train].mean(axis=0)
    design = np.column_stack((np.ones(len(x)), (x-center)/np.where(factor > 0, factor, 1)))
    k = min(neighbors + 1, len(train))
    d, ix = cKDTree(xy[train]).query(xy[test], k=k)
    sample = train[ix]
    bandwidth = np.nextafter(d[:, -1], np.inf)
    weights = np.maximum(1 - (d / bandwidth[:, None])**2, 0)
    left = design[sample] * weights[..., None]
    right = y[sample] * weights
    # Pseudoinverse also handles a locally rank-deficient design.
    coefs = (np.linalg.pinv(left, rcond=1e-10) @ right[..., None])[..., 0]
    prediction = (coefs * design[test]).sum(axis=1)
    result = (prediction, coefs[:, 1] / factor[0], bandwidth)
    return (*result, coefs[:, 1:] / factor) if return_slopes else result


def choose_parameter(candidates, inner_folds, predictor, y):
    rows = []
    for param in candidates:
        absolute_error = [np.abs(predictor(param, train, valid) - y[valid])
                          for train, valid in inner_folds]
        rows.append((np.concatenate(absolute_error).mean(), param))
    return min(rows)[1]


def compare(frame, axis, axis_folds):
    x = frame[PREDICTORS].to_numpy()
    xy = frame[['x_m','y_m']].to_numpy()
    y = frame.avgprice.to_numpy()
    results, estimates, shap_rows = [], [], []
    for fold, (train, test) in enumerate(PredefinedSplit(axis_folds).split()):
        train = remove_buffer(xy, train, test)
        inner = []
        for block in np.unique(axis_folds[train]):
            validation = train[axis_folds[train] == block]
            fit = remove_buffer(xy, train[axis_folds[train] != block], validation)
            inner.append((fit, validation))
        assert (cKDTree(xy[test]).query(xy[train])[0] >= 500).all()
        assert all(not set(valid).intersection(test) for _, valid in inner)

        def add(name, param, predicted):
            results.append(pd.DataFrame({'axis': axis, 'fold': fold+1, 'model': name,
                                         'parameter': str(param), 'cell_id': frame.cell_id.iloc[test],
                                         'observed': y[test], 'predicted': predicted}))

        for name, model in fitted_models().items():
            add(name, '固定', model.fit(x[train], y[train]).predict(x[test]))
        for name, candidates in alternatives().items():
            constructors = dict(candidates)
            def fit_predict(param, learn, valid):
                return constructors[param]().fit(x[learn], y[learn]).predict(x[valid])
            param = choose_parameter(constructors, inner, fit_predict, y)
            model = constructors[param]().fit(x[train], y[train])
            predicted = model.predict(x[test])
            add(name, param, predicted)
            if name == 'XGBoost（内层选深度）':
                values = model.get_booster().predict(DMatrix(x[test]), pred_contribs=True)
                assert np.allclose(values.sum(axis=1), predicted, atol=.25, rtol=0)
                explained = frame.iloc[test][['cell_id',*PREDICTORS]].copy()
                for i, key in enumerate(PREDICTORS):
                    explained['shap_' + key] = values[:, i]
                shap_rows.append(explained.assign(axis=axis, fold=fold+1, base=values[:, -1],
                                                  predicted=predicted))

        bw = choose_parameter((60, 120, 240, 480, 800), inner,
                              lambda k,t,v: gwr_predict(x,y,xy,t,v,k)[0], y)
        predicted, beta, radius = gwr_predict(x,y,xy,train,test,bw)
        add('GWR', bw, predicted)
        estimates.append(frame.iloc[test][['cell_id','x_m','y_m','district']].assign(
            axis=axis, fold=fold+1, neighbors=bw, radius_m=radius, metro_slope_per_km=beta))
        print(axis, '外层折', fold+1, 'GWR 邻居数', bw, flush=True)
    return [pd.concat(s) for s in (results, estimates, shap_rows)]


def charts(metrics, gwr, outlines, output):
    plt = setup_plot()
    fig, (ax,bx) = plt.subplots(1,2,figsize=(12.3,5.4),layout='constrained',
                              gridspec_kw={'width_ratios':[1,1.25]})
    pivot = metrics.pivot(index='model',columns='axis',values='MAE')
    pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index, ['东西','南北']].plot.barh(
        ax=ax, width=.78, color=['#2a6c7b','#bb774a'])
    ax.set(ylabel='', xlabel='折外 MAE（原字段单位）', title='四项设施特征与常数基准 · 两种空间切分')
    ax.legend(title=None, frameon=False)
    draw_boundaries(bx, outlines)
    panel = gwr.loc[gwr.axis.eq('东西')]
    values = panel.metro_slope_per_km / 10000
    cap = values.abs().quantile(.95)
    dots = bx.scatter(panel.x_m/1000, panel.y_m/1000, c=values, cmap='RdBu_r',
                      vmin=-cap, vmax=cap, s=9, lw=0)
    bx.set_title('GWR 局部地铁距离斜率 · 东西切分')
    fig.colorbar(dots,ax=bx,label='每公里斜率（原价格值 ÷ 10,000）',shrink=.62,
                 extend='both')
    fig.savefig(output/'扩展房价模型与GWR局部系数.png',dpi=175)
    plt.close(fig)


def main():
    data, output = arguments()
    folder = output/'任务一_房价'
    labeled = pd.read_csv(folder/'房价特征与标签.csv').dropna(subset=['avgprice'])
    res = [compare(labeled, axis, labeled.space_fold.to_numpy().astype(int) if axis=='东西'
                   else pd.qcut(labeled.y_m,5,labels=False).to_numpy()) for axis in ('东西','南北')]
    predictions, gwr, explained = [pd.concat(pieces) for pieces in zip(*res)]
    rows = []
    for (axis, name), group in predictions.groupby(['axis','model']):
        assert len(group) == group.cell_id.nunique() == len(labeled)
        rows.append(dict(axis=axis, model=name, n=len(group), **score(group.observed, group.predicted)))
    metrics = pd.DataFrame(rows)
    for label, table in [('扩展模型_逐单元预测',predictions),('扩展模型_空间验证',metrics),
                         ('GWR_局部地铁距离斜率',gwr),('调深度XGBoost_TreeSHAP折外明细',explained)]:
        write_csv(table,folder,label)
    charts(metrics,gwr,districts(data),folder)
    print(metrics.sort_values(['axis','MAE']).to_string(index=False))


if __name__ == '__main__':
    main()

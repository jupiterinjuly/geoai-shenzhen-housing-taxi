from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from xgboost import DMatrix

from common import arguments, setup_plot, write_csv
from compare_housing_models import alternatives, gwr_predict, remove_buffer
from explain_housing import NAMES
from housing import PREDICTORS, fitted_models


def subset_shap(predictor, x, background):
    """Exact enumeration of the four-feature game for a fixed background sample."""
    n = x.shape[1]
    scores = {}
    for size in range(n + 1):
        for keys in combinations(range(n), size):
            if size == 0:
                values = np.full(len(x), predictor(background).mean())
            elif size == n:
                values = predictor(x)
            else:
                combined = np.broadcast_to(background, (len(x), *background.shape)).copy()
                for key in keys:
                    combined[:,:,key] = x[:,key,None]
                values = predictor(combined.reshape(-1,n)).reshape(len(x),len(background)).mean(axis=1)
            scores[frozenset(keys)] = values
    shap = np.zeros_like(x, dtype=float)
    for item in range(n):
        for keyset, values in scores.items():
            if item not in keyset:
                shap[:,item] += (scores[keyset | {item}] - values)/(n * comb(n-1,len(keyset)))
    return shap, scores[frozenset()]


def explain(frame, predictions, winner):
    xy = frame[['x_m','y_m']].to_numpy()
    x = frame[PREDICTORS].to_numpy()
    y = frame.avgprice.to_numpy()
    frames = []
    for axis in ('东西','南北'):
        selected = predictions.loc[predictions.axis.eq(axis) & predictions.model.eq(winner)]
        ref = selected.set_index('cell_id').reindex(frame.cell_id)
        for fold, (train,test) in enumerate(PredefinedSplit(ref.fold.to_numpy()-1).split()):
            train = remove_buffer(xy,train,test)
            param = ref.iloc[test].parameter.iloc[0]
            bg = x[np.random.default_rng(42+fold).choice(train,32,replace=False)]
            if winner == 'GWR':
                # With location held fixed, the fitted local model is linear in the four features.
                prediction, _, _, slopes = gwr_predict(x,y,xy,train,test,int(param),return_slopes=True)
                values = slopes*(x[test]-bg.mean(axis=0))
                base = prediction-values.sum(axis=1)
                method = '固定位置局部线性SHAP'
            else:
                model = (fitted_models()[winner] if param == '固定'
                         else dict(alternatives()[winner])[float(param)]())
                model.fit(x[train],y[train])
                if winner.startswith('XGBoost'):
                    tree = model.get_booster().predict(DMatrix(x[test]),pred_contribs=True)
                    values,base = tree[:,:-1],tree[:,-1]
                    method = 'TreeSHAP'
                else:
                    values,base = subset_shap(model.predict,x[test],bg)
                    method = '固定训练背景的特征子集SHAP'
            expected = ref.predicted.to_numpy()[test]
            assert np.max(np.abs(base+values.sum(axis=1)-expected)) < .25
            result = frame.iloc[test][['cell_id','district',*PREDICTORS]].copy()
            for i,key in enumerate(PREDICTORS):
                result['shap_'+key] = values[:,i]
            frames.append(result.assign(axis=axis, fold=fold+1, model=winner, method=method,
                                         base=base, observed=y[test], predicted=expected))
    return pd.concat(frames)


def figure(result, winner, output):
    plt = setup_plot()
    ranking = pd.DataFrame({axis: {k: s['shap_'+k].abs().mean() for k in PREDICTORS}
                            for axis,s in result.groupby('axis')})
    rank_order = ranking.mean(axis=1).sort_values().index
    fig,(ax,bx) = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    ranked = ranking.loc[rank_order].rename(index=NAMES)[['东西','南北']]
    (ranked/10000).plot.barh(ax=ax,width=.75,color=['#2a6c7b','#bb774a'])
    ax.legend(title=None,frameon=False)
    ax.set(xlabel='平均绝对 SHAP（原价格值 ÷ 10,000）', ylabel='',
           title='总体贡献排序')
    sample = result.loc[result.axis.eq('东西')].copy()
    deviations = (sample.predicted - sample.observed).abs()
    example = sample.loc[(deviations-deviations.median()).abs().idxmin()]
    vals = example[['shap_'+k for k in PREDICTORS]].astype(float).to_numpy()/10000
    bx.barh([NAMES[k] for k in PREDICTORS],vals,color=np.where(vals<0,'#447c9b','#ba674a'))
    bx.axvline(0,c='#829198',lw=.8)
    bx.set(xlabel='对该单元预测的贡献（原价格值 ÷ 10,000）',
           title=f'单元 {int(example.cell_id)} · 测试误差接近中位数')
    bx.text(.02,-.22, f'基准 {example.base:,.0f}   预测 {example.predicted:,.0f}   观测 {example.observed:,.0f}',
            transform=bx.transAxes, fontsize=10)
    fig.suptitle(winner+' 的折外 SHAP 解释',fontsize=16)
    fig.savefig(output/'设施模型中排名第一的GWR_SHAP.png',dpi=178)
    plt.close(fig)
    return ranked


def main():
    _, output = arguments()
    folder = output/'任务一_房价'
    metrics = pd.read_csv(folder/'扩展模型_空间验证.csv')
    ranking = metrics.groupby('model').MAE.mean().sort_values()
    winner = ranking.index[0]
    frame = pd.read_csv(folder/'房价特征与标签.csv').dropna(subset=['avgprice']).reset_index(drop=True)
    predictions = pd.read_csv(folder/'扩展模型_逐单元预测.csv')
    result = explain(frame,predictions,winner).reset_index(drop=True)
    assert (result.groupby('axis').cell_id.nunique() == len(frame)).all()
    ranked = figure(result,winner,folder)
    write_csv(ranked.rename_axis('feature').reset_index(),folder,'设施模型中排名第一的GWR_SHAP重要性')
    write_csv(result,folder,'设施模型中排名第一的GWR_SHAP明细')
    print(ranking.to_string())
    print(result.method.unique(),ranked)


if __name__ == '__main__':
    main()

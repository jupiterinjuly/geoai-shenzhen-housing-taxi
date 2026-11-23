import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from xgboost import DMatrix

from common import arguments, districts, draw_boundaries, setup_plot, write_csv
from compare_housing_models import remove_buffer
from explain_housing import NAMES
from housing import PREDICTORS, fitted_models, score

FEATURES = [*PREDICTORS, 'x_m', 'y_m']
LABELS = {**NAMES, 'x_m':'东向坐标', 'y_m':'北向坐标'}


def explain(frame, outer, axis):
    xy = frame[['x_m','y_m']].to_numpy()
    x = frame[FEATURES].to_numpy()
    y = frame.avgprice.to_numpy()
    records = []
    for fold, (train,test) in enumerate(PredefinedSplit(outer).split()):
        train = remove_buffer(xy,train,test)
        model = fitted_models()['XGBoost'].fit(x[train],y[train])
        predicted = model.predict(x[test])
        attribution = model.get_booster().predict(DMatrix(x[test]),pred_contribs=True)
        assert np.allclose(attribution.sum(axis=1),predicted,atol=.25,rtol=0)
        table = frame.iloc[test][['cell_id','district',*FEATURES]].copy()
        for index,col in enumerate(FEATURES):
            table['shap_'+col] = attribution[:,index]
        records.append(table.assign(axis=axis,fold=fold+1,observed=y[test],
                                    predicted=predicted,base=attribution[:,-1]))
    return pd.concat(records)


def plot_result(table, comparison, outlines, folder):
    plt = setup_plot()
    fig,(ax,bx) = plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    show = comparison.pivot(index='feature',columns='axis',values='mean_abs_shap')[['东西','南北']]
    show = show.loc[show.mean(axis=1).sort_values().index]
    (show/10000).plot.barh(ax=ax,width=.75,color=['#2a6c7b','#bb774a'])
    ax.set(ylabel='',xlabel='平均绝对 SHAP（原价格值 ÷ 10,000）',title='空间切分下的特征贡献')
    ax.legend(title=None,frameon=False)
    draw_boundaries(bx,outlines)
    panel = table.loc[table.axis.eq('东西')]
    error = (panel.predicted-panel.observed)/10000
    cap = error.abs().quantile(.95)
    dots = bx.scatter(panel.x_m/1000,panel.y_m/1000,c=error,cmap='RdBu_r',
                      vmin=-cap,vmax=cap,s=9,lw=0)
    bx.set_title('折外预测误差 · 东西切分')
    fig.colorbar(dots,ax=bx,shrink=.63,label='预测减观测（原价格值 ÷ 10,000）',extend='both')
    fig.suptitle('XGBoost 加入坐标后的 TreeSHAP 与预测误差',fontsize=16)
    fig.savefig(folder/'坐标增强XGBoost_TreeSHAP与误差地图.png',dpi=178)
    plt.close(fig)


def main():
    data, output = arguments()
    folder = output/'任务一_房价'
    frame = pd.read_csv(folder/'房价特征与标签.csv').dropna(subset=['avgprice'])
    table = pd.concat([explain(frame,frame.space_fold.to_numpy().astype(int),'东西'),
                       explain(frame,pd.qcut(frame.y_m,5,labels=False).to_numpy(),'南北')])
    expected = pd.read_csv(folder/'GWR_空间信息与多样本对照.csv')
    expected = expected.loc[expected.model.eq('XGBoost＋坐标对照')].set_index('axis')
    importance = []
    for axis,group in table.groupby('axis'):
        assert group.cell_id.nunique() == len(group) == len(frame)
        assert np.isclose(score(group.observed,group.predicted)['MAE'],expected.loc[axis,'MAE'])
        importance += [dict(axis=axis,feature=LABELS[k],mean_abs_shap=group['shap_'+k].abs().mean())
                       for k in FEATURES]
    importance = pd.DataFrame(importance)
    write_csv(table,folder,'坐标增强XGBoost_TreeSHAP明细')
    write_csv(importance,folder,'坐标增强XGBoost_TreeSHAP重要性')
    plot_result(table,importance,districts(data),folder)
    print(importance.sort_values(['axis','mean_abs_shap'],ascending=[True,False]).to_string(index=False))


if __name__ == '__main__':
    main()

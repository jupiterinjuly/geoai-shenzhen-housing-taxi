import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import PredefinedSplit

from common import arguments, write_csv
from compare_housing_models import choose_parameter, gwr_predict, remove_buffer
from housing import PREDICTORS, fitted_models, score


def local_mean(y, xy, train, test, neighbors):
    distance, idx = cKDTree(xy[train]).query(xy[test], k=min(neighbors+1,len(train)))
    bandwidth = np.nextafter(distance[:,-1], np.inf)
    weight = np.maximum(1-(distance/bandwidth[:,None])**2,0)**2
    return (weight*y[train[idx]]).sum(axis=1)/weight.sum(axis=1)


def audit(frame, label, axis, positions):
    x = frame[PREDICTORS].to_numpy()
    y = frame.avgprice.to_numpy()
    xy = frame[['x_m','y_m']].to_numpy()
    output = []
    for train, test in PredefinedSplit(positions).split():
        train = remove_buffer(xy,train,test)
        inner = []
        for block in np.unique(positions[train]):
            validation = train[positions[train] == block]
            learn = remove_buffer(xy,train[positions[train] != block],validation)
            inner.append((learn,validation))
        if label == '全样本空间信息对照':
            predict = lambda k,t,v: local_mean(y,xy,t,v,k)
            bandwidth = choose_parameter((60,120,240,480,800), inner,predict,y)
            contenders = {'地理局部加权均价': predict(bandwidth,train,test),
                          'XGBoost＋坐标对照': fitted_models()['XGBoost'].fit(
                              np.column_stack((x[train],xy[train])),y[train]).predict(
                              np.column_stack((x[test],xy[test])))}
        else:
            predict = lambda k,t,v: gwr_predict(x,y,xy,t,v,k)[0]
            bandwidth = choose_parameter((60,120,240,480,800),inner,predict,y)
            contenders = {name: fitted_models()[name].fit(x[train],y[train]).predict(x[test])
                          for name in ('XGBoost','线性回归')}
            contenders['GWR'] = predict(bandwidth,train,test)
        for name, values in contenders.items():
            output.append(pd.DataFrame({'model':name,'observed':y[test],'predicted':values}))
    return [dict(axis=axis, sample=label, model=name, n=len(group),
                 **score(group.observed,group.predicted))
            for name, group in pd.concat(output).groupby('model')]


def main():
    _, out = arguments()
    folder = out/'任务一_房价'
    source = pd.read_csv(folder/'房价特征与标签.csv').dropna(subset=['avgprice']).copy()
    source['north_fold'] = pd.qcut(source.y_m,5,labels=False)
    rows = []
    for label, frame in [('全样本空间信息对照',source),
                         ('至少两个价格样本',source.loc[source['count'].ge(2)])]:
        for axis, field in [('东西','space_fold'),('南北','north_fold')]:
            rows.extend(audit(frame,label,axis,frame[field].to_numpy()))
    scores = pd.DataFrame(rows)
    write_csv(scores,folder,'GWR_空间信息与多样本对照')
    print(scores.to_string(index=False))


if __name__ == '__main__':
    main()

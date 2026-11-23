import pandas as pd

from common import arguments, districts, features, lookup_district, projection, write_csv


def housing_check(output):
    from housing import evaluate

    folder = output / '任务一_房价'
    frame = pd.read_csv(folder / '房价特征与标签.csv')
    labeled = frame.loc[frame.avgprice.notna()].copy()
    labeled['space_fold'] = pd.qcut(labeled.y_m, 5, labels=False)
    folds, predictions, summary = evaluate(labeled)
    scores = summary.loc[summary.split.eq('空间分块')].assign(axis='南北')
    original = pd.read_csv(folder / '房价汇总指标.csv')
    original = original.loc[original.variant.eq('全部有均价单元') &
                            original.split.eq('空间分块')].assign(axis='东西')
    write_csv(pd.concat([original, scores]), folder, '空间切分方向对照')
    write_csv(folds.loc[folds.split.eq('空间分块')], folder, '南北空间分块逐折指标')
    write_csv(predictions.loc[predictions.split.eq('空间分块')], folder, '南北空间分块折外预测')
    print(scores.to_string(index=False))


def taxi_check(data, output):
    folder = output / '任务二_轨迹'
    stops = pd.read_csv(folder / '停留点_200米5分钟.csv')
    lon, lat = projection.transform(stops.x_m, stops.y_m, direction='INVERSE')
    outlines = districts(data)
    stops['district'] = lookup_district(lon, lat, outlines)
    city = stops.loc[stops.district.ne('界外或区界')]
    morning = city.loc[city.start_s.between(25200, 36000, inclusive='left')]
    evening = city.loc[city.start_s.between(61200, 72000, inclusive='left')]
    shared = set(morning.taxi_id) & set(evening.taxi_id)
    rows = []
    for period, subset in [('07:00–10:00', morning), ('17:00–20:00', evening)]:
        for sample, eligible in [('时段内全部车辆', subset),
                                 ('两个时段均观测到停留的车辆', subset.loc[subset.taxi_id.isin(shared)])]:
            assert not eligible.empty
            counts = eligible.groupby('district').size().reindex(outlines, fill_value=0)
            assert counts.sum() == len(eligible)
            rows.extend({'sample': sample, 'period': period, 'district': name, 'n': count,
                         'period_n': len(eligible), 'cars': eligible.taxi_id.nunique(),
                         'share_pct': count * 100 / len(eligible)} for name, count in counts.items())
    summary = pd.DataFrame(rows)
    write_csv(summary, folder, '同车对照_分时停留区域占比')
    print(summary.sort_values(['sample', 'period', 'n'], ascending=[True, True, False])
          .groupby(['sample', 'period'], sort=False).head(3).to_string(index=False))


def main():
    data, output = arguments()
    names = {row['properties']['stationnames'] for row in features(data, '*地铁站点*.geojson')}
    assert '岗厦北' in names
    housing_check(output)
    taxi_check(data, output)


if __name__ == '__main__':
    main()

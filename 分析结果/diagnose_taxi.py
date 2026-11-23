from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from common import setup_plot, write_csv
from taxi import quality


def profile(points, output):
    legal = points.loc[points.valid, ['taxi_id', 'second']]
    hourly = legal.assign(hour=legal.second // 3600).groupby('hour').taxi_id.agg(
        observations='size', vehicles='nunique').reindex(range(24), fill_value=0)
    per_car = legal.groupby('taxi_id').size().rename('observations')
    comparable = ~points.barrier
    preceding = points.taxi_id.eq(points.taxi_id.shift())
    use = comparable & comparable.shift(fill_value=False) & preceding
    dt = points.second.diff()
    use &= dt.gt(0)
    step = np.hypot(points.x_m.diff(), points.y_m.diff())
    delta = dt.loc[use]
    velocity = step.loc[use] * 3.6 / delta
    interval_distribution = delta.value_counts().sort_index().rename('pairs')
    velocity_bins = pd.cut(velocity, [0, 30, 60, 90, 120, 150, 300, np.inf],
                           include_lowest=True).value_counts(sort=False).rename('pairs')
    write_csv(hourly.reset_index(), output, '轨迹观测按小时统计')
    write_csv(per_car.reset_index(), output, '轨迹观测按车辆统计')
    write_csv(interval_distribution.rename_axis('gap_s').reset_index(), output, '轨迹相邻时间差分布')
    write_csv(velocity_bins.rename_axis('kmh_interval').reset_index(), output, '轨迹相邻推算速度分布')
    stats = {
        '相邻可比较观测对': len(delta),
        '中位时间间隔（秒）': delta.median(),
        '95%时间间隔（秒）': delta.quantile(.95),
        '时间间隔超过60秒的观测对': delta.gt(60).sum(),
        '时间间隔不超过60秒且推算速度超过120kmh的观测对': (velocity.gt(120) & delta.le(60)).sum(),
        '最低每车点数（合法去重）': per_car.min(),
        '中位每车点数（合法去重）': per_car.median(),
        '最高每车点数（合法去重）': per_car.max(),
    }
    write_csv(pd.DataFrame(stats.items(), columns=['检查项','结果']), output, '轨迹分布诊断摘要')

    plt = setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.5), layout='constrained')
    ax, bx, cx, dx = axes.ravel()
    ax.bar(hourly.index, hourly.observations / 10000, color='#2b6b79')
    min_cars, max_cars = hourly.vehicles.min(), hourly.vehicles.max()
    ax.set(xlabel='时刻（小时）', ylabel='观测点数（万）',
           title=f'合法去重观测的时段覆盖 · 每小时 {min_cars}–{max_cars} 辆车')
    bx.hist(per_car / 1000, bins=22, color='#2b6b79', edgecolor='white')
    bx.axvline(per_car.median()/1000, color='#a55a32', ls='--', label='中位数')
    bx.set(xlabel='每车观测点数（千）', ylabel='车辆数', title='不同车辆的记录量')
    bx.legend(frameon=False)
    show = interval_distribution.head(90)
    cx.step(show.index, show.cumsum() / len(delta) * 100, where='post', color='#2b6b79')
    cx.axvline(60, color='#a55a32', ls='--', lw=1, label='主切段门槛')
    cx.set(xlim=(0, 90), ylim=(0, 101), xlabel='相邻观测间隔（秒）', ylabel='累计观测对（%）',
           title='采样间隔与长断档')
    cx.legend(loc='lower right', frameon=False)
    dx.bar(range(len(velocity_bins)), velocity_bins,
           color=['#2b6b79']*4+['#a55a32']*3)
    dx.set_yscale('log')
    dx.set_xticks(range(len(velocity_bins)), ['0–30','30–60','60–90','90–120','120–150','150–300','>300'])
    dx.set(xlabel='两点直线位移推算速度（km/h）', ylabel='观测对数（对数轴）',
           title='推算速度分布与可疑连接')
    fig.suptitle('出租车轨迹基本描述与质量诊断', fontsize=16)
    fig.savefig(output/'轨迹基本描述与质量诊断.png', dpi=175)
    plt.close(fig)
    return stats


def conflicts_and_od(points, conflicts, output):
    disputed = conflicts.loc[conflicts.state_conflict].reset_index()[['taxi_id','clock']]
    possible = points.loc[points.valid & points.ambiguous].merge(disputed, on=['taxi_id','clock'])
    pair_candidates = possible.groupby(['taxi_id','clock']).filter(lambda g: len(g) == 2)
    grouped = pair_candidates.groupby(['taxi_id','clock'])
    extent = grouped[['x_m','y_m']].max() - grouped[['x_m','y_m']].min()
    distance = np.hypot(extent.x_m, extent.y_m)
    show = distance.loc[distance.between(1000, 5000)]
    key = (show - 2500).abs().idxmin()
    exemplar = grouped.get_group(key).copy()
    exemplar['separation_m'] = distance.loc[key]
    write_csv(exemplar[['source_line','taxi_id','clock','state','speed_field','lon','lat',
                        'separation_m']], output, '轨迹冲突诊断_两条源记录')
    stages = pd.read_csv(output/'候选OD诊断.csv').iloc[0]
    numbers = [stages[k] for k in ['该状态连续段','两次切换完整','候选OD']]
    assert numbers[0] > numbers[1] > numbers[2]
    plt = setup_plot()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.1), layout='constrained')
    xy = exemplar[['x_m','y_m']].to_numpy()
    xy -= xy[0]
    ax.plot(xy[:,0], xy[:,1], color='#a55a32', ls='--', lw=1)
    for i, row in enumerate(exemplar.itertuples()):
        ax.scatter(*xy[i], s=120, color=('#2b6b79','#a55a32')[i], zorder=3,
                   label=f'状态 {row.state} · 原表行 {row.source_line:,}')
    ax.set(xlabel='相对第一条记录的东向距离（米）', ylabel='北向距离（米）',
           title=f'同车同秒样例 · {key[1]} · 相距 {distance.loc[key]/1000:.2f} 公里')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(frameon=False)
    labels = ['状态 1 连续段', '同一片段内两端切换齐全', '通过点数、时长与事件间隔筛选']
    bx.barh(labels, numbers, color=['#9fb1b8','#64929d','#2b6b79'], height=.6)
    for i, n in enumerate(numbers):
        bx.text(n + 550, i, f'{n:,}', va='center')
    bx.invert_yaxis()
    bx.set(xlim=(0, numbers[0] * 1.24), xlabel='连续状态段或候选行程数量',
           title='候选 OD 的筛选损失 · 假定状态 1 为载客')
    fig.savefig(output/'轨迹冲突实例与候选OD筛选.png', dpi=175)
    plt.close(fig)
    return distance.loc[key]


def district_figure(output):
    data = pd.read_csv(output/'同车对照_分时停留区域占比.csv')
    data = data.loc[data['sample'].str.startswith('两个时段')]
    shares = data.pivot(index='district', columns='period', values='share_pct')
    shares = shares.sort_values('07:00–10:00')
    plt = setup_plot()
    fig, ax = plt.subplots(figsize=(8.5,4.9), layout='constrained')
    shares.plot.barh(ax=ax, color=['#2b6b79','#bd7545'], width=.75)
    ax.set(xlabel='占本时段界内停留事件比例（%）', ylabel='',
           title='同一批 278 辆车的早晚停留区域占比')
    ax.legend(title=None, frameon=False, loc='lower right')
    fig.savefig(output/'同车分时停留_区域占比.png', dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output/'任务二_轨迹'
    points, _, conflicts = quality(args.data)
    stats = profile(points, output)
    separation = conflicts_and_od(points, conflicts, output)
    district_figure(output)
    print(stats)
    print('Same-vehicle, same-second case distance (m):', separation)


if __name__ == '__main__':
    main()

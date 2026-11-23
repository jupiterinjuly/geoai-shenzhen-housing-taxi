import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from common import arguments, districts, draw_boundaries, projection, setup_plot, write_csv
from taxi import candidate_trip_table, grid, plot_grid, quality, restrict_to_city, segments

PERIODS = [('07:00–10:00', 25200, 36000), ('17:00–20:00', 61200, 72000)]
ENDPOINTS = [('o','候选上客'), ('d','候选下客')]


def endpoints(trips, end, outlines, start=0, finish=86400):
    subset = trips.loc[trips[end + '_s'].between(start, finish, inclusive='left')]
    events = subset.rename(columns={end+'_x_m': 'x_m', end+'_y_m': 'y_m'})
    return restrict_to_city(events, outlines)


def endpoint_maps(primary, reverse, outlines, folder):
    plt = setup_plot()
    datasets = []
    for period, start, finish in PERIODS:
        for end, label in ENDPOINTS:
            inside, outside = endpoints(primary, end, outlines, start, finish)
            datasets.append((period,label,inside,outside,grid(inside, 500, 0)[1]))
    fig, axes = plt.subplots(2,2,figsize=(11.0,7.0),layout='constrained')
    maximum = max(items[-1].share_pct.max() for items in datasets)
    coverage = []
    for ax, (period,label,inside,outside,table) in zip(axes.ravel(), datasets):
        overlay = plot_grid(ax, table, outlines, 500, maximum)
        ax.set_title(f'{period} · {label} · 界内 {len(inside):,} 个')
        coverage.append(dict(period=period,event=label,inside=len(inside),outside=outside,
                             cars=inside.taxi_id.nunique(), busiest_cells=table.n.iloc[0]))
    fig.colorbar(overlay,ax=axes,shrink=.72,label='占对应时段与事件类型的界内比例（%）')
    fig.suptitle('候选上下客分时栅格 · 假定状态 1 代表载客', fontsize=16)
    fig.savefig(folder/'候选上下客_早晚500米网格.png', dpi=176)
    plt.close(fig)
    write_csv(pd.DataFrame(coverage),folder,'候选上下客_时段覆盖')

    fig, axes = plt.subplots(2,2,figsize=(11.0,7.0),layout='constrained')
    scenarios = []
    for value, trips in [(1,primary),(0,reverse)]:
        for end, label in ENDPOINTS:
            subset, outside = endpoints(trips,end,outlines)
            scenarios.append((value, label, len(subset), outside, grid(subset,500,0)[1]))
    maximum = max(items[-1].share_pct.max() for items in scenarios)
    for ax, (value, label, n, _, table) in zip(axes.ravel(),scenarios):
        overlay = plot_grid(ax,table,outlines,500,maximum)
        ax.set_title(f'假定状态 {value} 为载客 · {label} · 界内 {n:,} 个')
    fig.colorbar(overlay, ax=axes, shrink=.72, label='占对应假设与事件类型的界内比例（%）')
    fig.suptitle('载客编码反向假设下的地图对照', fontsize=16)
    fig.savefig(folder/'候选上下客_编码假设对照.png',dpi=176)
    plt.close(fig)
    write_csv(pd.concat([tab.assign(hypothesized_occupied=value, event=event, inside=n, outside=out)
                         for value,event,n,out,tab in scenarios]), folder,'候选上下客_编码假设网格表')


def flow_maps(primary, outlines, folder):
    plt = setup_plot()
    start, _ = endpoints(primary,'o',outlines)
    stop, _ = endpoints(primary,'d',outlines)
    # Join on the original trip row so both endpoints belong to the same trip.
    trips = start[['district','o_s']].join(stop.district.rename('destination'), how='inner')
    trips = trips.rename(columns={'district':'origin'})
    blocks = []
    fig, axes = plt.subplots(1,2,figsize=(12.2,5),layout='constrained')
    centers = [outlines[name].representative_point() for name in outlines]
    coord = np.array(projection.transform([p.x for p in centers], [p.y for p in centers])).T / 1000
    anchors = dict(zip(outlines,coord))
    for ax, (period, start_s, finish_s) in zip(axes,PERIODS):
        draw_boundaries(ax,outlines)
        active = trips.loc[trips.o_s.between(start_s,finish_s,inclusive='left')]
        counts = active.groupby(['origin','destination']).size().rename('n').reset_index()
        assert counts.n.sum() == len(active)
        intra = counts.origin.eq(counts.destination)
        shown = counts.loc[~intra].sort_values(['n','origin','destination'],
                                              ascending=[False,True,True]).head(8)
        for row in shown.itertuples():
            ax.add_patch(FancyArrowPatch(anchors[row.origin],anchors[row.destination],
                arrowstyle='-|>', shrinkA=6,shrinkB=7,mutation_scale=12,alpha=.74,
                connectionstyle='arc3,rad=.15',linewidth=.7+row.n/20,color='#245f75'))
        for district, (x,y) in anchors.items():
            ax.scatter(x,y,c='#d0864d',s=14,zorder=5)
            ax.text(x,y+.9,district.removesuffix('区'),ha='center',va='bottom',fontsize=9,
                    color='#213840',zorder=7)
        ax.set_title(f'{period} · 双端界内 {len(active):,} 对 · 区内 {counts.loc[intra,"n"].sum():,} 对')
        labels = set(zip(shown.origin,shown.destination))
        blocks.append(counts.assign(period=period, displayed=[(a,b) in labels for a,b
                                                              in zip(counts.origin,counts.destination)]))
    fig.suptitle('按起点时刻划分的候选 OD · 每时段绘制前八条跨区流', fontsize=15)
    fig.supxlabel('箭头为起点区 → 终点区，宽度随配对数增加。区内配对计入数量。假定状态 1 为载客。', fontsize=10)
    fig.savefig(folder/'候选OD_早晚主要跨区流向.png',dpi=176)
    plt.close(fig)
    write_csv(pd.concat(blocks),folder,'候选OD_按时段区际流量')


def main():
    root, output = arguments()
    folder = output/'任务二_轨迹'
    outlines = districts(root)
    primary = pd.read_csv(folder/'候选OD_状态1假设.csv')
    points, _, _ = quality(root)
    usable, _ = segments(points,60,120)
    reverse, _ = candidate_trip_table(usable,0,30)
    old = pd.read_csv(folder/'候选OD诊断.csv')
    assert len(primary) == old.iloc[0]['候选OD'] and len(reverse) == old.iloc[1]['候选OD']
    endpoint_maps(primary,reverse,outlines,folder)
    flow_maps(primary,outlines,folder)
    print('Primary and reversed candidates:',len(primary),len(reverse))


if __name__ == '__main__':
    main()

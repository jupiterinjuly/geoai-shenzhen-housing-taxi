import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
try:
    from numba import njit
except ImportError:
    def njit(function):
        return function

from common import (arguments, districts, draw_boundaries, file_in, lookup_district,
                    projection, setup_plot, write_csv)


def quality(data):
    raw = pd.read_csv(file_in(data, '*Taxi*.csv'))
    raw.columns = ['taxi_id', 'clock', 'lon', 'lat', 'state', 'speed_field']
    raw.insert(0, 'source_line', np.arange(2, len(raw) + 2))
    valid_coordinates = raw.lon.between(-180, 180) & raw.lat.between(-90, 90)
    exact_duplicate = raw.duplicated(subset=list(raw.columns[1:]))
    points = raw.loc[~exact_duplicate].sort_values(['taxi_id', 'clock', 'source_line']).copy()
    points['second'] = pd.to_timedelta(points.clock).dt.total_seconds().astype('int32')
    valid = points.lon.between(-180, 180) & points.lat.between(-90, 90)
    eligible = points.loc[valid]
    same_time = eligible.duplicated(['taxi_id', 'clock'], keep=False)
    groups = eligible.loc[same_time].groupby(['taxi_id', 'clock'])
    conflicting = groups.state.nunique().gt(1)
    conflicts = groups.size().rename('n').to_frame().assign(state_conflict=conflicting)
    points['valid'] = valid
    points['ambiguous'] = points.duplicated(['taxi_id', 'clock'], keep=False)
    points['barrier'] = ~valid | points.ambiguous
    points['x_m'] = np.nan
    points['y_m'] = np.nan
    points.loc[valid, ['x_m', 'y_m']] = np.column_stack(
        projection.transform(points.loc[valid, 'lon'], points.loc[valid, 'lat']))
    audit = {
        '原始记录': len(raw), '车辆': raw.taxi_id.nunique(),
        '空值数量': raw.isna().sum().sum(),
        '非法坐标记录（原表）': (~valid_coordinates).sum(), '完全重复行（原表）': exact_duplicate.sum(),
        '去重与过滤非法坐标后：同车同秒多记录组': len(conflicts),
        '去重与过滤非法坐标后：同车同秒载客标记矛盾组': conflicting.sum(),
        '保守排除的时间冲突或无效坐标记录': points.barrier.sum(),
    }
    assert raw.state.isin([0, 1]).all()
    return points, audit, conflicts


def segments(points, gap, max_speed):
    previous_same_car = points.taxi_id.eq(points.taxi_id.shift())
    current_valid = ~points.barrier
    pair = previous_same_car & current_valid & current_valid.shift(fill_value=False)
    seconds = points.second.diff()
    step = np.hypot(points.x_m.diff(), points.y_m.diff())
    with np.errstate(divide='ignore', invalid='ignore'):
        implied_speed = step * 3.6 / seconds
    connect = pair & seconds.between(1, gap) & implied_speed.le(max_speed)
    points = points.assign(segment=(~connect).cumsum(), gap_s=seconds.where(connect))
    usable = points.loc[current_valid]
    transitions = usable.segment.eq(usable.segment.shift()) & usable.state.ne(usable.state.shift())
    stats = {
        '保留记录': len(usable), '片段数量': usable.segment.nunique(),
        '单点片段数量': usable.segment.value_counts().eq(1).sum(),
        '相邻有效点因间隔过长切断': (pair & seconds.gt(gap)).sum(),
        '相邻有效点因位移过大切断（时间在门槛内）':
            (pair & seconds.between(1, gap) & implied_speed.gt(max_speed)).sum(),
        '片段内状态切换': transitions.sum(),
    }
    return usable, stats


@njit
def detect_stops(segment, seconds, x, y, radius, duration):
    anchors = []
    ends = []
    anchor = 0
    while anchor < len(seconds):
        end = anchor + 1
        while end < len(seconds) and segment[anchor] == segment[end]:
            dx = x[end] - x[anchor]
            dy = y[end] - y[anchor]
            if dx * dx + dy * dy > radius * radius:
                break
            end += 1
        if end - anchor >= 3 and seconds[end - 1] - seconds[anchor] >= duration:
            anchors.append(anchor)
            ends.append(end - 1)
            anchor = end
        else:
            anchor += 1
    return np.array(anchors, dtype=np.int64), np.array(ends, dtype=np.int64)


def stop_table(points, radius):
    segment, secs, x, y = (points[column].to_numpy() for column in
                          ['segment', 'second', 'x_m', 'y_m'])
    start, end = detect_stops(segment, secs, x, y, radius, 300)
    assert np.array_equal(segment[start], segment[end])
    assert (secs[end] - secs[start] >= 300).all()
    n = end - start + 1
    cx = (np.r_[0, x.cumsum()][end + 1] - np.r_[0, x.cumsum()][start]) / n
    cy = (np.r_[0, y.cumsum()][end + 1] - np.r_[0, y.cumsum()][start]) / n
    return pd.DataFrame({
        'taxi_id': points.taxi_id.to_numpy()[start], 'segment': segment[start],
        'start_line': points.source_line.to_numpy()[start], 'end_line': points.source_line.to_numpy()[end],
        'start_s': secs[start], 'end_s': secs[end], 'duration_s': secs[end] - secs[start],
        'n_points': n, 'x_m': cx, 'y_m': cy,
    })


def candidate_trip_table(points, state_value, event_gap):
    segment = points.segment.to_numpy()
    state = points.state.to_numpy()
    run_starts = np.r_[0, np.where((segment[1:] != segment[:-1]) | (state[1:] != state[:-1]))[0] + 1]
    run_ends = np.r_[run_starts[1:] - 1, len(points) - 1]
    choose = state[run_starts] == state_value
    starts, ends = run_starts[choose], run_ends[choose]
    complete = (starts > 0) & (ends < len(points) - 1)
    complete &= segment[np.maximum(starts - 1, 0)] == segment[starts]
    complete &= segment[np.minimum(ends + 1, len(points) - 1)] == segment[ends]
    start, end = starts[complete], ends[complete]
    sec = points.second.to_numpy()
    gap_o, gap_d = sec[start] - sec[start - 1], sec[end + 1] - sec[end]
    criteria = (np.maximum(gap_o, gap_d) <= event_gap) & (end - start >= 2)
    criteria &= (sec[end] - sec[start] >= 60) & (sec[end] - sec[start] <= 7200)
    o, d = start[criteria], end[criteria]
    x, y = points.x_m.to_numpy(), points.y_m.to_numpy()
    trip = pd.DataFrame({
        'taxi_id': points.taxi_id.to_numpy()[o], 'segment': segment[o],
        'before_o_line': points.source_line.to_numpy()[o - 1],
        'after_o_line': points.source_line.to_numpy()[o],
        'before_d_line': points.source_line.to_numpy()[d],
        'after_d_line': points.source_line.to_numpy()[d + 1],
        'o_s': (sec[o - 1] + sec[o]) / 2, 'd_s': (sec[d] + sec[d + 1]) / 2,
        'o_gap_s': sec[o] - sec[o - 1], 'd_gap_s': sec[d + 1] - sec[d],
        'o_x_m': (x[o - 1] + x[o]) / 2, 'o_y_m': (y[o - 1] + y[o]) / 2,
        'd_x_m': (x[d] + x[d + 1]) / 2, 'd_y_m': (y[d] + y[d + 1]) / 2,
        'candidate_occupied_state': state_value,
    })
    assert (trip.o_s < trip.d_s).all() & (trip.o_gap_s.gt(0).all() & trip.d_gap_s.gt(0).all())
    assert np.array_equal(segment[o - 1], segment[d + 1])
    return trip, {
        '候选载客状态': state_value, '该状态连续段': len(starts),
        '被片段边界截断': (~complete).sum(), '两次切换完整': complete.sum(),
        '完整但未通过时间间隔与时长门槛': (~criteria).sum(), '候选OD': len(trip),
        '候选OD涉及车辆': trip.taxi_id.nunique(),
    }


def pilot(points, conflicts, output):
    per_car = conflicts.groupby(level=0).size().reindex(points.taxi_id.unique(), fill_value=0)
    ids = per_car.sort_values().iloc[[20, 200, 380]].index
    sample = points.loc[points.taxi_id.isin(ids)]
    usable, _ = segments(sample, 60, 120)
    examples, _ = candidate_trip_table(usable, 1, 30)
    candidates = examples.groupby('taxi_id').head(1)
    write_csv(candidates, output, '小样_三辆车候选OD')
    fig, axes = setup_plot().subplots(1, 3, figsize=(12.3, 3.8), layout='constrained')
    detail = []
    for ax, row in zip(axes, candidates.itertuples()):
        car = usable.loc[usable.segment.eq(row.segment) & usable.taxi_id.eq(row.taxi_id)]
        selected = car.loc[car.second.between(row.o_s - 90, row.d_s + 90)]
        detail.append(selected[['source_line','taxi_id','clock','state','speed_field','lon','lat']])
        ax.plot(selected.x_m, selected.y_m, color='#9badb3', lw=1, zorder=1)
        for state, group in selected.groupby('state'):
            ax.scatter(group.x_m, group.y_m, s=14, label=f'状态 {state}', zorder=2)
        ax.scatter([row.o_x_m, row.d_x_m], [row.o_y_m, row.d_y_m], s=55, facecolor='white',
                   edgecolors='#202a33', marker='D', zorder=3)
        ax.set_title(f'车辆 {row.taxi_id} · 候选行程')
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.legend(frameon=False, loc='best')
    if detail:
        write_csv(pd.concat(detail), output, '小样_候选行程原始记录')
    fig.savefig(output / '三辆出租车候选行程核查.png', dpi=165)
    return len(candidates)


def plot_stop_example(points, stops, output):
    # Pick a stop near the typical duration with valid observations on both sides.
    typical = stops.duration_s.median()
    ordered = stops.assign(distance_from_median=(stops.duration_s - typical).abs())
    for row in ordered.sort_values('distance_from_median').head(150).itertuples():
        clip = points.loc[points.segment.eq(row.segment)]
        window = clip.loc[clip.second.between(row.start_s - 90, row.end_s + 90)]
        if window.second.min() < row.start_s and window.second.max() > row.end_s:
            break
    inside = window.second.between(row.start_s, row.end_s)
    assert inside.sum() == row.n_points
    assert (np.hypot(window.loc[inside, 'x_m'] - window.loc[inside, 'x_m'].iloc[0],
                     window.loc[inside, 'y_m'] - window.loc[inside, 'y_m'].iloc[0]) <= 200).all()
    write_csv(window.assign(is_stop=inside), output, '小样_停留点原始记录')

    from matplotlib.patches import Circle
    plt = setup_plot()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    x0 = window.loc[inside, 'x_m'].iloc[0]
    y0 = window.loc[inside, 'y_m'].iloc[0]
    ax.add_patch(Circle((0, 0), 200, fill=False, ls='--', color='#607786'))
    ax.plot(window.x_m - x0, window.y_m - y0, lw=1, color='#97a8b0')
    ax.scatter(window.loc[inside, 'x_m'] - x0, window.loc[inside, 'y_m'] - y0, s=12,
               color='#9b552c', label='停留窗口内')
    ax.set(xlabel='距起始点向东（米）', ylabel='距起始点向北（米）', title='停留点的空间范围')
    ax.set_aspect('equal')
    ax.legend(loc='best', frameon=False)
    minutes = (window.second - row.start_s) / 60
    bx.plot(minutes, np.hypot(window.x_m - x0, window.y_m - y0), color='#265e73', lw=1.3)
    bx.axhline(200, color='#607786', ls='--', lw=1)
    bx.axvspan(0, row.duration_s / 60, color='#dab195', alpha=.36)
    bx.set(xlabel='相对停留开始的时间（分钟）', ylabel='距起始点（米）', title='距离与时间门槛')
    fig.savefig(output / '停留点原始轨迹复核.png', dpi=165)
    plt.close(fig)


def grid(events, size, offset):
    ix = np.floor((events.x_m.to_numpy() - offset) / size).astype(int)
    iy = np.floor((events.y_m.to_numpy() - offset) / size).astype(int)
    frame = pd.DataFrame({'ix': ix, 'iy': iy}, index=events.index)
    counts = frame.value_counts(sort=False).rename('n').reset_index().sort_values(
        ['n', 'ix', 'iy'], ascending=[False, True, True], ignore_index=True)
    counts['share_pct'] = counts.n / len(events) * 100
    assert counts.n.sum() == len(events)
    counts['x_center'] = (counts.ix + .5) * size + offset
    counts['y_center'] = (counts.iy + .5) * size + offset
    return frame, counts


def plot_grid(ax, counts, outlines, size, limit):
    draw_boundaries(ax, outlines)
    left = (counts.x_center - size / 2) / 1000
    right = (counts.x_center + size / 2) / 1000
    bottom = (counts.y_center - size / 2) / 1000
    top = (counts.y_center + size / 2) / 1000
    polygons = np.stack((np.column_stack((left, bottom)), np.column_stack((right, bottom)),
                         np.column_stack((right, top)), np.column_stack((left, top))), axis=1)
    return ax.add_collection(PolyCollection(polygons, array=counts.share_pct.to_numpy(),
                                           cmap='YlOrRd', norm=Normalize(0, limit), edgecolor='none'))


def restrict_to_city(events, outlines):
    lon, lat = projection.transform(events.x_m, events.y_m, direction='INVERSE')
    labeled = events.assign(district=lookup_district(lon, lat, outlines), lon=lon, lat=lat)
    city = labeled.loc[labeled.district.ne('界外或区界')].copy()
    return city, len(events) - len(city)


def hotspot_analysis(stop, od, outlines, output):
    plt = setup_plot()
    city, out = restrict_to_city(stop, outlines)
    od_events = []
    for end, label in [('o', '候选起点'), ('d', '候选终点')]:
        event = od.rename(columns={end + '_x_m': 'x_m', end + '_y_m': 'y_m'})
        mapped, od_outside = restrict_to_city(event, outlines)
        od_events.append((label, mapped, od_outside))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    sets = [(name, grid(events, 500, 0)[1], len(events), outside)
            for name, events, outside in od_events]
    shared_limit = max(counts.share_pct.max() for _, counts, _, _ in sets)
    for ax, (name, counts, n, outside) in zip(axes, sets):
        overlay = plot_grid(ax, counts, outlines, 500, shared_limit)
        ax.set_title(f'{name} · 界内 {n:,} 点')
        write_csv(counts.assign(n_in=n, n_out=outside), output, '500米网格_' + name)
    fig.colorbar(overlay, ax=axes, label='占对应界内事件比例（%）', shrink=.75)
    fig.suptitle('假定状态 1 为载客的候选 OD')
    fig.savefig(output / '出租车候选OD格网.png', dpi=165)
    plt.close(fig)

    scopes = [('07:00–10:00', city.loc[city.start_s.between(25200, 36000, inclusive='left')]),
              ('17:00–20:00', city.loc[city.start_s.between(61200, 72000, inclusive='left')])]
    variants = [(500, 0), (500, 250), (1000, 0)]
    results = []
    tables = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.7), layout='constrained')
    period_tables = [(period, events, [(key, *grid(events, *key)) for key in variants])
                     for period, events in scopes]
    limit = max(counts.share_pct.max() for _, _, tables_ in period_tables for _, _, counts in tables_)
    for index, (period, events, periods) in enumerate(period_tables):
        baseline_hits = None
        for ax, ((size, offset), membership, counts) in zip(axes[index], periods):
            overlay = plot_grid(ax, counts, outlines, size, limit)
            ax.set_title(f'{period} · {size} 米' + (' · 平移半格' if offset else ''))
            top = counts.head(10)
            hits = set(membership.reset_index().merge(top[['ix', 'iy']])['index'])
            if baseline_hits is None:
                baseline_hits = hits
            center_lon, center_lat = projection.transform(top.x_center.to_numpy(), top.y_center.to_numpy(),
                                                          direction='INVERSE')
            locations = lookup_district(center_lon, center_lat, outlines)
            results.append(dict(period=period, grid_m=size, offset_m=offset, n=len(events),
                                cars=events.taxi_id.nunique(), top1_n=top.n.iloc[0],
                                top1_lon=center_lon[0], top1_lat=center_lat[0], top1_district=locations[0],
                                top10_pct=top.share_pct.sum(), common_event_pct=100 * len(hits & baseline_hits) /
                                len(hits | baseline_hits)))
            tables.append(counts.assign(period=period, grid_m=size, offset_m=offset))
    fig.colorbar(overlay, ax=axes, label='占本时段界内停留比例（%）', shrink=.72)
    fig.savefig(output / '出租车分时停留热点与稳健性.png', dpi=165)
    plt.close(fig)
    write_csv(pd.concat(tables), output, '分时停留格网明细')
    write_csv(pd.DataFrame(results), output, '停留热点敏感性')
    write_csv(city.groupby('district').size().rename('n').reset_index(), output, '界内停留分区数')
    return {'界内停留': len(city), '界外或区界停留': out}


def main():
    data, output = arguments()
    outlines = districts(data)
    points, audit, conflicts = quality(data)
    audit['已复原候选行程的小样车辆'] = pilot(points, conflicts, output)
    usable, standard = segments(points, 60, 120)
    tighter, strict = segments(points, 30, 100)
    write_csv(pd.DataFrame([dict(rule='主规则_60秒_120kmh', **standard),
                            dict(rule='严格_30秒_100kmh', **strict)]), output, '轨迹片段统计')
    write_csv(usable.groupby('state').speed_field.agg(
        n='size', median='median', mean='mean', zero_share=lambda v: v.eq(0).mean()).reset_index(),
        output, '载客状态编码旁证')
    od, forward = candidate_trip_table(usable, 1, 30)
    _, reverse = candidate_trip_table(usable, 0, 30)
    _, stricter = candidate_trip_table(tighter, 1, 15)
    write_csv(pd.DataFrame([dict(rule='主规则', **forward), dict(rule='反向编码', **reverse),
                            dict(rule='严格切段与事件时间差', **stricter)]), output, '候选OD诊断')
    write_csv(od, output, '候选OD_状态1假设')
    stops = stop_table(usable, 200)
    stops_100 = stop_table(usable, 100)
    strict_stops = stop_table(tighter, 200)
    plot_stop_example(usable, stops, output)
    write_csv(stops, output, '停留点_200米5分钟')
    write_csv(pd.DataFrame([
        ('主规则_200米', len(stops), stops.duration_s.median(), stops.taxi_id.nunique()),
        ('主规则_100米', len(stops_100), stops_100.duration_s.median(), stops_100.taxi_id.nunique()),
        ('严格切段_200米', len(strict_stops), strict_stops.duration_s.median(), strict_stops.taxi_id.nunique()),
        ], columns=['variant','stops','median_duration_s','cars']), output, '停留阈值对照')
    audit.update(hotspot_analysis(stops, od, outlines, output))
    write_csv(pd.DataFrame(audit.items(), columns=['检查项', '数量']), output, '轨迹数据审计')
    examples = conflicts.loc[conflicts.state_conflict].head(6).reset_index()[['taxi_id', 'clock']]
    write_csv(points.merge(examples, on=['taxi_id','clock']).drop(
        columns=['x_m','y_m','valid','ambiguous','barrier']), output, '同车同秒冲突样例')
    print(audit)
    print(forward, reverse, stricter)


if __name__ == '__main__':
    main()

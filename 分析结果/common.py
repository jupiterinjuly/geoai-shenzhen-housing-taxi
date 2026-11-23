import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import contains_xy, union_all
from shapely.geometry import shape

projection = Transformer.from_crs('OGC:CRS84', 'EPSG:32650', always_xy=True)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path, help='解压后的课程数据目录（可包含子目录）')
    parser.add_argument('output', type=Path, help='输出目录')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    return args.data, args.output


def file_in(data, pattern):
    return next(data.rglob(pattern))


def features(data, pattern):
    return json.loads(file_in(data, pattern).read_text(encoding='utf-8'))['features']


def districts(data):
    return {f['properties']['name']: shape(f['geometry'])
            for f in features(data, '*行政区划*.geojson')}


def lookup_district(lon, lat, outlines):
    result = np.full(len(lon), '界外或区界', dtype=object)
    for name, polygon in outlines.items():
        result[contains_xy(polygon, lon, lat)] = name
    return result


def write_csv(frame, output, name):
    frame.to_csv(output / (name + '.csv'), index=False, encoding='utf-8-sig')


def setup_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for font_path in ('/System/Library/Fonts/STHeiti Light.ttc',
                      '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def draw_boundaries(ax, outlines):
    for area in outlines.values():
        for polygon in area.geoms:
            x, y = projection.transform(*polygon.exterior.xy)
            ax.fill(np.array(x) / 1000, np.array(y) / 1000,
                    facecolor='#f5f7f7', edgecolor='#a6b2b4', linewidth=.55)
    ax.set_aspect('equal')
    ax.set_axis_off()

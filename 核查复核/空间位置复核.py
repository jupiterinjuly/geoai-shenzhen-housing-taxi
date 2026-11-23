"""Independent relative-position checks; these do not certify an unknown CRS."""
from pathlib import Path
import argparse,json
import xml.etree.ElementTree as E
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape,LineString
from pyproj import Transformer

def main():
    parser=argparse.ArgumentParser();parser.add_argument('data',type=Path);a=parser.parse_args()
    out=Path(__file__).resolve().parent/'证据'
    project=out.parent.parent
    projection=Transformer.from_crs('OGC:CRS84','EPSG:32650',always_xy=True)
    def read(pattern):return json.loads(next(a.data.rglob(pattern)).read_text())
    def transform_geom(geoms):
        def convert(xy):return np.column_stack(projection.transform(xy[:,0],xy[:,1]))
        return shapely.transform(geoms,convert)
    meta=[]
    for pattern in ['*房价*.geojson','*地铁站点*.geojson','*地铁线路*.geojson','*行政区划*.geojson']:
        layer=read(pattern);geom=np.array([shape(f['geometry']) for f in layer['features']],dtype=object)
        meta.append(dict(layer=pattern,features=len(geom),crs=json.dumps(layer.get('crs'),ensure_ascii=False),
                         invalid=int((~shapely.is_valid(geom)).sum()),bounds=str(shapely.total_bounds(geom).tolist())))
    pd.DataFrame(meta).to_csv(out/'图层坐标声明与范围.csv',index=False,encoding='utf-8-sig')
    station=read('*地铁站点*.geojson');line=read('*地铁线路*.geojson')
    stations=transform_geom(np.array([shape(f['geometry']) for f in station['features']],dtype=object))
    lines=transform_geom(np.array([shape(f['geometry']) for f in line['features']],dtype=object))
    _,d=shapely.STRtree(lines).query_nearest(stations,return_distance=True,all_matches=False)
    summary={'station_line':{'n':len(d),'median_m':float(np.median(d)),'max_m':float(np.max(d))}}
    selected=[f['properties'] for f in station['features'] if f['properties']['stationnames']=='岗厦北']
    (out/'岗厦北原始线路归属.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    raw=pd.read_csv(next(a.data.rglob('*Taxi*.csv')))
    raw.columns=['taxi_id','clock','lon','lat','state','speed']
    clean=raw.drop_duplicates()
    clean=clean.loc[clean.lon.between(-180,180)&clean.lat.between(-90,90)]
    # Sampling independently from original row order, rather than the previous run's sorted table.
    clean=clean.loc[~clean.duplicated(['taxi_id','clock'],keep=False)&clean.speed.gt(0)]
    sampled=clean.sample(n=10000,random_state=42)
    gx,gy=projection.transform(sampled.lon,sampled.lat)
    pts=shapely.points(gx,gy)
    ns={'g':'http://graphml.graphdrawing.org/xmlns'}
    root=E.parse(next(a.data.rglob('*.graphml'))).getroot()
    keys={k.attrib['id']:k.attrib['attr.name'] for k in root.findall('g:key',ns)}
    def attributes(el):return {keys[d.attrib['key']]:d.text for d in el.findall('g:data',ns)}
    nodes={n.attrib['id']:attributes(n) for n in root.findall('g:graph/g:node',ns)}
    geoms=[]
    for edge in root.findall('g:graph/g:edge',ns):
        p=attributes(edge)
        if p.get('geometry'):g=shapely.from_wkt(p['geometry'])
        else:
            u,v=nodes[edge.attrib['source']],nodes[edge.attrib['target']]
            g=LineString([(float(u['x']),float(u['y'])),(float(v['x']),float(v['y']))])
        geoms.append(g)
    roads=transform_geom(np.array(geoms,dtype=object))
    _,distance=shapely.STRtree(roads).query_nearest(pts,return_distance=True,all_matches=False)
    summary['independent_gps_road_sample']={'n':len(distance),'seed':42,
        'sampling':'原始行顺序，合法去重且无同秒多记录，速度字段>0',
        'median_m':float(np.median(distance)),'p90_m':float(np.quantile(distance,.9)),
        'within_50m':int((distance<=50).sum()),'road_edges':len(roads)}
    sampled=sampled.assign(road_distance_m=distance)
    sampled.to_csv(out/'独立抽样GPS道路距离.csv',index=False,encoding='utf-8-sig')
    (out/'空间位置复核汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

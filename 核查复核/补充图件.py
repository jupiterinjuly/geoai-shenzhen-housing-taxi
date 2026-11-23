"""Scientific figures for the reviewed report; all inputs are stored evidence."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'分析结果'))
from common import setup_plot, districts, draw_boundaries, projection

def map_furniture(ax):
    ax.annotate('',xy=(.94,.90),xytext=(.94,.70),xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>',lw=1,color='#333333'))
    ax.text(.94,.95,'N',transform=ax.transAxes,ha='center',va='center',fontsize=11)
    # Map coordinates are UTM kilometres; draw an exact ten-kilometre bar.
    xlo,xhi=ax.get_xlim();ylo,yhi=ax.get_ylim()
    x=xlo+.08*(xhi-xlo);y=ylo+.075*(yhi-ylo)
    ax.plot([x,x+10],[y,y],color='#333333',lw=2,scalex=False,scaley=False)
    ax.text(x+5,y-.035*(yhi-ylo),'10 km',ha='center',va='top',fontsize=9)

def main(data):
    out=HERE/'证据';folder=HERE.parent/'分析结果/任务一_房价'
    plt=setup_plot();outlines=districts(data)
    frame=pd.read_csv(folder/'房价特征与标签.csv')
    valid=frame.loc[frame.avgprice.notna()];missing=frame.loc[frame.avgprice.isna()]
    check=pd.read_csv(out/'同名站点跨源位置独立核验.csv')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,4.1),layout='constrained',gridspec_kw={'width_ratios':[1.1,1]})
    draw_boundaries(ax,outlines)
    ax.scatter(missing.x_m/1000,missing.y_m/1000,c='#b7b7b7',marker='x',s=9,lw=.5,label='缺少均价 424 单元')
    dots=ax.scatter(valid.x_m/1000,valid.y_m/1000,c=valid.avgprice/10000,s=8,lw=0,cmap='viridis',
                    vmin=valid.avgprice.quantile(.02)/10000,vmax=valid.avgprice.quantile(.98)/10000)
    fig.colorbar(dots,ax=ax,label='观测均价（原字段值 ÷ 10,000）',shrink=.65,extend='both')
    ax.set_title('房价与缺失标签的空间覆盖',fontsize=13)
    ax.legend(frameon=False,loc='upper left',fontsize=9)
    map_furniture(ax)
    dd=np.sort(check.difference_m)
    bx.step(dd,np.arange(1,len(dd)+1)/len(dd)*100,where='post',color='#255c80',lw=2)
    bx.set(xlabel='同名站点跨源代表位置差（米）',ylabel='累计比例（%）',title='167 个同名站点的位置对照',ylim=(0,102),xlim=(0,190))
    for q,label in [(.5,'中位数'),(.9,'90 分位')]:
        value=np.quantile(dd,q);bx.plot([0,value,value],[q*100,q*100,0],ls='--',lw=.8,color='#777')
        bx.text(value+3,q*100-9,f'{label} {value:.1f} 米',fontsize=10,color='#333')
    bx.grid(alpha=.15)
    fig.savefig(out/'修订图1_样本覆盖与位置核验.png',dpi=240);plt.close(fig)

    slopes=pd.read_csv(folder/'分区地铁距离斜率.csv')
    gwr=pd.read_csv(folder/'GWR_局部地铁距离斜率.csv')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,3.9),layout='constrained',gridspec_kw={'width_ratios':[1,1.22]})
    draw_boundaries(ax,outlines)
    g=gwr.loc[gwr.axis.eq('东西')];cap=np.abs(g.metro_slope_per_km).quantile(.95)/10000
    dot=ax.scatter(g.x_m/1000,g.y_m/1000,c=g.metro_slope_per_km/10000,cmap='RdBu_r',s=9,lw=0,vmin=-cap,vmax=cap)
    fig.colorbar(dot,ax=ax,label='距离每公里斜率（原值 ÷ 10,000）',shrink=.7,extend='both')
    ax.set_title('GWR 折外局部系数 · 东西切分',fontsize=12);map_furniture(ax)
    order=[('福田区','全部有均价单元'),('南山区','全部有均价单元'),('福田区','至少两个价格样本'),('南山区','至少两个价格样本')]
    labels=[]
    for i,(area,kind) in enumerate(order):
        row=slopes.loc[slopes.district.eq(area)&slopes.variant.eq(kind)].iloc[0]
        color='#2d6b8a' if area=='福田区' else '#b05227'
        bx.hlines(i,row.leave_tile_min/10000,row.leave_tile_max/10000,color=color,lw=2)
        bx.scatter(row.slope_per_km/10000,i,color=color,s=36,zorder=3)
        labels.append(f'{area} · '+('全部' if kind=='全部有均价单元' else '样本数≥2')+f' (n={row.n})')
    bx.axvline(0,color='#777',lw=.9);bx.set_yticks(range(4),labels,fontsize=10);bx.invert_yaxis()
    bx.set(xlabel='地铁距离每增加 1 km 的条件斜率（原值 ÷ 10,000）',title='共同距离窗 0.19–0.85 km · 分区线性回归')
    bx.grid(axis='x',alpha=.15)
    fig.savefig(out/'修订图4_空间系数与分区稳健性.png',dpi=240);plt.close(fig)

if __name__=='__main__':main(Path(sys.argv[1]))

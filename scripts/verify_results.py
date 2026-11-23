"""Data-free checks of archived aggregate results and publication boundaries."""
from pathlib import Path
import csv,json,math,unittest

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'核查复核/证据'

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

class ArchivedResults(unittest.TestCase):
    def test_model_metrics_share_the_same_target_variance(self):
        rows=read_csv(ROOT/'分析结果/任务一_房价/扩展模型_空间验证.csv')
        self.assertEqual(len(rows),24)
        variances=[]
        for row in rows:
            self.assertEqual(int(row['n']),2366)
            self.assertGreaterEqual(float(row['RMSE']),float(row['MAE']))
            variances.append(float(row['RMSE'])**2/(1-float(row['R2'])))
        self.assertTrue(all(math.isclose(v,variances[0],rel_tol=1e-8) for v in variances))

    def test_taxi_record_and_od_conservation(self):
        summary=json.loads((EVIDENCE/'复核汇总_taxi.json').read_text())['taxi_raw']
        self.assertEqual(summary['原始记录']-summary['完全重复行（原表）']-
            summary['保守排除的时间冲突或无效坐标记录'],3699628)
        rows=read_csv(EVIDENCE/'轨迹全量复跑统计.csv')
        for r in rows:
            self.assertEqual(int(r['该状态连续段'])-int(r['被片段边界截断']),int(r['两次切换完整']))
            self.assertEqual(int(r['两次切换完整'])-int(r['完整但未通过时间间隔与时长门槛']),int(r['候选OD']))
        # Optional fields in the archived table were serialized as integral floats.
        self.assertEqual(float(rows[0]['停留次数']),6108)
        self.assertEqual(int(rows[0]['候选OD']),9195)

    def test_same_vehicle_district_counts_and_denominators(self):
        rows=read_csv(ROOT/'分析结果/任务二_轨迹/同车对照_分时停留区域占比.csv')
        rows=[r for r in rows if r['sample'].startswith('两个时段')]
        for period,total in [('07:00–10:00',671),('17:00–20:00',721)]:
            group=[r for r in rows if r['period']==period]
            self.assertEqual(len(group),9)
            self.assertEqual(sum(int(r['n']) for r in group),total)
            self.assertTrue(all(int(r['cars'])==278 and int(r['period_n'])==total for r in group))
            self.assertAlmostEqual(sum(float(r['share_pct']) for r in group),100,places=8)

    def test_archived_independent_audit_checks(self):
        for r in read_csv(EVIDENCE/'历史逐单元预测与指标重算.csv'):
            self.assertLess(float(r['difference']),1e-7)
        for r in read_csv(EVIDENCE/'GWR独立加权最小二乘核验.csv'):
            self.assertLess(float(r['max_difference']),1e-5)
        for r in read_csv(EVIDENCE/'SHAP加和性核验.csv'):
            self.assertLess(float(r['maximum_additivity_residual']),.25)
        for r in read_csv(EVIDENCE/'复跑与排除疑点房价指标.csv'):
            self.assertEqual(int(r['n']),2245 if r['sample'].startswith('排除') else 2366)

    def test_published_files_do_not_contain_raw_records_or_local_paths(self):
        forbidden={'taxi_id','source_line','before_o_line','after_o_line','before_d_line','after_d_line','lon','lat','o_x_m','d_x_m'}
        for p in ROOT.rglob('*'):
            if not p.is_file() or any(part in {'.git','local_runs','__pycache__','.venv'}for part in p.parts):continue
            relative=p.relative_to(ROOT)
            if relative.parts[:2]==('data','raw'):
                self.assertEqual(p.name,'.gitkeep')
            self.assertNotIn(p.suffix,{'.7z','.zip','.geojson','.graphml','.docx','.pdf'})
            if p.suffix=='.csv':
                with p.open(encoding='utf-8-sig') as f:columns=next(csv.reader(f))
                self.assertFalse(forbidden.intersection(columns),str(relative))
            if p.suffix in {'.py','.md','.txt','.json','.yml'}:
                self.assertNotIn('/Users/'+'luoxiaoqin/',p.read_text(),str(relative))
            if p.suffix=='.py':compile(p.read_text(),str(relative),'exec')

if __name__=='__main__':unittest.main(verbosity=2)

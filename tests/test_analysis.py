"""Synthetic reproducibility and boundary/constraint tests, no actual HR data."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_analysis import load_inputs,calculate,allocate_month,check_presentation_numbers
from check_ai_tags import check_tags


class AllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.projects, cls.staff, cls.ops=load_inputs()

    def test_results_match_ppt(self):
        results,assign=calculate(self.cfg,self.projects,self.staff,self.ops,120)
        check_presentation_numbers(results)
        self.assertEqual(len(results),12)
        self.assertEqual(len(assign),6)
        b=results.query('시나리오 == "기본"').set_index('월')
        self.assertEqual(b.loc['2027-01','잔여부족시간'],320)
        self.assertEqual(b.loc['2027-02','잔여부족시간'],180)

    def test_assignments_match_skills_and_limits(self):
        _,assign=calculate(self.cfg,self.projects,self.staff,self.ops,120)
        for (scenario,month),month_ass in assign.groupby(['시나리오','월']):
            self.assertLessEqual(month_ass['배정시간'].sum(),self.ops.loc[month,'ITO전체_가용시간']-self.ops.loc[month,'ITO필수_운영시간']+1e-6)
            for person,person_ass in month_ass.groupby('직원'):
                row=self.staff.set_index('직원').loc[person]
                self.assertLessEqual(person_ass['배정시간'].sum(),row[month]+1e-6)
                self.assertFalse(row['상시운영필수'])
                for _,a in person_ass.iterrows():
                    skill=self.projects.loc[(self.projects['시나리오']==scenario)&(self.projects['월']==month)&(self.projects['프로젝트']==a['프로젝트']),'요구기술'].iloc[0]
                    self.assertEqual(row['표준기술'],skill)

    def test_zero_operations_slack_blocks_redeployment(self):
        ops=self.ops.copy()
        ops['ITO필수_운영시간']=ops['ITO전체_가용시간']
        results,assign=calculate(self.cfg,self.projects,self.staff,ops,120)
        self.assertEqual(results['제약반영_재배치시간'].sum(),0)
        self.assertTrue(assign.empty)

    def test_essential_operations_staff_not_moved(self):
        st=self.staff.copy()
        st['상시운영필수']=True
        results,assign=calculate(self.cfg,self.projects,st,self.ops,120)
        self.assertEqual(results['제약반영_재배치시간'].sum(),0)
        self.assertTrue(assign.empty)

    def test_skill_mismatch_blocks_movement(self):
        st=self.staff.copy()
        st['표준기술']='DATABASE'
        results,assign=calculate(self.cfg,self.projects,st,self.ops,120)
        self.assertEqual(results['제약반영_재배치시간'].sum(),0)
        self.assertTrue(assign.empty)

    def test_more_si_supply_means_zero_gap(self):
        results,assign=calculate(self.cfg,self.projects,self.staff,self.ops,200)
        self.assertEqual(results['초기부족시간'].sum(),0)
        self.assertTrue(assign.empty)

    def test_model_sensitivity(self):
        expected={110:(860,2060),120:(500,1340),130:(260,740)}
        for hours,(base,growth) in expected.items():
            results,_=calculate(self.cfg,self.projects,self.staff,self.ops,hours)
            sums=results.groupby('시나리오')['잔여부족시간'].sum()
            self.assertEqual(sums['기본'],base)
            self.assertEqual(sums['추가수주'],growth)

    def test_duplicate_project_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            for file in (ROOT/'data').iterdir():
                shutil.copy(file,temp/file.name)
            p=temp/'project_demand.csv'
            df=pd.read_csv(p,encoding='utf-8-sig')
            df=pd.concat([df,df.iloc[[0]]],ignore_index=True)
            df.to_csv(p,index=False,encoding='utf-8-sig')
            with self.assertRaisesRegex(ValueError,'중복'):
                load_inputs(temp)

    def test_negative_hours_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            for file in (ROOT/'data').iterdir():
                shutil.copy(file,temp/file.name)
            p=temp/'project_demand.csv'
            df=pd.read_csv(p,encoding='utf-8-sig')
            df.loc[0,'예상업무시간']=-1
            df.to_csv(p,index=False,encoding='utf-8-sig')
            with self.assertRaisesRegex(ValueError,'0 이상'):
                load_inputs(temp)

    def test_changed_input_affects_output(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            for file in (ROOT/'data').iterdir():
                shutil.copy(file,temp/file.name)
            path=temp/'project_demand.csv'
            d=pd.read_csv(path,encoding='utf-8-sig')
            ix=d.index[(d['시나리오']=='기본') & (d['월']=='2027-01') & (d['프로젝트']=='구축 A')][0]
            d.loc[ix,'예상업무시간'] += 120
            d.to_csv(path,index=False,encoding='utf-8-sig')
            cfg,pro,staff,ops=load_inputs(temp)
            result,_=calculate(cfg,pro,staff,ops,120)
            jan=result[(result['시나리오']=='기본')&(result['월']=='2027-01')].iloc[0]
            self.assertEqual(jan['초기부족시간'],480)

    def test_ai_tag_example_format_and_input_match(self):
        self.assertEqual(check_tags(),7)


if __name__=='__main__':
    unittest.main()

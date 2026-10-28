#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic HRM allocation prototype. Not real Hyundai AutoEver staffing data.

Run: python run_analysis.py
Input: data/*.csv, data/config.json
Output: outputs/*.csv (inputs are never overwritten)
Optimization: maximize eligible redeployed person-hours within monthly skill,
individual, residual-project-demand, and ITO operations constraints.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
IN = ROOT / 'data'
OUT = ROOT / 'outputs'
EPS = 1e-6


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def load_inputs(directory=IN):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    projects = pd.read_csv(directory / 'project_demand.csv', encoding='utf-8-sig')
    candidates = pd.read_csv(directory / 'ito_candidates.csv', encoding='utf-8-sig')
    operations = pd.read_csv(directory / 'ito_operations.csv', encoding='utf-8-sig')
    months, scenarios = config['기간'], config['시나리오']
    require(len(months)==6 and len(set(months))==6, '기간은 중복 없는 6개월이어야 합니다')
    require(len(scenarios)==2 and len(set(scenarios))==2, '시나리오 구성이 잘못됐습니다')
    require(config['SI_FTE'] > 0 and config['기준_FTE당_월시간'] > 0, '기준 투입시간/FTE 오류')
    require(all(x > 0 for x in config['민감도_FTE당_월시간']), '민감도 FTE 기준 오류')

    needed_project={'시나리오','월','프로젝트','요구기술','작업설명','예상업무시간'}
    needed_staff={'직원','원문역량','표준기술','상시운영필수',*months}
    needed_ops={'월','ITO전체_가용시간','ITO필수_운영시간'}
    require(needed_project.issubset(projects.columns), '프로젝트 CSV 필수 열 누락')
    require(needed_staff.issubset(candidates.columns), '직원 CSV 필수 열 누락')
    require(needed_ops.issubset(operations.columns), 'ITO CSV 필수 열 누락')
    require(not projects.empty and not candidates.empty, '프로젝트/직원 데이터가 비어 있음')
    require(not projects[list(needed_project)].isnull().any().any(), '프로젝트 데이터 누락값')
    require(not candidates[list(needed_staff)].isnull().any().any(), '직원 데이터 누락값')
    require(not operations[list(needed_ops)].isnull().any().any(), '운영 데이터 누락값')
    require(not projects.duplicated(['시나리오','월','프로젝트']).any(), '중복 프로젝트·월·시나리오')
    require(not candidates.duplicated(['직원']).any(), '중복 직원 ID')
    require(not operations.duplicated(['월']).any(), '중복 ITO 운영 월')
    require(set(projects['시나리오'])==set(scenarios), '시나리오 누락/잘못된 값')
    require(set(projects['월'])==set(months), '프로젝트 월 누락/잘못된 값')
    require(set(operations['월'])==set(months), 'ITO 운영 월 누락/잘못된 값')
    for s in scenarios:
        for m in months:
            require(((projects['시나리오']==s)&(projects['월']==m)).any(), f'{s} {m} 데이터 누락')
    allowed=set(config['허용기술'])
    require(set(projects['요구기술']).issubset(allowed), '프로젝트 기술태그 허용 목록 위반')
    require(set(candidates['표준기술']).issubset(allowed), '직원 기술태그 허용 목록 위반')

    for name, frame, cols in [
        ('프로젝트',projects,['예상업무시간']),
        ('직원',candidates,months),
        ('ITO',operations,['ITO전체_가용시간','ITO필수_운영시간'])]:
        for col in cols:
            values=pd.to_numeric(frame[col], errors='coerce')
            require(values.notna().all() and np.isfinite(values).all() and (values>=0).all(),
                    f'{name}: {col}의 시간은 0 이상의 유한한 숫자여야 함')
            frame[col]=values.astype(float)
    require((operations['ITO필수_운영시간']<=operations['ITO전체_가용시간']).all(),
            'ITO 필수 운영시간이 총가용시간을 초과')
    flags=candidates['상시운영필수'].astype(str).str.strip().str.lower()
    require(flags.isin(['true','false']).all(), '상시운영필수는 True/False만 허용')
    candidates['상시운영필수']=flags.eq('true')
    return config, projects, candidates, operations.set_index('월')


def allocate_month(rows, staff, ops_row, month, si_capacity):
    demand=float(rows['예상업무시간'].sum())
    shortage=max(0.0,demand-si_capacity)
    slack=max(0.0,float(ops_row['ITO전체_가용시간'])-float(ops_row['ITO필수_운영시간']))
    if shortage<=EPS or slack<=EPS:
        return shortage,0.0,[],slack

    # Simplifying assumption: available SI capacity already deployed to projects
    # in proportion to their demand. This does NOT capture real project schedules.
    tasks=[{'프로젝트':r['프로젝트'],'요구기술':r['요구기술'],
            '잔여업무':shortage*float(r['예상업무시간'])/demand}
           for _,r in rows.iterrows() if r['예상업무시간']>0]
    edges=[]
    for _,c in staff.iterrows():
        cap=float(c[month])
        if cap<=EPS or c['상시운영필수']:
            continue
        for j,t in enumerate(tasks):
            if c['표준기술']==t['요구기술']:
                edges.append((str(c['직원']),j,cap))
    if not edges:
        return shortage,0.0,[],slack

    # LP max sum(x_ij), i: staff, j: project, for one month.
    A=[]; b=[]
    for employee in sorted(set(e[0] for e in edges)):
        A.append([float(e[0]==employee) for e in edges])
        b.append(next(e[2] for e in edges if e[0]==employee))
    for j,t in enumerate(tasks):
        A.append([float(e[1]==j) for e in edges])
        b.append(t['잔여업무'])
    A.append([1.0]*len(edges)); b.append(slack)
    solution=linprog(-np.ones(len(edges)),A_ub=np.asarray(A),b_ub=np.asarray(b),
                     bounds=(0,None),method='highs')
    if not solution.success:
        raise RuntimeError(f'LP solver failed: {solution.message}')
    moved=float(solution.x.sum())
    assignments=[{'직원':e[0],'프로젝트':tasks[e[1]]['프로젝트'],
                  '배정시간':round(float(x),6)}
                 for e,x in zip(edges,solution.x) if x>EPS]
    require(moved<=min(shortage,slack)+EPS, 'LP 부족/운영 상한 위반')
    require(all(x>=-EPS for x in solution.x),'음수 배정시간')
    for employee in staff['직원']:
        sub=sum(a['배정시간'] for a in assignments if a['직원']==employee)
        row=staff.loc[staff['직원']==employee].iloc[0]
        require(sub<=row[month]+1e-5,'직원별 투입 상한 위반')
        require(not row['상시운영필수'] or sub<=EPS,'필수 온콜 직원 배정 오류')
    for t in tasks:
        sub=sum(a['배정시간'] for a in assignments if a['프로젝트']==t['프로젝트'])
        require(sub<=t['잔여업무']+1e-5,'프로젝트 잔여업무 상한 위반')
    return shortage,moved,assignments,slack


def calculate(config, projects, staff, operations, fte_hours):
    require(fte_hours>0 and np.isfinite(fte_hours),'FTE당 월시간 오류')
    results=[]; detail=[]
    si_capacity=float(config['SI_FTE'])*float(fte_hours)
    for scenario in config['시나리오']:
        for month in config['기간']:
            rows=projects.loc[(projects['시나리오']==scenario)&(projects['월']==month)]
            shortage,moved,assign,slack=allocate_month(rows,staff,operations.loc[month],month,si_capacity)
            results.append({'시나리오':scenario,'월':month,'예상업무시간':float(rows['예상업무시간'].sum()),
                            'SI투입가능시간':si_capacity,'초기부족시간':round(shortage,3),
                            '제약반영_재배치시간':round(moved,3),
                            '잔여부족시간':round(shortage-moved,3),
                            'ITO운영유지_잔여시간':round(slack-moved,3)})
            for a in assign:
                detail.append({'시나리오':scenario,'월':month,**a})
    result=pd.DataFrame(results)
    detail=pd.DataFrame(detail,columns=['시나리오','월','직원','프로젝트','배정시간'])
    require(len(result)==len(config['시나리오'])*len(config['기간']),'월별 결과 행 수 오류')
    require(np.allclose(result['초기부족시간'],result['제약반영_재배치시간']+result['잔여부족시간']),
            '잔여 부족시간 검증 실패')
    require((result['ITO운영유지_잔여시간']>=-EPS).all(),'운영 유휴시간 음수')
    return result,detail


def summarize(result, hours):
    return (result.groupby('시나리오',sort=False)[['초기부족시간','제약반영_재배치시간','잔여부족시간']]
            .sum().reset_index().assign(FTE당_월시간=hours))


def check_presentation_numbers(result):
    summary=result.groupby('시나리오').sum(numeric_only=True)
    expected={'기본':(600,100,500,8520),'추가수주':(1440,100,1340,10080)}
    for scenario,(gap,moved,left,demand) in expected.items():
        got=summary.loc[scenario]
        require(np.allclose([got['초기부족시간'],got['제약반영_재배치시간'],
                              got['잔여부족시간'],got['예상업무시간']],
                             [gap,moved,left,demand]), f'{scenario} 발표 수치와 불일치')
    basic=result[result['시나리오']=='기본'].set_index('월')
    require(np.isclose(basic.loc['2027-01','제약반영_재배치시간'],40),'1월 40시간 불일치')
    require(np.isclose(basic.loc['2027-02','제약반영_재배치시간'],60),'2월 60시간 불일치')


def main():
    parser=argparse.ArgumentParser(description='가상 프로젝트 인력 수급·재배치 시뮬레이션')
    parser.add_argument('--skip-reference-check', action='store_true',
                        help='입력 CSV를 의도적으로 변경한 경우 PPT 기준 수치 비교 생략')
    args=parser.parse_args()
    config,projects,staff,operations=load_inputs()
    results,assignments=calculate(config,projects,staff,operations,config['기준_FTE당_월시간'])
    # Compare published PPT numbers only for the unchanged synthetic example.
    if not args.skip_reference_check:
        check_presentation_numbers(results)
    sensitivity=pd.concat([summarize(calculate(config,projects,staff,operations,h)[0],h)
                           for h in config['민감도_FTE당_월시간']],ignore_index=True)
    OUT.mkdir(parents=True,exist_ok=True)
    results.to_csv(OUT/'monthly_results.csv',index=False,encoding='utf-8-sig')
    assignments.to_csv(OUT/'allocations.csv',index=False,encoding='utf-8-sig')
    sensitivity.to_csv(OUT/'fte_sensitivity.csv',index=False,encoding='utf-8-sig')
    print('PASS: 가상 모델 실행 및 제약조건 검사')
    print(summarize(results,config['기준_FTE당_월시간']).to_string(index=False))
    print('Outputs:',OUT)


if __name__=='__main__':
    main()

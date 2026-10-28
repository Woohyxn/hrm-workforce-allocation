#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI-tag *format/reference consistency* check, NOT semantic accuracy testing."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'

def check_tags():
    example=json.loads((DATA/'ai_tag_review.json').read_text(encoding='utf-8'))
    projects=pd.read_csv(DATA/'project_demand.csv',encoding='utf-8-sig')
    staff=pd.read_csv(DATA/'ito_candidates.csv',encoding='utf-8-sig')
    allowed=set(example['허용_기술태그'])
    conf=json.loads((DATA/'config.json').read_text(encoding='utf-8'))
    if allowed!=set(conf['허용기술']): raise ValueError('AI 예시/분석모형 태그 목록 불일치')
    record_refs={}
    for _,r in projects.drop_duplicates('프로젝트').iterrows():
        if r['프로젝트']!='추가 수주':
            record_refs[r['프로젝트']]=(r['작업설명'],r['요구기술'])
    for _,r in staff.iterrows(): record_refs[r['직원']]=(r['원문역량'],r['표준기술'])
    rows=example['분류초안']
    if len(rows)!=7 or {r.get('대상') for r in rows}!=set(record_refs):
        raise ValueError('분류초안 7개 대상의 중복/누락')
    for r in rows:
        if not all(r.get(k) for k in ('대상','원문','AI_제안태그','추가확인')):
            raise ValueError('AI 분류 예시 필수 항목 누락')
        if r['AI_제안태그'] not in allowed: raise ValueError('허용되지 않은 기술태그')
        text,tag=record_refs[r['대상']]
        if r['원문']!=text: raise ValueError('AI 분류 예시 원문/입력 데이터 불일치')
        # This only checks whether the example matches predetermined synthetic tags;
        # it is NOT an independent or empirical assessment of AI classification.
        if r['AI_제안태그']!=tag: raise ValueError('분류 예시 태그/시뮬레이션 입력값 불일치')
    return len(rows)

if __name__=='__main__':
    n=check_tags()
    print(f'PASS: {n}개 기술태그 형식·입력 일치 검사 (실제 숙련도 검증은 아님)')

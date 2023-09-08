"""
[Back testing list]

1. 싱글 팩터 
    - Quintile 포트폴리오 테스트
    - 리밸런싱 주기 테스트 (3개월, 6개월, 1년)
    - 가중방식 (Equal, Market cap) 테스트

2. 멀티 팩터
    - Quintile 포트폴리오 테스트
    - 리밸런싱 주기 테스트 (3개월, 6개월, 1년)
    - 가중방식 (Equal, Market cap) 테스트

3. 멀티 팩터 (팩터 랜덤써치 조합)
    - Quintile 포트폴리오 테스트
    - 리밸런싱 주기 테스트 (3개월, 6개월, 1년)
    - 가중방식 (Equal, Market cap) 테스트

4. 멀티 팩터 (팩터 RL 써치 조합)
    - Quintile 포트폴리오 테스트
    - 리밸런싱 주기 테스트 (3개월, 6개월, 1년)
    - 가중방식 (Equal, Market cap) 테스트
"""

import os
import argparse
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from agent import RANDOMSEARCH
from agent import RLSEARCH

all = ['1M', '3M', '6M', '9M', '12M', 
       '12_1M', '12_3M', '12_6M', 
       '12_9M', 'Kratio']

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--number", type=int, default=10)
parser.add_argument("--balance", type=int, default=1000)
parser.add_argument("--algorithm", type=str, default='RL')
parser.add_argument("--quintile", type=int, default=1)
parser.add_argument("--quarter", type=str, default='1Q')
parser.add_argument("--factors", nargs='*', default=all)
args = parser.parse_args()


if __name__ == '__main__':

    config = {'Number': args.number, 
              'Quantile': args.quintile,
              'Balance': args.balance,
              'Quarter': args.quarter,
              'Factors': args.factors,
              'Dim': args.number}
    

    if args.algorithm == 'RL':
        RLsearch = RLSEARCH(config)
        RLsearch.search(10000, '2010', '2015')

        optimal = RLsearch.get_w(False)
        RLsearch.init(optimal.detach().numpy())
        PVs, PFs, TIs, POs, result = RLsearch.test('2016')


    if args.algorithm == 'random':
        randomsearch = RANDOMSEARCH(config)
        randomsearch.search(10000, '2010', '2015')

        optimal = randomsearch.optimal
        randomsearch.init(optimal)
        PVs, PFs, TIs, POs, result = randomsearch.test('2016')


    if args.algorithm == 'test':
        randomsearch = RANDOMSEARCH(config)
        randomsearch.init()
        PVs, PFs, TIs, POs, result = randomsearch.test('2016')


    seed = args.seed
    algo = args.algorithm
    pd.DataFrame(PVs).to_csv(f'result/seed{seed}/PV_{algo}.csv')
    pd.DataFrame(PFs).to_csv(f'result/seed{seed}/PF_{algo}.csv')
    pd.DataFrame(TIs).to_csv(f'result/seed{seed}/TI_{algo}.csv')
    pd.DataFrame(POs).to_csv(f'result/seed{seed}/PO_{algo}.csv')
    pd.DataFrame.from_dict(result, orient='index')\
        .to_csv(f'result/seed{seed}/Me_{algo}.csv')

    

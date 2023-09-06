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


parser = argparse.ArgumentParser()
parser.add_argument("--number", type=int, default=10)
parser.add_argument("--balance", type=int, default=2e7)
parser.add_argument("--algorithm", type=str, default='random')
parser.add_argument("--quintile", type=str, default='top')
parser.add_argument("--weight", type=str, default='eql')
parser.add_argument("--quarter", type=str, default='1Q')
parser.add_argument("--factors", nargs='*', default=[])
args = parser.parse_args()

if __name__ == '__main__':

    config = {'Number': args.number, 
              'Quantile': args.quintile,
              'Balance': args.balance,
              'Quarter': args.quarter,
              'Weight': args.weight,
              'Factors': args.factors,
              'Dim': 14}
    
    if args.algorithm == 'RL':
        RLsearch = RLSEARCH(config)
        RLsearch.search(50000)

        optimal = RLsearch.get_w(False)
        RLsearch.init(optimal.detach().numpy())
        PVs, PFs, TIs, POs, result = RLsearch.test()

    if args.algorithm == 'random':
        randomsearch = RANDOMSEARCH(config)
        randomsearch.search(50000)

        optimal = randomsearch.optimal
        randomsearch.init(optimal)
        PVs, PFs, TIs, POs, result = randomsearch.test()
    

    pd.DataFrame(PVs).to_csv(f'result/PV_{args.algorithm}.csv')
    pd.DataFrame(PFs).to_csv(f'result/PF_{args.algorithm}.csv')
    pd.DataFrame(TIs).to_csv(f'result/TI_{args.algorithm}.csv')
    pd.DataFrame(POs).to_csv(f'result/PO_{args.algorithm}.csv')

    

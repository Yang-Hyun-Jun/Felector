import pandas as pd
import numpy as np

class FactorManager:

    """
    투자 유니버스가 되는 종목들의 
    Quality + Value Factor 데이터를 관리한다.

    [Quality (Profitability)]
    GPOA, CFOA, GMAR, ROE, ROA

    [Value]
    PER, PBR, PSR, PCR, EPS

    [Mid Momentum]
    3M, 6M, 9M, 12M
    """

    path1 = 'data/factors_kr.csv'
    path2 = 'data/price_kr.csv'
    path3 = 'data/kospi.csv'
    path4 = 'data/cap_kr.csv'

    def __init__(self):    
        """
        all: 전 종목의 모든 팩터의 값을 담고 있는 데이터프레임
        price: 전 종목의 종가 값을 담고 있는 데이터프레임
        kospi: 코스피 지수 종가 값을 담고 있는 데이터프레임
        
        [Ex: all]
        종목코드  기준일       팩터   값
        000990  2016-03-31  CFOA  0.047071
        000990  2016-03-31  EPS   613.9786
        ...     ...         ...     ...
        035000  2023-06-31  ROA   0.008364
        """    
        self.all = pd.read_csv(self.path1, index_col=0, dtype={'종목코드':str})
        self.price = pd.read_csv(self.path2, index_col=0)
        self.kospi = pd.read_csv(self.path3, index_col=0)
        self.capit = pd.read_csv(self.path4, index_col=0)
    
    def get_FactorData(self, name:str) -> pd.DataFrame:
        """
        하나의 팩터 이름을 받아, 
        해당 팩터의 전 종목에 대한 값 데이터를 리턴

        
        [Ex: self.get_FactorData('PER')]
        종목코드      000990  001040  001230   001250  ... 
        기준일
        
        2016-03-31  29.3984 7.6998  17.3678  95.3482 ...   
        2016-06-30  41.3820 5.9536  32.9001  75.9153 ...
        ...
        2022-06-30 
        """

        factor_data = self.all[self.all['팩터'] == name][['종목코드', '기준일', '값']]
        factor_data = factor_data.pivot(index='기준일', columns='종목코드', values='값')
        return factor_data
    
    def get_ScoreEACH(self, date:str) -> pd.DataFrame:
        """
        특정 Date에서 각 종목들의 팩터별 스코어 데이터 리턴

        
        [Ex: self.get_ScoreEACH('2022-12-31')]
        팩터       CFOA   EPS   GMAR   ...   ROE
        종목코드
        000990    1.0    9.0   41.0   ...   45.0
        001040    13.0   5.0   13.0   ...   1.0 
        ...
        001520    8.0    16.0  20.0   ...   13.0
        """

        values = self.all[self.all['기준일'] == date][['종목코드', '팩터', '값']]
        values = values.pivot(index='종목코드', columns='팩터', values='값')

        is_ascending = {
            'PCR': False, 'PER': False, 
            'PSR': False,  'PBR': False,
            'CFOA': True, 'GMAR': True, 
            'GPOA': True, 'EPS': True,
            'ROE': True, 'ROA': True, 
            '3M': True, '6M': False,
            '9M': False, '12M': True,
            }  
        
        rankin_func = lambda Series: Series.rank(ascending=is_ascending[Series.name], method='first')
        weight_func = lambda Series: Series * self.weight_dict[Series.name]
        minmax_func = lambda Series: (Series - min(Series)) / (max(Series) - min(Series)) + 1

        factor_score = values.apply(rankin_func)
        factor_score = factor_score.apply(minmax_func)
        factor_score = factor_score.apply(weight_func)
        return factor_score
    
    def get_RankALL(self, factors:list=[]):
        """
        멀티팩터 또는 싱글팩터 점수를 합하여 토탈 랭킹 데이터 리턴

        
        [Ex: self.get_RankALL()]
        종목코드    000990  001040  001230   001250  ... 
        기준일
        
        2016-03-31  9.0     7.0     17.0    2.0 ...   
        2016-06-30  9.0     6.0     1.0     51.0 ...
        ...
        2022-06-30 
        """

        factors = [
            'GPOA', 'CFOA', 'GMAR', 'ROE', 'ROA', 
            'PER', 'PBR', 'PSR', 'PCR', 'EPS',
            '3M', '6M', '9M', '12M',
            ] if not factors else factors
        
        dates = self.price.index
        rank_all = [self.get_ScoreEACH(date)[factors].sum(axis=1).\
                    rank(method='first', ascending=False) for date in dates]
        rank_all = pd.concat(rank_all, axis=1).transpose()
        rank_all.index = dates
        return rank_all
    
    def set_w(self, value=None):
        """
        각 팩터의 가중치를 결정하는 함수
        """

        weight = {
            'PCR': 1, 'PER': 1, 
            'PSR': 1,  'PBR': 1,
            'CFOA': 1, 'GMAR': 1, 
            'GPOA': 1, 'EPS': 1,
            'ROE': 1, 'ROA': 1, 
            '3M': 1, '6M': 1,
            '9M': 1, '12M': 1,
            }
        
        self.weight_dict = weight if value is None else \
            dict(zip(weight.keys(), value))
        

    
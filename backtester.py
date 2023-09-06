import numpy as np
import pandas as pd

from object import Portfolio
from object import Order
from datetime import datetime
from metrics import Metrics
from factormanager import FactorManager

st = datetime.strptime

class BackTester(Metrics, FactorManager):
    def __init__(self, config):
        FactorManager.__init__(self)
        
        self.config = config
        self.init()

    def init(self, FactorWeight:np.array=None):
        """
        백테스트에 필요한 랭킹 테이블 로드
        FactorWeight: 각 팩터 스코어의 가중치

        [Ex: FactorWeight]
        FactorWeight = [0.12, 0.23, ..., 0.05]
        """
        self.set_w(FactorWeight)
        self.rank_all = self.\
            get_RankALL(self.config['Factors'])
        
    def act(self, index:int, n:int, q='top', w='eql') -> np.array:
        """
        특정 타임스텝에서 팩터 랭킹에 따른, 티커 리스트와 가중 벡터 리턴

        index: 데이터 타임스텝
        n: 포트폴리오 종목수
        q: 포트폴리오 분위 (top, btm)
        w: 포트폴리오 가중방식
        """

        sort = {'top':1, 'btm':-1}
        universe = self.price
        rank_all = self.rank_all
        capital = self.capit

        tickers_all = universe.columns.to_numpy()        
        tickers = tickers_all[np.argsort(rank_all.iloc[index].to_numpy())[::sort[q]][:n]]

        capits = capital.iloc[index][tickers].values
        # market cap weighted
        weight_cap = capits / np.sum(capits)
        # equal weighted 
        weight_eql = np.ones(n) / n 
        # market cap weight / equal weight
        if w=='cap':
            weight = weight_cap
        if w=='eql':
            weight = weight_eql

        return tickers, weight 


    def test(self):
        """ 
        [Return list]
        1. PVs: 타임스텝별 포트폴리오 벨류
        2. PFs: 타임스텝별 포트폴리오 수익률
        3. TIs: 타임스텝별 포트폴리오 티커
        4. POs: 타임스텝별 포트폴리오 비중
        5. result: 메트릭 값 포함 딕셔너리
        """
        PVs = []
        PFs = []
        POs = []
        TIs = []

        universe = self.price
        rank_all = self.rank_all
        act = self.act

        profitloss = 0
        balance = 0

        # 초기 평가 금액
        portfolio_value = \
            self.config['Balance']
        # 초기 투자 금액 
        init_balance = \
            self.config['Balance']
        # 포트폴리오 퀀타일
        q = self.config['Quantile']
        # 포트폴리오 가중 방식
        w = self.config['Weight'] 
        # 포트폴리오 투자 자산수
        n = self.config['Number'] 
        # 포트폴리오 리밸런싱 주기
        Q = self.config['Quarter'] 

        freq = {'1Q': [3, 6, 9, 12], '2Q': [3, 9], '4Q': [12]}

        for i in range(0, len(universe)):

            ticker_old, weight_old = act(i, n, q, w)
            p_old = Portfolio(ticker_old, weight_old) if i == 0 else p_old
            price_old = universe.iloc[i][p_old.ticker].values if i == 0 else price_old

            POs.append(p_old.weight)
            TIs.append(p_old.ticker)
            PVs.append(portfolio_value)
            PFs.append(profitloss)
            
            # 여기는 get_price 함수로 (인자는 ticker 받도록)
            price_old = universe.iloc[i-1][p_old.ticker].values
            price_now = universe.iloc[i][p_old.ticker].values

            # 다음 타임 스텝에서 가격 변동으로 인한 포트폴리오 변화
            ratio = (price_now - price_old) / price_old
            ratio = np.where(np.isnan(ratio), np.float64(-0.99), ratio)
            weight_now = p_old.weight * (1+ratio) 
            weight_now = weight_now / np.sum(weight_now)

            p_old.update_weight(weight_now)
            
            # 다음 타입 스텝에서 가격 변동으로 인한 포트폴리오 평가금액, 수익률 계산
            portfolio_value = np.dot(portfolio_value * p_old.weight, 1+ratio)
            profitloss = ((portfolio_value / init_balance) -1) * 100

            # Desired Portfolio
            check = st(rank_all.index[i], '%Y-%m-%d').month in freq[Q]
            ticker, weight = act(i, n, q, w) if check else (p_old.ticker, p_old.weight)
            p_new = Portfolio(ticker, weight)

            """
            Order 계산
            """
            # Gap 계산 대상
            gap_ticker = p_old.ticker[np.isin(p_old.ticker, p_new.ticker)] 

            # Gap 사이즈
            gap_size = p_new.weight[np.isin(p_old.ticker, p_new.ticker)] - \
                p_old.weight[np.isin(p_old.ticker, p_new.ticker)]
            
            # 매도 대상
            sell_ticker = p_old.ticker[~ np.isin(p_old.ticker, p_new.ticker)]

            # 매도 대상 사이즈
            sell_size = -p_old.weight[~ np.isin(p_old.ticker, p_new.ticker)]

            # 매수 대상
            buy_ticker = p_new.ticker[~ np.isin(p_new.ticker, p_old.ticker)]

            # 매수 대상 사이즈
            buy_size = p_new.weight[~ np.isin(p_new.ticker, p_old.ticker)]

            # 오더
            order = Order()

            gap_order = (gap_ticker, gap_size)
            sell_order = (sell_ticker, sell_size)
            buy_order = (buy_ticker, buy_size)

            order.append(*gap_order)
            order.append(*sell_order)
            order.append(*buy_order)
                
            # 보유하고 있는 종목과 보유할 종목을 combine 해놓기
            combine = {}.fromkeys(order.ticker, 0.0)
            combine.update(p_old.dict)
            weight = np.fromiter(combine.values(), dtype=np.float64)
            
            """
            거래 
            """
            CHARGE = 0.000 #0.001
            TEX = 0.0000 #0.0025
            FEE = 0.0

            sell_cost = CHARGE + TEX
            buy_cost = CHARGE

            action = order.size

            # 매도 주문부터
            sell_ind = np.where( (action < 0) )[0]
            weight[sell_ind] += action[sell_ind]
            sell_moneys = portfolio_value * abs(action[sell_ind]) * (1.0-sell_cost)
            sell_amount = np.sum(sell_moneys) 
            balance += sell_amount
            FEE += sell_amount * sell_cost

            # 매수 주문 처리
            buy_ind = np.where( (action > 0) )[0]
            buy_moneys = portfolio_value * action[buy_ind] * (1.0+buy_cost)
            buy_amount = np.sum(buy_moneys) 

            allocation = buy_moneys / buy_amount

            buy_fee = balance * (buy_cost/(1+buy_cost)) 
            feasible_buy_moneys = (balance - buy_fee) * allocation 
            feasible_buy_amount = np.sum(feasible_buy_moneys)
            feasible_buy_action = feasible_buy_moneys / portfolio_value
            FEE += feasible_buy_amount * buy_cost # (= buy_fee)

            weight[buy_ind] += feasible_buy_action
            weight = weight / np.sum(weight)

            portfolio_value -= FEE
            balance -= feasible_buy_amount 

            p_old = Portfolio(order.ticker[weight>0], weight[weight>0])
        
        PV = round(portfolio_value, 2)
        FP = round(profitloss, 2)
        AL = round(self.get_alpha(PVs), 4)
        SR = round(self.get_sr(PVs), 4)
        IC = round(self.get_rankIC(q), 4)
        MDD = round(self.get_mdd(PVs), 4)

        # print('=====================')
        # print(f'[BackTesting Done]')
        # print(f'(n:{n}, q:{q}, w:{w}, Q:{Q})')
        # print(f'Portfolio Value: {PV}')
        # print(f'Final Profitloss: {FP} %')
        # print(f'Alpha: {AL}')
        # print(f'MDD: {MDD}')
        # print(f'SR: {SR}')
        # print(f'IC: {IC}')
        # print('=====================')

        Result = {'alpha': AL, 'sharpe': SR,
                  'rankic': IC, 'mdd': MDD}
        
        return PVs, PFs, TIs, POs, Result    
        
        


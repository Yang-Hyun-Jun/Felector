import pandas as pd
import numpy as np
from factormanager import FactorManager

class Metrics:
    
    def get_mdd(self, pvs:list):
        df = pd.DataFrame(pvs)
        premaxs = df.cummax()
        drawdowns = (1-df / premaxs) * 100
        mdd = drawdowns.max().iloc[0]
        return mdd

    def get_sr(self, pvs:list):
        free = (0.04) / 4
        pvs = np.array(pvs)
        pct = (pvs[1:] - pvs[:-1]) / pvs[:-1]
        ratio = np.mean(pct - free) / np.std(pct)
        return ratio

    def get_rankIC(self, q=1):
        n = self.config['Number']
        rank = self.rank_all
        rate = self.universe.pct_change().iloc[1:].rank(ascending=False)
        rank = rank[(rank <= n*q) & (rank > n*(q-1))]
        coeffs = []

        for i in range(len(rate)):
            data1 = rank.iloc[i].dropna()
            data2 = rate.iloc[i][data1.index.to_numpy()]
            coeff = np.abs(np.corrcoef(data1, data2)[0,1])
            coeffs.append(coeff)

        RankIC = np.mean(coeffs)
        RankIC = max(0, RankIC)
        return RankIC
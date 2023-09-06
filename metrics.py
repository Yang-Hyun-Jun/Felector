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

    def get_rankIC(self, q='top'):
        n = self.config['Number']
        rank = self.rank_all
        rate = self.price.pct_change().iloc[1:].rank(ascending=False)

        coeffs = []
        sort = {'top':1, 'btm':-1}

        for i in range(len(rate)):
            rank_data = rank.iloc[i]
            rate_data = rate.iloc[i]

            data1 = rank_data[np.argsort(rank_data.to_numpy())[::sort[q]][:n]]
            data2 = rate_data[data1.index.to_numpy()]
            coeff = np.abs(np.corrcoef(data1, data2)[0,1])
            coeffs.append(coeff)

        RankIC = np.mean(coeffs)
        return RankIC

    def get_alpha(self, pvs:list):
        R_m = self.kospi.pct_change().iloc[1:].values
        R_p = pd.DataFrame(pvs).pct_change().iloc[1:].values
        R_f = 0.04 / 4

        excess_Rm = (R_m - R_f).reshape(-1)
        excess_Rp = (R_p - R_f).reshape(-1)

        _, alpha = np.polyfit(excess_Rm, excess_Rp, 1)
        return alpha
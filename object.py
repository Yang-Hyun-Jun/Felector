import numpy as np 

from typing import List
from typing import Tuple
from typing import Iterable

class Portfolio:
    def __init__(self, ticker:Iterable[str], weight:Iterable[float]):
        self.ticker = np.array(ticker)
        self.weight = np.array(self.__con(weight))
        self.dict = self.__get_dict()
    
    def update_weight(self, weight:Iterable[float]):
        self.weight = np.array(self.__con(weight))
        self.dict = self.__get_dict()

    def update_ticker(self, ticker:Iterable[str]):
        self.ticker = np.array(ticker)
        self.dict = self.__get_dict()

    def __repr__(self) -> str:
        return f"{self.dict}"

    def __len__(self) -> int:
        return len(self.dict)

    def __get_dict(self) -> dict:
        return dict(zip(self.ticker, self.weight))
    
    def __con(self, w):
        # if np.sum(w, dtype=np.float64) != 1.0:
        #     raise Exception('Sum to One Error')
        return w 
    
class Order: 
    def __init__(self):
        self.ticker = np.array([])
        self.size = np.array([]) 
        self.dict = self.__get_dict()

    def append(self, ticker:Iterable[str], size:Iterable[float]):
        self.ticker = np.append(self.ticker, ticker)
        self.size = np.append(self.size, size)
        self.dict = self.__get_dict()

    def __repr__(self) -> str:
        return f"{self.dict}"
    
    def __get_dict(self) -> dict:
        return dict(zip(self.ticker, self.size))
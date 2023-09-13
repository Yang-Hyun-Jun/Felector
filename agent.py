import torch
import random
import torch.nn as nn 
import numpy as np

from collections import deque
from backtester import BackTester
from torch.optim import Adam
from torch.optim import SGD
from torch.nn import MSELoss
from network import Mask
from network import Rnet

<<<<<<< HEAD
torch.set_printoptions(sci_mode=False)

=======
>>>>>>> 9a6069f428728bea260c2ab2e692f32008bcbb0e

class RLSEARCH(BackTester):
    def __init__(self, config):
        BackTester.__init__(self, config)

        dim = config['Dim']
        self.mnet = Mask(dim)
        self.rnet = Rnet(dim)
        self.mse = MSELoss()

        self.opt_r = Adam(self.rnet.parameters(), lr=1e-4)
        self.opt_a = Adam(self.mnet.parameters(), lr=1e-3)
    
    def save(self, path):
        torch.save(self.mnet.state_dict(), path)
        torch.save(self.rnet.state_dict(), path)

    def get_w(self, noise=True):
        """
        Policy로부터 팩터 가중치 샘플링
        """
        return self.mnet.sample(noise)

    def get_r(self, result:dict):
        """
        결과 메트릭으로부터 reward 계산
        """
        reward = result['sharpe']
        reward = torch.tensor([reward]) * 10
        return reward
        
    def update(self, w, r):
        """
        DDPG 스타일 업데이트
        """
        self.lam = 0.0
        alpha = 0.6219

        # R network update
        r_hat = self.rnet(w.detach())
        r_loss = self.mse(r_hat, r)
        
        self.opt_r.zero_grad()
        r_loss.backward()
        self.opt_r.step()

        # Policy update
        reg = self.mnet.cost(w)
        w_loss = -(self.rnet(w) - 0.0*reg).mean()

        self.opt_a.zero_grad()
        w_loss.backward(retain_graph=True)
        self.opt_a.step()

        # Lambda update
        lam_grad = -(reg - alpha).mean()
        self.lam -= 1e-2 * lam_grad

        # Noise scheduling
        self.mnet.sigma -= 5.0/45000 
        self.mnet.sigma = max(self.mnet.sigma, 0.005)
        return r_loss.item(), w_loss.item()
        
    def search(self, iter, start='1990', end='2024'):
        """
        RL 에이전트 학습 Loop
        """
        
        w_tensor = deque(maxlen=10000)
        r_tensor = deque(maxlen=10000)
        score = 0
        batch_size = 128

        for i in range(iter):
            weight = self.get_w()
            self.init(weight.detach().numpy())
            result = self.test(start, end)[-1]
            reward = self.get_r(result)

            score += 0.01 * (reward.item() - score)
            w_tensor.append(weight)
            r_tensor.append(reward)

            if len(w_tensor) >= batch_size:
                w_batch = random.sample(w_tensor, batch_size)
                r_batch = random.sample(r_tensor, batch_size)

                w_batch = torch.stack(w_batch).float()
                r_batch = torch.stack(r_batch).float()
                
                r_loss, w_loss = self.update(w_batch, r_batch)

                print(f'iter:{i}')
                print(f'reward:{reward.item()}')
                print(f'score:{score}')
                print(f'reward:{reward}')
                print(f'lambda:{self.lam}')
                print(f'sigma:{self.mnet.sigma}')
                print(f'r loss:{r_loss}')
                print(f'w loss:{w_loss}')
                print(f'{weight.detach()}')
                print(f'{self.get_w(False).detach()}\n')

                

class RANDOMSEARCH(BackTester):
    def __init__(self, config):
        BackTester.__init__(self, config)

        self.dim = config['Dim']
        self.optimal = None

    def get_w(self):
        """
        랜덤 가중치를 리턴
        """
        w = np.random.rand(self.dim)
        w[np.argsort(w)[:8]] = 0.0 
        w = w / np.sum(w)
        return w
    
    def search(self, iter, start='1990', end='2024'):
        """
        랜덤 써치를 통한 최적 가중치 탐색
        """

        best = 0

        for i in range(iter):
            weight = self.get_w()
            self.init(weight)
            result = self.test(start, end)[-1] 
            reward = result['sharpe'] 

            self.optimal = weight \
                if reward > best else self.optimal
            
            best = reward \
                if reward > best else best
            
            print(f'iter:{i}')
            print(f'best:{best}\n')
        
    def save(self, path):
        param = torch.tensor(self.optimal)
        torch.save(param, path)
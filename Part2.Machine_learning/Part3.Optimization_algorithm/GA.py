import numpy as np
from sko.GA import GA
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import os

class Net(torch.nn.Module):
    def __init__ (self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(2, 8, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.BatchNorm1d(8),
                                    nn.Sigmoid(),
                                    )
        self.layer2 = nn.Sequential(nn.Conv1d(8, 32, kernel_size=2, stride=1, padding=0, bias=True),
                                    nn.BatchNorm1d(32),
                                    nn.Sigmoid(),
                                    nn.Dropout(p=0.2),
                                    )
        self.layer3 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0, bias=True),
                                    nn.BatchNorm1d(64),
                                    nn.Sigmoid(),
                                    nn.Dropout(p=0.2),
                                    )
        self.layer4 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=0, bias=True),
                                    nn.BatchNorm1d(128),
                                    nn.Sigmoid(),
                                    )
        self.liners = nn.Linear(128,1)
    def forward(self, x):
        x = x.view(1, 2, 6)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.flatten(1)
        x = self.liners(x)
        x = x /(6.02214076)*10      # Convert to mol per nanosecond × e-24
        return x


def demo_func(x):
    x = torch.tensor(x,dtype=torch.float).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(x).detach().flatten().cpu().item()
    return -predictions

a = 16.54770780  # Lower limit for the sum of the densities of the hydrophobic component
b = 16.59009153  # Upper limit for the sum of the densities of the hydrophobic component
c = 5.405450963  # Lower limit for the sum of the densities of the hydrophilic component
d = 5.410484495  # Upper limit for the sum of the densities of the hydrophilic component
e = [   1.00793225,	0.650772,	1.0208,	0.480321667,	0.958379833,	0.09504495
]  # Lower limit of density of hydrophobic component
f = [1.735497,	1.3885525,	1.688923333,	1.166510833,	1.452406667,	0.47178485
]  # Upper limit of density of hydrophobic component
g = [0.004955282,	0.344279,	0.164172333,	0.434363,	0.13264284,	0.000195022]  # Lower limit of density of hydrophilic component
h = [0.2699143,	0.689573,	0.5276395,	0.787735667,	0.375954667,	0.002393714]  # Upper limit of density of hydrophilic component
k = [1.3028447,	1.337171667,	1.508889167,	1.253011333,	1.293346717,	0.095285937
]  # Lower limit for the sum of the densities at the corresponding positions of the hydrophilic and hydrophobic components
j = [1.7801253,	1.7289775,	1.869621167,	1.632162833,	1.665883383,	0.472861305
]  # Upper limit for the sum of the densities at the corresponding positions of the hydrophilic and hydrophobic components

'''Defining constraints'''
constraint_ueq = []
for i in range(6):
    constraint_ueq.append(lambda x, i=i: x[i] + x[i + 6] - j[i])  
    constraint_ueq.append(lambda x, i=i: -(x[i] + x[i + 6]) + k[i]) 
    constraint_ueq.append(lambda x, i=i: x[i] - f[i]) 
    constraint_ueq.append(lambda x, i=i: -x[i] + e[i])
    constraint_ueq.append(lambda x, i=i: x[i + 6] - h[i])
    constraint_ueq.append(lambda x, i=i: -x[i + 6] + g[i])

constraint_ueq.append(lambda x: (x[0] + x[5]) * 2 + sum(x[1:6]) * 3 - b) 
constraint_ueq.append(lambda x: -((x[0] + x[5]) * 2 + sum(x[1:6]) * 3) + a)
constraint_ueq.append(lambda x: (x[0 + 6] + x[5 + 6]) * 2 + sum(x[1 + 6:6 + 6]) * 3 - d)
constraint_ueq.append(lambda x: -((x[0 + 6] + x[5 + 6]) * 2 + sum(x[1 + 6:6 + 6]) * 3) + c)

constraint_ueq = tuple(constraint_ueq)

'''Select device and load model'''
if torch.cuda.is_available():
    device = "cuda:0"
    model = torch.load('../Part1.Model_trainning/CNN_training_data/max_net')
else:
    device = "cpu"
    model = torch.load('../Part1.Model_trainning/CNN_training_data/max_net', map_location='cpu')

device = torch.device(device)



ga = GA(func=demo_func, n_dim=12, size_pop=1000, max_iter=10000,prob_mut=0.1, lb=e+g, ub=f+h,constraint_ueq=constraint_ueq, precision=1e-7)
ga.record_mode = True
ga.run()


if not os.path.exists('GA_data'):
    os.mkdir('GA_data')


Y_history = pd.DataFrame(ga.all_history_Y)

plt.plot((-(Y_history.min(axis=1))).cummax())
plt.ylim(0, 30)
plt.savefig('GA_data/ga.png')
plt.show()

print("best_x:",ga.best_x,"best_y:",ga.best_y)


score = pd.DataFrame({'score':(-(Y_history.min(axis=1))).cummax()})
score.to_excel('GA_data/score.xlsx')
list = pd.DataFrame({'list':ga.best_x})
list.to_excel('GA_data/list.xlsx')

x=ga.best_x
for idx in range(6):
    if not (k[idx] <= x[idx] + x[idx + 6] <= j[idx]) or not (e[idx] <= x[idx] <= f[idx]) or not (g[idx] <= x[idx + 6] <= h[idx]):
        print('Unsatisfactory')
        break
else:
    if not a <= (x[0] + x[5]) * 2 + sum(x[1:6]) * 3 <= b or not c <= (x[0 + 6] + x[5 + 6]) * 2 + sum(x[1 + 6:6 + 6]) * 3 <= d:
        print('Unsatisfactory')
    else:
        print('Satisfactory')



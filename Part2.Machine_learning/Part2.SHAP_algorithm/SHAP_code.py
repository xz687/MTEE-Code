import pandas as pd
import numpy as np
import torch
from torch import nn
import shap
import matplotlib.pyplot as plt
import os


'''Select device'''
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
# device = "cpu"
device = torch.device(device)

'''seed'''
m_seed = 20
torch.manual_seed(m_seed)
torch.cuda.manual_seed_all(m_seed)

'''Read the dataset'''

df = pd.read_excel(r"../Total_data_set.xlsx", header=0, index_col=0)
X = df.iloc[:, :12]
Y = df.iloc[:, 16]


'''data format transformation'''
X_nparray = np.array(X)
X_torch = torch.tensor(X_nparray, dtype=torch.float32, device=device)  # Convert the dataset into a format recognized by torch


'''Defining CNN model parameters'''
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
        x = x.view(x.shape[0], 2, 6)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.flatten(1)
        x = self.liners(x)
        x = x /(6.02214076)*10      # Convert to mol per nanosecond × e-24
        return x

'''Load net'''
if torch.cuda.is_available():
    net = torch.load('../Part1.Model_trainning/CNN_training_data/max_net')
else:
    net = torch.load('../Part1.Model_trainning/CNN_training_data/max_net', map_location='cpu')


'''SHAP explanation'''
explainer = shap.DeepExplainer(net, X_torch)
base_values = explainer.expected_value
print(base_values[0])
shap_values = explainer.shap_values(X_torch)
explanation_multiple = shap.Explanation(shap_values, data=X_nparray, base_values=explainer.expected_value, feature_names=df.columns[:12])


'''Output to Excle'''
if not os.path.exists('SHAP_data'):
    os.mkdir('SHAP_data')
columns_shap = [col + '_shap' for col in df.columns[:12]]
df = pd.DataFrame(shap_values,columns=columns_shap)
df['base_value']=base_values[0]
df['prediction'] = df.apply(lambda x: x.sum(), axis=1)
df=pd.concat([df,X.reset_index(drop=True).rename(columns=lambda x: x.replace(f"{x}",f"{x}_feature")),Y.reset_index(drop=True)],axis=1)
df.to_excel('SHAP_data/shap_values.xlsx')


'''Plotting partial dependencies'''
for i in range(1,13):
    plt.figure(dpi=300)
    shap.plots.scatter(explanation_multiple[:,i-1], color='r', show=False)
    plt.tight_layout()
    plt.savefig(f'SHAP_data/scatter_{i}.png')
    plt.clf()


'''Plotting the significance of features (beeswarm)'''
plt.figure(dpi=300)
shap.plots.beeswarm(explanation_multiple,show=False,max_display=12)    #plot_size =(15,13)
plt.tight_layout()
plt.savefig('SHAP_data/beeswarm.png')
plt.clf()


'''Heatmap'''
list = [0,6,1,7,2,8,3,9,4,10,5,11]
for i in range(1,7):
    '''Sort by hbi from largest to smallest'''
    sort_result = shap_values[:,i-1].argsort()
    plt.figure(dpi=300)
    shap.plots.heatmap(explanation_multiple,show=False,max_display=12,plot_width=10.5,instance_order=sort_result,feature_order=[0,6,1,7,2,8,3,9,4,10,5,11])
    plt.tight_layout()
    plt.savefig(f'SHAP_data/heatmap_model_sort_hb{i}.png')
    plt.clf()


sort_result = shap_values[:,0].argsort()
plt.figure(dpi=300)
shap.plots.heatmap(explanation_multiple,show=False,max_display=12,plot_width=10.5,instance_order=sort_result)
plt.tight_layout()
plt.savefig(f'SHAP_data/heatmap.png')
plt.clf()




'Feature data for PSO optimization'
X=[1.1735718,	0.874463877,	1.304887803,	0.912174191,	1.202369835,	0.272234089,	0.129374233	,0.507989601,	0.350584464,	0.550691688,	0.306597649,	0.000786849
]



X_torch = torch.tensor(X, dtype=torch.float32, device=device)
X_torch=X_torch.view(1,12)
shap_values = explainer.shap_values(X_torch)
explanation = shap.Explanation(shap_values[0], data=X, base_values=explainer.expected_value[0], feature_names=df.columns[:12])

plt.figure(dpi=300)
shap.plots.waterfall(explanation, max_display=12, show=False)
plt.tight_layout()
plt.savefig('SHAP_data/waterfall_PSO.png')
plt.clf()





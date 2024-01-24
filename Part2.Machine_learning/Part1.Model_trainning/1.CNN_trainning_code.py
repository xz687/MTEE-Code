import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import random
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import os


'''Setting hyperparameters'''
epochs = 10000   #Training epochs
lr = 0.0001     #Learning rate

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

'''Read and divide the dataset'''
def read_data():
    df = pd.read_excel(r"../Total_data_set.xlsx", header=0, index_col=0)
    index = list(set(df.iloc[:, 15]))
    random.seed(m_seed)
    ls = random.sample(index, 20)

    groups = df.groupby(df.iloc[:, 15])
    test_index = []
    for group_id, group in groups:
        if group_id in ls:
            test_index.extend(group.index)

    all_index = list(df.index)
    train_index=list(set(all_index)-set(test_index))

    '''Import of input features (hydrophilic and hydrophobic densities)'''
    X_test=(df.iloc[test_index, :12])
    X_train=(df.iloc[train_index, :12])

    '''Import output target (units per nanosecond)'''
    Y_test = df.iloc[test_index, 14].reset_index(drop=True)
    Y_train = df.iloc[train_index, 14].reset_index(drop=True)

    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test = read_data()


print(Y_train.shape,Y_test.shape)
print(sorted(list(set(list(Y_test)))))


'''Training set data format transformation'''
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)  # Convert the dataset into a format recognized by torch
Y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)
torch_dataset_train = torch.utils.data.TensorDataset(X_train, Y_train) 
train_data = torch.utils.data.DataLoader(torch_dataset_train, 287, shuffle=True)

'''Test set data format transformation'''
X_test_datafram = X_test
X_test = X_test
X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)  # Convert the dataset into a format recognized by torch
Y_test = torch.tensor(Y_test, dtype=torch.float32, device=device)
torch_dataset_test = torch.utils.data.TensorDataset(X_test, Y_test) 
data_test = torch.utils.data.DataLoader(torch_dataset_test, 60, shuffle=True)


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
        return x


r2_max = 0

net = Net().to(device)
criteon = torch.nn.MSELoss()  # Using the mean square error
optimizer = optim.Adam(net.parameters(), lr=lr)  # Updating parameters using Adam algorithm

average_loss_train_axis=[]
average_loss_test_axis=[]
r2_axis=[]
epoch_axis=[]

path = str('CNN_training_data')
if not os.path.exists(path):
    os.mkdir(path)

'''train'''
for epoch in range(1, epochs + 1):
    net.train()
    for batch_idx, (data, target) in enumerate(train_data):
        logits = net.forward(data)
        loss = criteon(logits.flatten(), target.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    prediction = []
    test_y = []

    net.eval()
    with torch.no_grad():
        for batch_idx, (test_x, test_ys) in enumerate(data_test):
            predictions = net(test_x).detach().flatten()
            prediction.extend(predictions.cpu())
            test_y.extend(test_ys.detach().flatten().cpu())

    '''Output intermediate value'''
    r2_test = r2_score(torch.tensor(test_y, dtype=torch.float32), torch.tensor(prediction, dtype=torch.float32))
    test_loss = criteon(torch.tensor(prediction, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))

    print('Train Epoch: {}\tTraining Loss: {:.6f}\tTest Loss: {:.6f}\tTest R2 = {:.6f}'.format(epoch, loss,
                                                                                        test_loss.detach().numpy(),
                                                                                        r2_test))



    '''Save training process data'''
    average_loss_train_axis.append(loss.detach().cpu())
    average_loss_test_axis.append(test_loss)
    r2_axis.append(r2_test)
    epoch_axis.append(epoch)

    if r2_test > r2_max:
        r2_max = r2_test
        if r2_max > 0.9:
            prediction_max_output = [num.item() for num in prediction]
            test_y_output = [num.item() for num in test_y]

            model_filename = 'max_net'
            prediction_filename = 'max_prediction.csv'
            distribution_filename = 'max_distribution.png'
            loss_r2_filename='loss_r2.png'
            model_path = os.path.join(path, model_filename)
            prediction_path = os.path.join(path, prediction_filename)
            distribution_path = os.path.join(path, distribution_filename)
            loss_r2=os.path.join(path, loss_r2_filename)

            '''Save the model'''
            torch.save(net, model_path)

            '''Save excel-max prediction group'''
            max_prediction = pd.DataFrame({'lable': test_y_output, 'prediction': prediction_max_output})
            max_prediction.to_csv(prediction_path, index=False)

            '''Output prediction plot'''
            plt.figure()
            plt.scatter(test_y_output, prediction_max_output, color='red')
            plt.plot([1, 14], [1, 14], color='black', linestyle='-')
            plt.xlim([1, 14])
            plt.ylim([1, 14])
            plt.xlabel('true')
            plt.ylabel('prediction')
            plt.title('R2={:.6f}'.format(r2_max))
            plt.savefig(distribution_path)
            plt.close()


'''Loss'''
plt.figure(figsize=(10, 4), dpi=300)
plt.subplot(1, 2, 1)
plt.plot(epoch_axis, average_loss_test_axis, 'r', lw=1)  # lw为曲线宽度
plt.plot(epoch_axis, average_loss_train_axis, 'b', lw=1)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Loss_test", "Loss_train"])
'''R^2'''
plt.subplot(1, 2, 2)
plt.plot(epoch_axis, r2_axis, 'b', lw=1)
plt.title("R^2_test")
plt.ylim([0, 1])
plt.xlabel("Epoch")
plt.ylabel("R^2")
loss_R2_path = os.path.join(path, 'loss-R2.png')
plt.savefig(loss_R2_path)
plt.close()







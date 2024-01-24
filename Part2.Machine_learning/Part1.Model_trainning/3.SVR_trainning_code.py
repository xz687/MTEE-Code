import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import random
from sklearn import metrics


m_seed = 20
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

    X_test=(df.iloc[test_index, :12])
    X_train=(df.iloc[train_index, :12])

    Y_test = df.iloc[test_index, 14].reset_index(drop=True)
    Y_train = df.iloc[train_index, 14].reset_index(drop=True)

    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test = read_data()



model = SVR(C=3.2527229965987012, epsilon=0.45097459503621723, gamma='scale', kernel='poly')

model.fit(X_train, Y_train)
test_pred = model.predict(X_test)
test_score = r2_score(Y_test, test_pred)
mse_test = metrics.mean_squared_error(Y_test, test_pred)

print("test_score:", test_score, "\nRmse_test:", mse_test**0.5/6.022*10)



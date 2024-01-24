import xgboost as xgb
import pandas as pd
from sklearn.metrics import r2_score
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


params = {
    "n_estimators": 900,
    "max_depth": 3,
    "learning_rate": 0.2999704447018923,
    "subsample": 0.8142938036508222,
    "colsample_bytree": 0.7404830262434537,
    "gamma": 2.182349543272545,
    "reg_alpha": 0.49744958808463,
    "reg_lambda": 8.55753183467539,
}

# 训练模型
model = xgb.XGBRegressor(**params)
model.fit(X_train, Y_train)
test_pred = model.predict(X_test)
test_score = r2_score(Y_test, test_pred)
mse_test = metrics.mean_squared_error(Y_test, test_pred)

print("test_score:", test_score, "\nRmse_test:", mse_test**0.5/6.022*10)
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

replace_set = {'y': 1, 'n': 2, '?': np.NAN, 'republican': 1, 'democrat': 2}
data = pd.read_csv('./data_train/house-votes-84.data')
data.columns = np.arange(1, data.shape[-1]+1)
data_copy = data.copy()
data_copy = data_copy.replace(replace_set)

sindex = np.argsort(data_copy.isna().sum().values.tolist())

for i in sindex:
    if data_copy.iloc[:, i].isna().sum() == 0:
        continue
    df = data_copy
    fillc = df.iloc[:, i]
    df = df.iloc[:, df.columns != df.columns[i]]
    df_0 = SimpleImputer(missing_values=np.nan,
                         strategy="constant",
                         fill_value=0).fit_transform(df)
    Ytrain = fillc[fillc.notnull()]
    Ytest = fillc[fillc.isnull()]
    Xtrain = df_0[Ytrain.index, :]
    Xtest = df_0[Ytest.index, :]

    rfc = RandomForestClassifier()
    rfc.fit(Xtrain, Ytrain)
    Ypredict = rfc.predict(Xtest)


    data_copy.loc[data_copy.iloc[:, i].isnull(), data_copy.columns[i]] = Ypredict

data_copy.to_csv('./result_data/vote_rfc.csv')

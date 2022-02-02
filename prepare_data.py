import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('train.csv')


gkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

df['fold'] = -1

for i, (_, idx) in enumerate(gkf.split(df, df.species)):
    df.loc[idx, 'fold'] = i

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train_kfold.csv')
le = LabelEncoder()
y = le.fit_transform(df.individual_id)
df['label'] = y

print(df.fold.value_counts())
df.to_csv('train_kfold.csv', index=False)


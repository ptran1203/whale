import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('data/train.csv')


gkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

df['fold'] = -1

for i, (_, idx) in enumerate(gkf.split(df, df.individual_id)):
    df.loc[idx, 'fold'] = i


df['sample_count'] = df.groupby(['individual_id'])['individual_id'].transform('count')
df['subset'] = 'train'

# df.loc[df['fold'] == 0, 'subset'] = 'test'
# df.loc[df['sample_count'] < 6, 'subset'] = 'train'



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df.individual_id)
df['label'] = y


train_df = df.query('fold != 0')
val_df = df.query('fold == 0')
print(f'nlabel={df.label.nunique()}train={train_df.label.nunique()}, test={val_df.label.nunique()}')
print(f"Test in train={len(set(val_df.label.unique()) - set(train_df.label.unique()))}")
print(len(train_df), len(val_df))
df.to_csv('data/train_kfold.csv', index=False)


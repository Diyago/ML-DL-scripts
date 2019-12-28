import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import joblib

total_folds = 10

df = pd.read_csv("./input/severstal-steel-defect-detection/train.csv")
df["ImageId"], df["ClassId"] = zip(*df["ImageId_ClassId"].str.split("_"))
df["ClassId"] = df["ClassId"].astype(int)
df = df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
df["defects"] = df.count(axis=1)

# Dumbing FOLD ids
FOLDS_PATH = "./input/folds.pkl"
folds = KFold(n_splits=10, shuffle=True, random_state=69)
folds_idx = [(train_idx, val_idx) for train_idx, val_idx in folds.split(df)]
joblib.dump(folds_idx, FOLDS_PATH)

# How to work work with FOLDS for cur_fold = 2
cur_fold = 2
folds_idx = joblib.load(FOLDS_PATH)
train_idx, val_idx = list(folds_idx)[cur_fold]
train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

# Standard Library Imports
import math
import json
import logging
import argparse as ap
import os
from joblib import dump
import time

# General Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# SKLearn Imports
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, RocCurveDisplay, roc_curve


def parse_cli():
    par = ap.ArgumentParser()
    par.add_argument("--data",
                     help='File path to training data input',
                     default='./data/train.csv')
    par.add_argument("--max-iter",
                     help='maximum iteration for logistic regression model',
                     default=10000)
    par.add_argument("--model",
                     help='File path to trained model',
                     default='./log_model.joblib')
    args = par.parse_args()
    return(args)

# Get training args
args = parse_cli()
data_file = args.data
max_iter = args.max_iter
model_file = args.model

# Load Data
df = pd.read_csv(data_file)
df_X = df.drop("y", axis=1)
df_label = df["y"]

numeric_features = ["x1", "x2", "x4", "x5"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["x3", "x6", "x7"]
categorical_transformer = OneHotEncoder(handle_unknown="infrequent_if_exist")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("classifier", LogisticRegression(max_iter=max_iter))]
)

# Make LogReg Pipeline

RANDOM_STATE=1337

X_train, X_test, y_train, y_test = train_test_split(
    df_X,
    df_label,
    random_state=RANDOM_STATE
    )

clf.fit(X_train, y_train)

# Save model
dump(clf, model_file)

tprobs = clf.predict_proba(X_test)[:, 1]

timestr = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(f"metrics/{timestr}")

# Save numeric metrics to a txt file
with open(f"metrics/{timestr}/metrics.txt", "w") as f:
    f.write("model score: %.3f" % clf.score(X_test, y_test))
    f.write("\n")
    f.write(classification_report(y_test, clf.predict(X_test)))
    f.write("\n")
    f.write("Confusion matrix:\n")
    f.write(np.array2string(confusion_matrix(y_test, clf.predict(X_test)), separator=', '))
    f.write("\n")
    f.write(f'AUC: {roc_auc_score(y_test, tprobs)}')

# Save ROC curve plot to a pdf file
fpr, tpr, _ = roc_curve(y_test,  tprobs)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(f'metrics/{timestr}/roc.pdf')


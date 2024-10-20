import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib

X_smo=pd.read_csv('data/processed/X_smo.csv')
y_smo=pd.read_csv('data/processed/y_smo.csv')

X_smo = X_smo.drop('Unnamed: 0',axis=1)
y_smo = y_smo.drop('Unnamed: 0',axis=1)

clf_xgb = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
clf_xgb.fit(X_smo, y_smo)

joblib.dump(clf_xgb,"models/clf_xgb")
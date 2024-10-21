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

print("importing processed datasets\n")
X_smo=pd.read_csv('data/processed/X_smo.csv')
y_smo=pd.read_csv('data/processed/y_smo.csv')
X_smo = X_smo.drop('Unnamed: 0',axis=1)
y_smo = y_smo.drop('Unnamed: 0',axis=1)

X_rus=pd.read_csv('data/processed/X_rus.csv')
y_rus=pd.read_csv('data/processed/y_rus.csv')
X_rus = X_rus.drop('Unnamed: 0',axis=1)
y_rus = y_rus.drop('Unnamed: 0',axis=1)

X_cc=pd.read_csv('data/processed/X_cc.csv')
y_cc=pd.read_csv('data/processed/y_cc.csv')
X_cc = X_cc.drop('Unnamed: 0',axis=1)
y_cc = y_cc.drop('Unnamed: 0',axis=1)

X_ros=pd.read_csv('data/processed/X_ros.csv')
y_ros=pd.read_csv('data/processed/y_ros.csv')
X_ros = X_ros.drop('Unnamed: 0',axis=1)
y_ros = y_ros.drop('Unnamed: 0',axis=1)

print("Fitting RandomForestClassifier model\n")
clf_rf = RandomForestClassifier(n_jobs = -1, random_state = 42)
clf_rf.fit(X_ros, y_ros)

print("Fitting RandomForestDecisionTreeClassifierClassifier model\n")
clf_dt = DecisionTreeClassifier(criterion ='entropy', max_depth=4, random_state = 42)
clf_dt.fit(X_rus, y_rus)

print("Fitting KNeighborsClassifier model\n")
clf_knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski')
clf_knn.fit(X_cc, y_cc)

print("Fitting SVC model\n")
clf_svm = svm.SVC(gamma=0.1, kernel='poly', random_state=42, probability=True) 
clf_svm.fit(X_cc, y_cc)

print("Fitting LogisticRegression model\n")
clf_lr = LogisticRegression(C=1.0,random_state=42)
clf_lr.fit(X_ros, y_ros)

print("Fitting XGBClassifier model\n")
clf_xgb = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
clf_xgb.fit(X_smo, y_smo)

print("Saving models\n")
joblib.dump(clf_rf,"models/clf_rf")
joblib.dump(clf_dt,"models/clf_dt")
joblib.dump(clf_knn,"models/clf_knn")
joblib.dump(clf_svm,"models/clf_svm")
joblib.dump(clf_lr,"models/clf_lr")
joblib.dump(clf_xgb,"models/clf_xgb")
import pandas as pd
import joblib
import random

print("importing processed dataset\n")

dfDate=pd.read_csv('data/processed/weatherAUS_processed_data_with_date.csv')
dfDate = dfDate.drop('Unnamed: 0',axis=1)
dfRaintomorrow=dfDate['RainTomorrow']
dfDate=dfDate.drop('RainTomorrow',axis=1)
dfLocation=dfDate.drop('LocationReel',axis=1)
dfDate=dfDate.drop('LocationReel',axis=1)
df = dfDate.drop('Date',axis=1)

#clf_rf=joblib.load("models/clf_rf")
clf_dt=joblib.load("models/clf_dt")

#random.seed(1)

for i in range(10):
    aleatoire=random.randint(0,df.shape[0])

    pred=clf_dt.predict([df.loc[aleatoire]])

    print("ligne :",aleatoire)
    print("Pluie le",dfDate.loc[aleatoire]["Date"],":",round(df['RainToday'][0]))
    print("Prédiction pour le lendemain :",round(pred[0]))
    print("Réalité le lendemain :",round(dfRaintomorrow.loc[aleatoire]),"\n")

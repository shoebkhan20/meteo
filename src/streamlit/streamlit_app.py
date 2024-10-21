import streamlit as st
import pandas as pd
import joblib
import random

dfDate=pd.read_csv('data/processed/weatherAUS_processed_data_with_date.csv')
dfDate = dfDate.drop('Unnamed: 0',axis=1)
dfRaintomorrow=dfDate['RainTomorrow']
dfDate=dfDate.drop('RainTomorrow',axis=1)
df = dfDate.drop('Date',axis=1)

st.title("Prédictions météo - by SKH")

st.write("Exercices de prédiction")

#modelRadio=st.radio('Sélectionner votre model',['RandomForestClassifier','RandomForestDecisionTreeClassifier','KNeighborsClassifier','SVC','XGBClassifier'])

#if st.button("Lancer un test de prédiction de météo"):
#    if modelRadio=="RandomForestDecisionTreeClassifier":
#        clf_dt=joblib.load("models/clf_dt")
#        aleatoire=random.randint(0,df.shape[0])
#        pred=clf_dt.predict([df.loc[aleatoire]])
#        st.write("Pluie le",dfDate.loc[aleatoire]["Date"],":",round(df['RainToday'][0]))
#        st.write("Prédiction pour le lendemain :",round(pred[0]))
#        st.write("Réalité le lendemain :",round(dfRaintomorrow.loc[aleatoire]),"\n")



    


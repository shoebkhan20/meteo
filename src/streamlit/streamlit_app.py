import streamlit as st
import pandas as pd
import joblib
import random

dfDate=pd.read_csv('data/processed/weatherAUS_processed_data_with_date.csv')
dfDate = dfDate.drop('Unnamed: 0',axis=1)
dfRaintomorrow=dfDate['RainTomorrow']
dfDate=dfDate.drop('RainTomorrow',axis=1)
dfLocation=dfDate['LocationReel']
dfDate=dfDate.drop('LocationReel',axis=1)
df = dfDate.drop('Date',axis=1)

st.title("Prédictions météo - by SKH")

st.write("Exercices de prédiction")

modelRadio=st.radio('Sélectionner votre model',['RandomForestDecisionTreeClassifier','KNeighborsClassifier','SVC','XGBClassifier'])
listeModels=['RandomForestDecisionTreeClassifier','KNeighborsClassifier','SVC','XGBClassifier']

def displayWeatherImage(meteo):
    if (meteo == 0):
        st.image("data/images/sun.png",width=30)
    else:
        st.image("data/images/rain.png",width=30)

def predict(modelName):
    modelPath="models/"+modelName
    model=joblib.load(modelPath)
    aleatoire=random.randint(0,df.shape[0])
    pred=model.predict([df.loc[aleatoire]])
    col1, col2 = st.columns(2)
    with col1:
         st.write("Météo le",dfDate.loc[aleatoire]["Date"]," à",dfLocation.loc[aleatoire],":")
         st.write("Prédiction pour le lendemain :")
         st.write("Réalité le lendemain :")
    with col2:
         displayWeatherImage(round(df['RainToday'].loc[df.index == aleatoire].values[0]))
         displayWeatherImage(round(pred[0]))
         displayWeatherImage(round(dfRaintomorrow.loc[aleatoire]))

    

if st.button("Lancer un test de prédiction de météo aléatoire"):
        if modelRadio=="RandomForestDecisionTreeClassifier":
              predict("clf_dt")
        if modelRadio=="KNeighborsClassifier":
              predict("clf_knn")
        if modelRadio=="SVC":
              predict("clf_svm")
        if modelRadio=="XGBClassifier":
              predict("clf_xgb")
        



    


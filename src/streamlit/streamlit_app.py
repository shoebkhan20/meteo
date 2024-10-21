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

pages=['Exercices de prédiction aléatoire','Météo de demain ?']
page=st.sidebar.radio("Aller vers",pages)

if page == pages[0]:

    st.write("Exercices de prédiction aléatoire")

    listeModels=['clf_dt','clf_knn','clf_svm','clf_xgb']
    listeModelTitles=['RandomForestDecisionTreeClassifier','KNeighborsClassifier','SVC','XGBClassifier']

    def displayWeatherImage(meteo):
        if (meteo == 0):
            st.image("data/images/sun.png",width=30)
        else:
            st.image("data/images/rain.png",width=30)

    def predict(listeModels,aleatoire):
        list_to_return=[]
        for modelName in listeModels:
            modelPath="models/"+modelName
            model=joblib.load(modelPath)
            pred=model.predict([df.loc[aleatoire]])
            list_to_return.append(pred)
        return list_to_return



    if st.button("Lancer un test de prédiction de météo aléatoire"):
        aleatoire=random.randint(0,df.shape[0])
        col1, col2 = st.columns(2)
        listPredictions=predict(listeModels,aleatoire)
        with col1:
            st.write("##### Météo le",dfDate.loc[aleatoire]["Date"]," à",dfLocation.loc[aleatoire]," #####")
            for title in listeModelTitles:
                st.write("Prédiction pour ",title)
            st.write("##### Réalité le lendemain #####")
        with col2:
            displayWeatherImage(round(df['RainToday'].loc[df.index == aleatoire].values[0]))
            for p in listPredictions:
                displayWeatherImage(round(p[0]))
            displayWeatherImage(round(dfRaintomorrow.loc[aleatoire]))

if page == pages[1]:

    st.write("Prédiction de la météo demain en Australie")

    option = st.selectbox(
    "Choisissez une localité en Australie",
    ('Albury','BadgerysCreek','Cobar','CoffsHarbour','Moree','Newcastle','NorahHead','NorfolkIsland','Penrith','Richmond','Sydney','SydneyAirport','WaggaWagga','Williamtown','Wollongong','Canberra','Tuggeranong','MountGinini','Ballarat','Bendigo','Sale','MelbourneAirport','Melbourne','Mildura','Nhil','Portland','Watsonia','Dartmoor','Brisbane','Cairns','GoldCoast','Townsville','Adelaide','MountGambier','Nuriootpa','Woomera','Albany','Witchcliffe','PearceRAAF','PerthAirport','Perth','SalmonGums','Walpole','Hobart','Launceston','AliceSprings','Darwin','Katherine','Uluru'),)




    


import streamlit as st
import pandas as pd
import joblib
import random
import requests

api_key='b7ef3ccb86e5c46d8f284ee5944c5cc5'

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
    ('Albury','Badgerys Creek','Cobar','Coffs Harbour','Moree','Newcastle','Nora hHead','Norfolk Island','Penrith','Richmond','Sydney','Sydney Airport','Wagga Wagga','Williamtown','Wollongong','Canberra','Tuggeranong','Mount Ginini','Ballarat','Bendigo','Sale','Melbourne Airport','Melbourne','Mildura','Nhill','Portland','Watsonia','Dartmoor','Brisbane','Cairns','Gold Coast','Townsville','Adelaide','Mount Gambier','Nuriootpa','Woomera','Albany','Witchcliffe','Perth Airport','Perth','Salmon Gums','Walpole','Hobart','Launceston','Alice Springs','Darwin','Katherine','Mutitjulu'),)

    requeteAuth="http://api.openweathermap.org/data/2.5/forecast?id=524901&appid={}"
    response=requests.get(requeteAuth.format(api_key))
    if response.status_code!=200:
        st.write(response.status_code)
    else:
        requeteGeoCoding="http://api.openweathermap.org/geo/1.0/direct?q={},AU&limit=5&appid={}"
        response=requests.get(requeteGeoCoding.format(option,api_key))
        if response.status_code!=200:
            st.write(response.status_code)
        else:
            lat=response.json()[0]['lat']
            lon=response.json()[0]['lon']
            st.write("Lattitude : ",lat)
            st.write("Longitude : ",lon)

            requeteGetWeather="https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&units=metric&appid={}"
            response=requests.get(requeteGetWeather.format(lat,lon,api_key))
            print(requeteGetWeather.format(lat,lon,api_key))
            if response.status_code!=200:
                st.write(response.status_code)
            else:
                st.write(response.json())



    


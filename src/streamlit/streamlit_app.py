import streamlit as st
import pandas as pd
import joblib
import random
import requests
from io import StringIO
import datetime
import itertools

api_key='b7ef3ccb86e5c46d8f284ee5944c5cc5'

dfDate=pd.read_csv('data/processed/weatherAUS_processed_data_with_date.csv')
dfDate = dfDate.drop('Unnamed: 0',axis=1)
dfRaintomorrow=dfDate['RainTomorrow']
dfDate=dfDate.drop('RainTomorrow',axis=1)
dfLocation=dfDate['LocationReel']
dfDate=dfDate.drop('LocationReel',axis=1)
df = dfDate.drop('Date',axis=1)

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

listeModels=['clf_dt','clf_knn','clf_svm','clf_xgb','clf_lr']
listeModelTitles=['RandomForestDecisionTreeClassifier','KNeighborsClassifier','SVC','XGBClassifier','LogisticRegression']

st.title("Prédictions météo - by SKH")

pages=['Exercices de prédiction aléatoire','Météo de demain ?']
page=st.sidebar.radio("Aller vers",pages)

if page == pages[0]:

    st.write("Exercices de prédiction aléatoire")

    if st.button("Lancer un test de prédiction de météo aléatoire"):
        aleatoire=random.randint(0,df.shape[0])
        col1, col2 = st.columns(2)
        listPredictions=predict(listeModels,aleatoire)
        #st.write(dfDate.loc[aleatoire])
        #st.write(dfDate.loc[aleatoire+1])
        with col1:
            st.write("##### Météo le",dfDate.loc[aleatoire]["Date"]," à",dfLocation.loc[aleatoire]," #####")
            st.write("**Prédictions**")
            for (title,p) in zip(listeModelTitles,listPredictions):
                iconResult="*"
                if (round(p[0]) == round(dfRaintomorrow.loc[aleatoire])):
                    iconResult=":white_check_mark:"
                st.write(iconResult,title)
            st.write(" ")
            st.write("##### Réalité le lendemain #####")
        with col2:
            displayWeatherImage(round(df['RainToday'].loc[df.index == aleatoire].values[0]))
            st.write(" ")
            for p in listPredictions:
                displayWeatherImage(round(p[0]))
            st.write(" ")
            displayWeatherImage(round(dfRaintomorrow.loc[aleatoire]))

if page == pages[1]:

    st.write("Prédiction de la météo demain en Australie")

    option = st.selectbox(
    "Choisissez une localité en Australie",
    ('Albury','Badgerys Creek','Cobar','Coffs Harbour','Moree','Newcastle','Norah Head','Norfolk Island','Penrith','Richmond','Sydney','Sydney Airport','Wagga Wagga','Williamtown','Wollongong','Canberra','Tuggeranong','Mount Ginini','Ballarat','Bendigo','Sale','Melbourne Airport','Melbourne','Mildura','Nhill','Portland','Watsonia','Dartmoor','Brisbane','Cairns','Gold Coast','Townsville','Adelaide','Mount Gambier','Nuriootpa','Woomera','Albany','Witchcliffe','Perth Airport','Perth','Salmon Gums','Walpole','Hobart','Launceston','Alice Springs','Darwin','Katherine','Uluru'),)

    locationDict={'Adelaide':'5081','Albany':'6001','Albury':'2002','Alice Springs':'8002','Badgerys Creek':'2005','Ballarat':'3005','Bendigo':'3008','Brisbane':'4019','Cairns':'4024','Canberra':'2801','Cobar':'2029','Coffs Harbour':'2030','Dartmoor':'3101','Darwin':'8014','Gold Coast':'4050','Hobart':'7021','Katherine':'8024','Launceston':'7025','Melbourne Airport':'3049','Melbourne':'3033','Mildura':'3051','Moree':'2084','Mount Gambier':'5041','Mount Ginini':'2804','Yulara':'8056','Newcastle':'2098','Nhill':'3059','Norah Head':'2099','Norfolk Island':'2100','Nuriootpa':'5049','Penrith':'2111','Perth Airport':'6110','Perth':'6111','Portland':'3068','Richmond':'4101','Sale':'3022','Salmon Gums':'6119','Sydney Airport':'2125','Sydney':'2124','Townsville':'4128','Tuggeranong':'2802','Wagga Wagga':'2139','Walpole':'6138','Watsonia':'3079','Williamtown':'2145','Witchcliffe':'6071','Wollongong':'2146','Woomera':'5072'}

    current_time=datetime.datetime.now()
    yearMonth=str(current_time.year)+str(current_time.month)

    #Exemple d'URL : http://www.bom.gov.au/climate/dwo/202410/text/IDCJDW2084.202410.csv
    url='http://www.bom.gov.au/climate/dwo/{}/text/IDCJDW{}.{}.csv'.format(yearMonth,locationDict[option],yearMonth)
    #Headers pour tricher, comme si on était un visiteur humain, sinon le site refuse l'accès depuis les scripts
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}    
    
    response = requests.get(url, headers=headers)

    if response.status_code!=200:
        st.write(response.status_code)
    else:
        
        st.write("Données récupérées du site : ",url)

        currentWeather=response.text

        tableInString=currentWeather[currentWeather.index("Date")-1:]

        table=pd.read_csv(StringIO(tableInString))

        st.write(table)
        

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Aujourd'hui à",option,", voici la météo :**")
        with col2:
            displayWeatherImage(round(table['Rainfall (mm)'].iloc[-1]))

        
        table=table.drop(columns=['Date','Minimum temperature (°C)','Time of maximum wind gust'])

        lastRow=table.iloc[-1]
        
        st.write(lastRow)

        st.write("Demain, en fonction de nos modèles de prédiction, il fera :")


#    requeteAuth="http://api.openweathermap.org/data/2.5/forecast?id=524901&appid={}"
#    response=requests.get(requeteAuth.format(api_key))
#    if response.status_code!=200:
#        st.write(response.status_code)
#    else:
#        requeteGeoCoding="http://api.openweathermap.org/geo/1.0/direct?q={},AU&limit=5&appid={}"
#        response=requests.get(requeteGeoCoding.format(option,api_key))
#        if response.status_code!=200:
#            st.write(response.status_code)
#        else:
#            lat=response.json()[0]['lat']
#            lon=response.json()[0]['lon']
#            st.write("Lattitude : ",lat)
#            st.write("Longitude : ",lon)

#            requeteGetWeather="https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&units=metric&appid={}"
#            requeteGetWeather="https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&hourly=temperature_2m&models=bom_access_global"
#            response=requests.get(requeteGetWeather.format(lat,lon,api_key))
#            print(requeteGetWeather.format(lat,lon,api_key))
#            if response.status_code!=200:
#                st.write(response.status_code)
#            else:
#                st.write(response.json())



    


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('data/raw/weatherAUS.csv')

# Transformation des variables qualitatives  'WindGustDir', 'WindDir9am', 'WindDir3pm' en variables qualitatives
# 16 points cardinaux : 'W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW','ENE','SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'
#E   : 0
#ENE : 2Pi/16 = Pi/8
#NE  : 4Pi/16 = Pi/4
#NNE : 6Pi/16 = 3Pi/8
#N   : 8Pi/16 = Pi/2
#NNW : 10Pi/16 = 5Pi/8
#NW  : 12Pi/16 = 3Pi/4
#WNW : 14Pi/16 = 7Pi/8
#W   : 16Pi/16 = Pi
#WSW : 18Pi/16 = 9Pi/8
#SW  : 20Pi/16 = 5Pi/4
#SSW : 22Pi/16 = 11Pi/8
#S   : 24Pi/16 = 3Pi/2
#SSE : 26Pi/16= 13Pi/8
#SE  : 28Pi/16 = 7Pi/4
#ESE : 30Pi/16 = 15Pi/8
df['WindGustDir'] = df['WindGustDir'].replace({'E':0, 'ENE':(0.125*np.pi),'NE':(0.250*np.pi),'NNE':(0.375*np.pi),'N':(0.500*np.pi),'NNW':(0.625*np.pi),'NW':(0.750*np.pi),'WNW':(0.875*np.pi),
                                               'W':np.pi,'WSW':(1.125*np.pi),'SW':(1.250*np.pi),'SSW':(1.375*np.pi),'S':(1.500*np.pi),'SSE':(1.625*np.pi),'SE':(1.750*np.pi),'ESE':(0.875*np.pi)})

df['WindDir9am'] = df['WindDir9am'].replace({'E':0, 'ENE':(0.125*np.pi),'NE':(0.250*np.pi),'NNE':(0.375*np.pi),'N':(0.500*np.pi),'NNW':(0.625*np.pi),'NW':(0.750*np.pi),'WNW':(0.875*np.pi),
                                               'W':np.pi,'WSW':(1.125*np.pi),'SW':(1.250*np.pi),'SSW':(1.375*np.pi),'S':(1.500*np.pi),'SSE':(1.625*np.pi),'SE':(1.750*np.pi),'ESE':(0.875*np.pi)})

df['WindDir3pm'] = df['WindDir3pm'].replace({'E':0, 'ENE':(0.125*np.pi),'NE':(0.250*np.pi),'NNE':(0.375*np.pi),'N':(0.500*np.pi),'NNW':(0.625*np.pi),'NW':(0.750*np.pi),'WNW':(0.875*np.pi),
                                               'W':np.pi,'WSW':(1.125*np.pi),'SW':(1.250*np.pi),'SSW':(1.375*np.pi),'S':(1.500*np.pi),'SSE':(1.625*np.pi),'SE':(1.750*np.pi),'ESE':(0.875*np.pi)})

# Afin d'éviter la création de 49 colonnes avec l'encodage One-Hot (pd.get_dummies), on utilisera l'encodage d'étiquettes (label Encoding)
label_encoder = LabelEncoder()
df['Location']= label_encoder.fit_transform(df['Location'])

# Transformation des variables qualitatives  'RainToday' et 'RainTomorrow' en variables qualitatives
df['RainToday'] = df['RainToday'].replace({'No':0, 'Yes':1})
df['RainTomorrow'] = df['RainTomorrow'].replace({'No':0, 'Yes':1})

# Suppression de la variable 'Date' qui n'apportera rien dans la prévision de la variable cible 'RainTomorrow'
dfDate=df['Date'].copy()
df= df.drop('Date', axis =1)

# Suppression des variables 'MinTemp','Temp9am' et 'Temp3pm'
df=df.drop('MinTemp',axis=1)
df=df.drop('Temp9am',axis=1)
df=df.drop('Temp3pm',axis=1)

# Suppression de la variable 'Pressure9am'
df=df.drop('Pressure9am',axis=1)

# Supression des 3 267 lignes où les valeurs de 'RainTomorrow' sont manquantes
df=df.dropna(subset='RainTomorrow', axis = 0)

# 'Cloud9am' et 'Cloud3pm' ne peuvent pas avoir des valeurs de 9 ( de 0 à 8 uniquement)
# Supression des 3 lignes concernées par la valeur 9 pour les variables 'Cloud'
df=df[df['Cloud9am']!=9]
df=df[df['Cloud3pm']!=9]

# Remplacement des valeurs extrêmes de 'MaxTemp' par des valeurs manquantes
q1 = df['MaxTemp'].quantile(q=0.25)
q3 = df['MaxTemp'].quantile(q=0.75)
iqr = q3-q1
seuil_haut = q3+1.5*iqr
seuil_bas = q1-1.5*iqr
df['MaxTemp'] = df['MaxTemp'].apply(lambda x: np.nan if x > seuil_haut or x < seuil_bas else x)


# Suppression des valeurs extrêmes de la variable 'Rainfall'

#Avec une distribution des données très asymétrique et une forte concentration des valeurs autour de zéro, la méthode classique utilisant
#les quantiles et l'écart interquartile (IQR) n'est appropriée pour déterminer les valeurs extrêmes : il est plus pertinent de définir un seuil empirique
#basé sur une proportion de données : on définit que les valeurs supérieures 95e centile sont des valeurs extrêmes.
seuil_haut = df['Rainfall'].quantile(q=0.95)
seuil_bas = df['Rainfall'].min()
# Remplacement des valeurs extrêmes de 'Rainfall' par des valeurs manquantes
df['Rainfall'] = df['Rainfall'].apply(lambda x: np.nan if x > seuil_haut or x < seuil_bas else x)

# Suppression des valeurs extrêmes de la variable 'Evaporation'
# Remplacement des valeurs extrêmes de 'Evaporation' par des valeurs manquantes
q1 = df['Evaporation'].quantile(q=0.25)
q3 = df['Evaporation'].quantile(q=0.75)
iqr = q3-q1
seuil_haut = q3+1.5*iqr
seuil_bas = q1-1.5*iqr
df['Evaporation'] = df['Evaporation'].apply(lambda x: np.nan if x > seuil_haut or x < seuil_bas else x)

# Remplacement des valeurs extrêmes de 'Sunshine' par des valeurs manquantes
q1 = df['Sunshine'].quantile(q=0.25)
q3 = df['Sunshine'].quantile(q=0.75)
iqr = q3-q1
seuil_haut = q3+1.5*iqr
seuil_bas = q1-1.5*iqr
df['Sunshine'] = df['Sunshine'].apply(lambda x: np.nan if x > seuil_haut or x < seuil_bas else x)

# Remplacement des valeurs extrêmes de 'WindGustSpeed' par des valeurs manquantes
q1 = df['WindGustSpeed'].quantile(q=0.25)
q3 = df['WindGustSpeed'].quantile(q=0.75)
iqr = q3-q1
seuil_haut = q3+1.5*iqr
seuil_bas = q1-1.5*iqr
df['WindGustSpeed'] = df['WindGustSpeed'].apply(lambda x: np.nan if x > seuil_haut or x < seuil_bas else x)

# Remplacement des valeurs extrêmes de 'WindSpeed9am' par des valeurs manquantes
q1 = df['WindSpeed9am'].quantile(q=0.25)
q3 = df['WindSpeed9am'].quantile(q=0.75)
iqr = q3-q1
seuil_haut = q3+1.5*iqr
seuil_bas = q1-1.5*iqr
df['WindSpeed9am'] = df['WindSpeed9am'].apply(lambda x: np.nan if x > seuil_haut or x < seuil_bas else x)

# Remplacement des valeurs extrêmes de 'WindSpeed3pm' par des valeurs manquantes
q1 = df['WindSpeed3pm'].quantile(q=0.25)
q3 = df['WindSpeed3pm'].quantile(q=0.75)
iqr = q3-q1
seuil_haut = q3+1.5*iqr
seuil_bas = q1-1.5*iqr
df['WindSpeed3pm'] = df['WindSpeed3pm'].apply(lambda x: np.nan if x > seuil_haut or x < seuil_bas else x)

# Remplacement des valeurs extrêmes de 'Humidity9am' par des valeurs manquantes
q1 = df['Humidity9am'].quantile(q=0.25)
q3 = df['Humidity9am'].quantile(q=0.75)
iqr = q3-q1
seuil_haut = q3+1.5*iqr
seuil_bas = q1-1.5*iqr
df['Humidity9am'] = df['Humidity9am'].apply(lambda x: np.nan if x > seuil_haut or x < seuil_bas else x)

# Remplacement des valeurs extrêmes de 'Humidity3pm' par des valeurs manquantes
q1 = df['Humidity3pm'].quantile(q=0.25)
q3 = df['Humidity3pm'].quantile(q=0.75)
iqr = q3-q1
seuil_haut = q3+1.5*iqr
seuil_bas = q1-1.5*iqr
df['Humidity3pm'] = df['Humidity3pm'].apply(lambda x: np.nan if x > seuil_haut or x < seuil_bas else x)

# Remplacement des valeurs extrêmes de 'Pressure3pm' par des valeurs manquantes
q1 = df['Pressure3pm'].quantile(q=0.25)
q3 = df['Pressure3pm'].quantile(q=0.75)
iqr = q3-q1
seuil_haut = q3+1.5*iqr
seuil_bas = q1-1.5*iqr
df['Pressure3pm'] = df['Pressure3pm'].apply(lambda x: np.nan if x > seuil_haut or x < seuil_bas else x)

# Traitement des données manquantes avec KNN Imputer
imputer = KNNImputer(n_neighbors=1)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

df['Date']=dfDate

#Export du dataframe 
df.to_csv('data/processed/weatherAUS_processed_data_with_date.csv')


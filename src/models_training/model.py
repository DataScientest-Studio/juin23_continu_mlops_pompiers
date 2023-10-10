import numpy as np
from data.make_dataset import df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from joblib import dump

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_data(df) :
    '''
    Sépare les features de la variable cible
    Sépare les données en set d'entrainement et de test
    Normalisation des données sur les ensembles de train et de test
    '''
    X = df.drop('AttendanceTimeSeconds', axis=1)
    y = df['AttendanceTimeSeconds']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test) :
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    dump(scaler, 'models/scaler_fitted.joblib') # Enregistrer le scaler MinMaxScaler() ajusté aux données d'entrainement.
    return X_train_scaled, X_test_scaled

def train_linear_reg(X_train, y_train):
    '''
    Entrainement du modèle de régression linéaire
    '''
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    return model_lr

def train_random_forest(X_train, y_train):
    '''
    Entrainement du modèle de Random Forest
    '''
    model_rf = RandomForestRegressor(n_estimators=200,max_depth=20, random_state=42, n_jobs = -1)
    model_rf.fit(X_train, y_train)
    return model_rf

def train_lightgbm(X_train, y_train):
    '''
    Entrainement du modèle Light GBM
    '''
    model_lgb = lgb.LGBMRegressor()
    model_lgb.fit(X_train, y_train)
    return model_lgb

def pred_model(model, X_test):
    '''
    Obtenir les prédictions du modèle
    '''
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_model(model_name, y_test, y_pred):
    '''
    Calcul et affichage des résultats : 
    MSE, MAE, R², RMSE
    '''

    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

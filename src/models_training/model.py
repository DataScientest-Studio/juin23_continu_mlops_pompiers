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
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, sc

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

    # Affichage des métriques
    print(f"Métriques du modèle {model_name}:")
    print("Mean Squared Error (MSE): ", mse)
    print("Mean Absolute Error (MAE): ", mae)
    print("R-squared (R²): ", r2)
    print("Root Mean Squared Error (RMSE): ", rmse)

    return mse, mae, r2, rmse

# Préparation des données :
X_train, X_test, y_train, y_test, scaler = prepare_data(df)

print(f"colonnes  : {df.drop('AttendanceTimeSeconds', axis=1).columns}")

# Entrainement et résultats du modèle de régression linéaire :
model_lr = train_linear_reg(X_train, y_train)
y_pred_lr = pred_model(model_lr, X_test)
mse_lr, mae_lr, r2_lr, rmse_lr = evaluate_model('Régression linéaire', y_test, y_pred_lr)

# Entrainement et résultats du modèle Light GBM :
model_lgb = train_lightgbm(X_train, y_train)
y_pred_lgb = pred_model(model_lgb, X_test)
mse_lgb, mae_lgb, r2_lgb, rmse_lgb = evaluate_model('Light GBM', y_test, y_pred_lgb)

dump(model_lgb, 'model_lgb.joblib')

# Entrainement et résultats du modèle Random Forest :
# model_rf = train_random_forest(X_train, y_train)
# y_pred_rf = pred_model(model_rf, X_test)
# mse_rf, mae_rf, r2_rf, rmse_rf = evaluate_model('Random Forest', y_test, y_pred_rf)

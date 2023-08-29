import numpy as np
from ..data.make_dataset import df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
import lightgbm as lgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


X = df.drop('AttendanceTimeSeconds', axis=1)
y = df['AttendanceTimeSeconds']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mpe = np.mean((y_test - y_pred) / y_test) * 100

# Affichage des métriques
print("Métriques LR:")
print("Mean Squared Error (MSE): ", mse)
print("Mean Absolute Error (MAE): ", mae)
print("R-squared (R²): ", r2)
print("Root Mean Squared Error (RMSE): ", rmse)
print("Mean Percentage Error (MPE): ", mpe)

print("\nScore LR:")
print('score train :', lr.score(X_train, y_train))
print('score test :', lr.score(X_test, y_test))
print('\n')

model_lgb = lgb.LGBMRegressor()
model_lgb.fit(X_train, y_train)

y_pred = model_lgb.predict(X_test)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mpe = np.mean((y_test - y_pred) / y_test) * 100

# Affichage des métriques
print("Métriques LGB:")
print("Mean Squared Error (MSE): ", mse)
print("Mean Absolute Error (MAE): ", mae)
print("R-squared (R²): ", r2)
print("Root Mean Squared Error (RMSE): ", rmse)
print('\n')
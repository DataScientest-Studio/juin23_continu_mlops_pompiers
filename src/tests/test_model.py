from joblib import load
import pytest

r2_lgb = load('models/r2_lgb.joblib')
rmse_lgb = load('models/rmse_lgb.joblib')

# Vérification de la performance du modèle LGBM
def test_r2():
  assert r2_lgb > 0.5

def test_rmse():
  assert rmse_lgb < 120
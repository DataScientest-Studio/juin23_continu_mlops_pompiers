from model import mse_lgb, mae_lgb, r2_lgb, rmse_lgb

import pytest

def test_r2():
  assert r2_lgb > 0.5

def test_rmse():
  assert rmse_lgb < 120
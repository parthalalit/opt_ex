from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def train_slearner(X_train, y_train):
    
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    yhat_train = model.predict(X_train)

    mse_train = mean_squared_error(y_train, yhat_train)
    r2_train = r2_score(y_train, yhat_train)

    print(f'MSE on train set is {round(mse_train)}')
    print(f'R2 on train set is {round(r2_train, 2)}')
    
    return model, yhat_train
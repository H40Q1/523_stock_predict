import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import math


stocks = ["AAPL", "AMZN", "FB"]
for i in range(len(stocks)):
    stock = stocks[i]
    path = "./data/" + stock + ".csv"
    df = pd.read_csv(path, header=0)
    print("Stock name:", stock)
    print()
    plt.figure()
    lag_plot(df['Open'], lag=20)
    plt.title('Stock - Autocorrelation plot with lag = 20')
    plt.show()

    train_data, test_data = df[0:int(len(df) * 0.7)], df[int(len(df) * 0.7):int(len(df) * 0.775)]
    training_data = train_data['Open'].values
    test_data = test_data['Open'].values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(20, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    MSE_error = mean_squared_error(test_data, model_predictions)
    print('Testing Root Mean Squared Error is {}'.format(math.sqrt(MSE_error)))

    test_set_range = df[int(len(df) * 0.7):int(len(df) * 0.775)].index
    plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed', label='Predicted Price')
    plt.plot(test_set_range, test_data, color='red', label='Actual Price')
    plt.title('FB Prices Prediction')
    plt.xlabel('Date')
    plt.ylabel('Prices')
    plt.legend()
    plt.show()
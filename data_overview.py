import pandas as pd
import matplotlib.pyplot as plt

stocks = ["AAPL", "AMZN", "FB"]

for i in range(len(stocks)):
    stock = stocks[i]
    path = "./data/" + stock + ".csv"
    stock_dataset = pd.read_csv(path, header=0)
    print("Stock name:", stock)
    print()
    print("Data Head:")
    print(stock_dataset.head())
    print()
    print("Data Description:")
    print()
    print(stock_dataset.describe())

    # count total length
    open_price = stock_dataset.Open.values.astype('float32')
    open_price = open_price.reshape(-1, 1)
    print("Total days of stocks:", open_price.shape[0])

    # plot open price
    plt.subplot(3, 1, i+1)
    plt.plot(open_price)
    plt.ylabel("Open prices")
    plt.xlabel("Days")

plt.show()

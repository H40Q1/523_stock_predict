import pandas as pd
import numpy as np
import os
from model import train, loss_plot, predict, evaluate
from args import get_arguments

class Config_Tuning():
    def __init__(self, hidden_size=100, time_step=20, lstm_layers=2, learning_rate=0.00005,
                 epoch=200, batch_size=64, label_columns=[1, 4], feature_start=1, feature_end=7,
                 data_path="./data/AAPL.csv", stock_name = "AAPL"):
        self.feature_columns = list(range(feature_start, feature_end))  # feature columns' indecies
        self.label_columns = label_columns  # predicted feature columns' indecies
        self.label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(self.feature_columns, self.label_columns)

        self.predict_day = 1  # predited length (days)

        self.input_size = len(self.feature_columns)
        self.output_size = len(self.label_columns)

        self.hidden_size = hidden_size  # hiddent size
        self.lstm_layers = lstm_layers  # stacked layer number
        self.dropout_rate = 0.2  # dropout
        self.time_step = time_step  # step size (day)
        self.epoch = epoch
        self.epoch_attention = 50

        self.train_data_rate = 0.7
        self.valid_data_rate = 0.075
        self.test_data_rate = 0.075

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_cuda = False

        # path
        self.model_name = stock_name + ".pth"
        self.train_data_path = data_path
        self.model_save_path = "./checkpoint/" + self.model_name
        self.make_dir()

    def make_dir(self):
        # save
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)


class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()
        self.data = np.array(self.data, dtype=np.float64)
        self.data_num = self.data.shape[0]
        ### interval of dataset
        # x_train [0:s]
        # y_train [d:s+d]
        # x_valid [s:v]
        # y_valid [s+d:v+d]
        # x_test [v:-d]
        # y_test [v+d:-1]

        self.train_num = int(self.data_num * self.config.train_data_rate)  # self.train_num = s

        self.valid_num = int(self.train_num + self.data_num * self.config.valid_data_rate)  # self.valid_num = v

        self.test_num = int(self.valid_num + self.data_num * self.config.test_data_rate)

        self.mean = np.mean(self.data, axis=0)  # mean and std
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean) / self.std  # normalization

    def read_data(self):  # read data
        init_data = pd.read_csv(self.config.train_data_path, usecols=self.config.feature_columns)

        return init_data.values, init_data.columns.tolist()

    def get_train_data(self):
        feature_data = self.norm_data[:self.train_num]  # interval [0:s]

        # interval [d:s+d]
        label_data = self.norm_data[self.config.predict_day: self.config.predict_day + self.train_num,
                     self.config.label_in_feature_index]

        train_x = [feature_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
        train_y = [label_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        return train_x, train_y

    def get_valid_data(self):
        feature_data = self.norm_data[self.train_num: self.valid_num]
        # interval [s:v]

        # interval [s+d:v+d]
        label_data = self.norm_data[self.config.predict_day + self.train_num: self.config.predict_day + self.valid_num,
                     self.config.label_in_feature_index]  # create label

        valid_x = [feature_data[i:i + self.config.time_step] for i in
                   range(self.valid_num - self.train_num - self.config.time_step)]
        valid_y = [label_data[i:i + self.config.time_step] for i in
                   range(self.valid_num - self.train_num - self.config.time_step)]

        valid_x, valid_y = np.array(valid_x), np.array(valid_y)

        return valid_x, valid_y

    def get_test_data(self):
        feature_data = self.norm_data[self.valid_num: self.test_num]
        # feature interval[v:-d]

        test_x = [feature_data[i:i + self.config.time_step] for i in
                  range(self.test_num - self.valid_num - self.config.time_step - 1)]

        test_x = np.array(test_x)

        return test_x

    def return_label(self, dataset):

        if dataset == "train":
            label_data = self.data[self.config.time_step:self.train_num,
                         self.config.label_in_feature_index]
        elif dataset == "valid":
            label_data = self.data[self.train_num + self.config.time_step:self.valid_num,
                         self.config.label_in_feature_index]
        else:
            label_data = self.data[self.valid_num + self.config.time_step:self.test_num - self.config.predict_day,
                         self.config.label_in_feature_index]

        return label_data


def main(config):
    data_gainer = Data(config)
    train_x, train_y = data_gainer.get_train_data()
    valid_x, valid_y = data_gainer.get_valid_data()
    test_x = data_gainer.get_test_data()
    train_label = data_gainer.return_label("train")
    valid_label = data_gainer.return_label("valid")
    test_label = data_gainer.return_label("test")

    # train
    _, tuning_loss_mean = train(train_x, train_y, config)
    loss_plot(tuning_loss_mean)
    # evaluate
    y_pred_valid = predict(valid_x, config)
    evaluate(y_pred_valid, valid_label, data_gainer)


if __name__=="__main__":
    args = get_arguments()
    config = Config_Tuning(hidden_size=args.hidden_size, time_step=args.step, lstm_layers=args.layer,
                                  learning_rate=args.learning_rate, epoch=args.epochs, batch_size= args.batch_size,
                                  data_path=args.stock_path,
                                 stock_name= args.stock_name)
    main(config)


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchnlp.nn as nlpnn


class Net(Module):
    def __init__(self, config, attention_net=False):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                            num_layers=config.lstm_layers, batch_first=True,
                            dropout=config.dropout_rate)
        self.attention = nlpnn.Attention(config.hidden_size)
        self.attention_linear = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.linear = nn.Linear(in_features=config.hidden_size, out_features=config.output_size)
        self.hidden_size = config.hidden_size  # 128
        self.time_step = config.time_step      # 20
        self.attention_net = attention_net     # True: attention layer

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        y = 0
        if self.attention_net:
            query = self.attention_linear(torch.ones(x.shape[0], self.time_step, self.hidden_size))
            attention_out, _ = self.attention(query, lstm_out)
            y = self.linear(attention_out)
        else:
            y = self.linear(lstm_out)
        return y, hidden


def train(x_train, y_train, config, attention_net=False):
    print("Start training ...")
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    net = Net(config, attention_net).to(device)

    train_x, train_y = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=config.batch_size)

    # totally s iterations
    #     s = train_x.shape[0]

    if not attention_net:
        epoches = config.epoch
    else:
        epoches = config.epoch_attention

    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    losses = []  # store losses of each iteration
    epc_mean = []  # store mean losses of each epoch
    for epoch in range(epoches):
        epoch_loss = []
        hidden = None
        for i, data in enumerate(train_loader):
            train_x, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            y_pred, hidden = net.forward(train_x, hidden)
            #             h_t, c_t = hidden
            #             h_t.detach_(), c_t.detach_()
            #             hidden = (h_t, c_t)
            hidden = None
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epoch_loss.append(loss.item())
        cur_loss = np.mean(np.array(epoch_loss))
        if cur_loss < 0.00017:
            break
        print("Epoch {}/{}".format(epoch + 1, config.epoch), " Train Loss :{}".format(cur_loss))
        epc_mean.append(cur_loss)

    torch.save(net.state_dict(), config.model_save_path + config.model_name)
    print('Finished Training Trainset')
    print('Net parameters are saved at {}'.format(config.model_save_path + config.model_name))
    return losses, epc_mean


def loss_plot(losses):
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.show()


def predict(x_test, config, attention_net=False):
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    y_pred = torch.empty((0, len(config.label_columns))).to(device)
    y_hat = []
    test_X = torch.from_numpy(x_test).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    net = Net(config, attention_net).to(device)
    net.load_state_dict(torch.load(config.model_save_path + config.model_name))
    net.eval()
    hidden = None
    for data in test_loader:
        tmp = []
        x = data[0].to(device)
        y, hidden = net.forward(x, hidden)
        hidden = None
        #         y_pred_0 = torch.cat((y_pred, y[0]), 0)
        tmp.append(y[0][-1][0].item())
        tmp.append(y[0][-1][1].item())
        y_hat.append(tmp)
    return np.array(y_hat)

def up_down_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_var_test=y_true[1:]-y_true[:len(y_true)-1]
    y_var_predict=y_pred[1:]-y_pred[:len(y_pred)-1]
    txt=np.zeros(len(y_var_test))
    for i in range(len(y_var_test-1)):#计算数量
        txt[i]=np.sign(y_var_test[i])==np.sign(y_var_predict[i])
    result=sum(txt)/len(txt)
    return result


def evaluate(y_pred, y_test, data_gainer, days=100):
    labels_open = []
    labels_close = []
    for i in range(y_test.shape[0]):
        labels_open.append(y_test[i][0])
    for i in range(y_test.shape[0]):
        labels_close.append(y_test[i][1])

    print("###############################################################")
    print("Evaluation of open price predction on test set:")

    y_pred_0 = y_pred[:, 0] * data_gainer.std[0] + data_gainer.mean[0]

    # Error comptuer of open price prediction
    # Root Mean Square Error
    RMSE = np.sqrt(np.sum((np.array(labels_open) - y_pred_0) ** 2) / len(labels_open))
    # Mean Absolute Percentage Error
    MAPE = np.sum((np.array(labels_open) - y_pred_0) / np.array(labels_open)) / len(labels_open) * 100
    # Mean Bias Error
    MBE = np.sum((np.array(labels_open) - y_pred_0)) / len(labels_open)
    print("RMSE on validation set is {}".format(RMSE))
    print("MAPE on validation set is {}".format(MAPE))
    print("MBE on validation set is {}".format(MBE))
    up_down_accu = up_down_accuracy(labels_open, y_pred_0)
    print("Up and down accuracy on validation set is {}%".format(round(up_down_accu * 100), 2))

    plt.subplot(2,1,1)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Evaluation of Open prices on test set for 100 days')
    plt.plot(y_pred_0.tolist()[:days], 'r', label="predict")
    plt.plot(labels_open[:days], 'b', label="real")
    plt.legend(loc="upper right")

    # Error comptuer of close price prediction

    print("###############################################################")
    print("Evaluation of close price predction on valid set:")
    y_pred_1 = y_pred[:, 1] * data_gainer.std[1] + data_gainer.mean[1]

    # Error comptuer of open price prediction
    # Root Mean Square Error
    RMSE = np.sqrt(np.sum((np.array(labels_close) - y_pred_1) ** 2) / len(labels_close))
    # Mean Absolute Percentage Error
    MAPE = np.sum((np.array(labels_close) - y_pred_1) / np.array(labels_close)) / len(labels_close) * 100
    # Mean Bias Error
    MBE = np.sum((np.array(labels_close) - y_pred_1)) / len(labels_close)
    print("RMSE on validation set is {}".format(RMSE))
    print("MAPE on validation set is {}%".format(MAPE))
    print("MBE on validation set is {}".format(MBE))
    up_down_accu = up_down_accuracy(labels_close, y_pred_1)
    print("Up and down accuracy on validation set is {}%".format(round(up_down_accu * 100), 2))

    plt.subplot(2, 1, 2)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Evaluation of Close prices on test set for 100 days')
    plt.plot(y_pred_1.tolist()[:days], 'r', label="predict close")

    plt.plot(labels_close[:days], 'b', label="real close")
    plt.legend(loc="upper right")
    plt.show()
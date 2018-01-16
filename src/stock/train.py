# coding=utf-8
from math import sqrt

import sklearn
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from stock import data_process

N = 8
# integer encode direction
encoder = LabelEncoder()

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def fix_values():
    # load dataset
    st_df = data_process.get_k_data()
    dataset = data_process.process_data(st_df)
    values = dataset.values

    values[:, N] = encoder.fit_transform(values[:, N])


    # ensure all data is float
    values = values.astype('float32')

    # normalize features
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    # drop columns we don't want to predict
    reframed.drop(reframed.columns[list(range(N, N * 2 + 1))], axis=1, inplace=True)

    print(reframed.head())

    return reframed.values


def split_train_test_data(values):
    # split into train and test sets
    split_index = int(len(values) * 0.7)

    train = values[:split_index, :]
    test = values[split_index:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    return train_X, train_y, test_X, test_y


def train(train_X, train_y, test_X, test_y):
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=1500, batch_size=30, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)

    return model, history


if __name__ == '__main__':

    values = fix_values()
    train_X, train_y, test_X, test_y = split_train_test_data(values)
    model, history = train(train_X, train_y, test_X, test_y)


    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()



    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    test_y = test_y.reshape((test_y.shape[0], 1))

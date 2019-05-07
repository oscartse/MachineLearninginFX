import pandas as pd
import talib
from keras.callbacks import EarlyStopping
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense ,Dropout
from keras.layers import LSTM
from datetime import datetime
from talib import MA_Type
import logging
from sklearn.metrics import mean_squared_error

early_stopper = EarlyStopping(patience=5)

def getdata(path):

    """Read data from CSV"""

    raw = pd.read_csv(path,usecols=[0,1,2,3,4,5],skiprows=1)
    #path = file location from main()
    raw.columns = ['Date','Open','High','Low','Close','Volumn']
    raw['Date'] = raw['Date'].apply(lambda y: datetime.strptime(y,'%d.%m.%Y %H:%M:%S.000'))
    raw = raw[(raw['Date'].dt.year >=2015)] #data range(2011-2018)
    raw = raw.drop_duplicates(keep=False)

    return raw

def candlebar_analysis(raw):

    openprice=raw['Open'].astype(float).values
    highprice=raw['High'].astype(float).values
    lowprice=raw['Low'].astype(float).values
    closeprice=raw['Close'].astype(float).values
    avg_price = talib.AVGPRICE(openprice, highprice, lowprice, closeprice)

    return openprice, highprice, lowprice, closeprice, avg_price

def technical_analysis(openprice, highprice, lowprice, closeprice, avg_price):

    sma_240 = talib.SMA(openprice, timeperiod=240) #sma_240 = 10 Days Avg
    ATR =talib.ATR(openprice,lowprice,closeprice,timeperiod=24)
    #RSI = talib.RSI(closeprice,120)
    bollupper,bollmiddle,bolllower = talib.BBANDS(closeprice,nbdevup=2, nbdevdn=2, matype=MA_Type.T3)

    return avg_price, sma_240, bollupper, bollmiddle, bolllower, ATR

def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def reshaping(avg_price, sma_240, bollupper,bollmiddle, bolllower, ATR):

    batch_size = 1000
    nb_classes = 48
    dataset = pd.DataFrame(data=[avg_price, sma_240, bollupper,bollmiddle, bolllower, ATR]).transpose()
    dataset.columns = ['avg_price', 'SMA240', 'BBupper', 'BBMiddle' ,'BBLower', 'ATR']  # NAMING COLUMN
    dataset.dropna(inplace=True)

    data = dataset.values
    data = data.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data=series_to_supervised(data,1,1)
    data.drop(data.columns[[7,8, 9, 10, 11]], axis=1, inplace=True)
    data = data.values #DF to list
    train_size = int(len(data)*0.7) #70% train ,30% test
    trainset = data[: train_size, :]
    testset = data[train_size :, :]
    x_train, y_train = trainset[:, :-1], trainset[:, -1]
    x_test, y_test = testset[:, :-1], testset[:, -1]
    x_train = x_train.reshape((x_train.shape[0],1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0],1, x_test.shape[1]))

    input_shape = x_train.shape

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test,scaler)

def compile_model(network, nb_classes, input_shape):
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    dropout = network['dropout_rate']

    model = Sequential()
    for i in range(nb_layers):
        if i == 0:
            model.add(LSTM(nb_neurons, activation=activation, return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
            model.add(Dropout(dropout))
        else:
            model.add(LSTM(nb_neurons, return_sequences=True))
    model.add(LSTM(nb_neurons, return_sequences=False))  # final LSTM layer
    model.add(Dropout(dropout))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer,
                  metrics=['mse', 'mae', 'mape'])

    return model

def train_and_score(network, dataset, path):
    if dataset == 'EURUSD60':

        raw = getdata(path)
        openprice, highprice, lowprice, closeprice, avg_price = candlebar_analysis(raw)
        avg_price, sma_240, bollupper, bollmiddle, bolllower, ATR = technical_analysis(openprice, highprice, lowprice,
                                                                                       closeprice, avg_price)
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, scaler = reshaping(avg_price, sma_240,
                                                                                                  bollupper, bollmiddle,
                                                                                                  bolllower, ATR)

    model = compile_model(network, nb_classes, input_shape)
    epoch = 2

    history = model.fit(x_train, y_train,  epochs=epoch,  batch_size=batch_size, verbose=0)

    trainPredict = model.predict(x_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[2]))  # reshape the data
    trainresult = np.concatenate((trainPredict, x_train[:, 1:]), axis=1)

    trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))

    # calculate the SD SCORE
    trainresult = scaler.inverse_transform(trainresult)
    trainPredict = trainresult[:, 0]
    """
    testPredict = model.predict(x_test)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))  # reshape the data
    testresult = np.concatenate((testPredict, x_test[:, 1:]), axis=1)

    testScore = math.sqrt(mean_squared_error(y_test, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    testresult = scaler.inverse_transform(testresult)
    testPredict = testresult[:, 0]

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))
    """
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f'Score:{score[1]}')
    return score[1]  # 1 is accuracy. 0 is loss.

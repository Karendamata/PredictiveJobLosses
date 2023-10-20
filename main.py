import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from openpyxl import Workbook

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sn
from sklearn import preprocessing
from keras.callbacks import EarlyStopping

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras import Model

import time
from sklearn.metrics import mean_squared_error
import math

def load_data(dataframe, seq_len):
    rawdata = dataframe.values
    dataset = []
    global train_size, valid_set_size, test_set_size

    for i in range(0, len(rawdata)-seq_len+1):
        dataset.append(rawdata[i:(i+seq_len), :])
    dataset = np.array(dataset)
    valid_set_size = int(np.around((valid_set_size_perc / 100) * dataset.shape[0]))
    test_set_size = int(np.around((test_set_size_perc / 100) * dataset.shape[0]))
    train_size = dataset.shape[0] - (valid_set_size + test_set_size)

    print("Size: Train: ", train_size, "Validation: ", valid_set_size, "Testing: ", test_set_size)

    X = dataset[:, :-1, :n_covariates]
    Y = dataset[:, -1, n_covariates]
    Delta_Y = dataset[:, -1, n_covariates+1]

    X_train = dataset[:train_size, :-1, :n_covariates]
    Y_train = dataset[:train_size, -1, n_covariates]
    Delta_Y_train = dataset[:train_size, -1, n_covariates+1]

    X_valid = dataset[train_size:train_size+valid_set_size, :-1, :n_covariates]
    Y_valid = dataset[train_size:train_size+valid_set_size, -1, n_covariates]
    Delta_Y_valid = dataset[train_size:train_size+valid_set_size, -1, n_covariates+1]

    X_test = dataset[train_size + valid_set_size:, :-1, :n_covariates]
    Y_test = dataset[train_size + valid_set_size:, -1, n_covariates]
    Delta_Y_test = dataset[train_size + valid_set_size:, -1, n_covariates+1]
    
    return dict(X = X, Y = Y, Delta_Y = Delta_Y,
                X_train =  X_train, Y_train = Y_train, Delta_Y_train = Delta_Y_train,
                X_valid = X_valid, Y_valid = Y_valid, Delta_Y_valid=Delta_Y_valid,
                X_test = X_test, Y_test = Y_test, Delta_Y_test=Delta_Y_test)

def goodness(predictions, y, n_covariates, All):
    predictions = np.array(predictions)
    y = np.array(y).flatten()
    num = len(y)
    sse = np.sum(pow(y-predictions, 2))
    rmse = np.sqrt(sse/(num))

    mape = np.sum(np.abs((y - predictions)/y))*100/num

    prr = np.sum(pow((predictions - y)/predictions, 2))

    pp = np.sum(pow((predictions - y)/y, 2))

    ssy = np.sum(pow(y - np.mean(y), 2))
    R2 = (ssy - sse)/ssy
    R2Adj = 1 - (1 - R2)*(num - 1)/(num - n_covariates - 1)
    if All==1:
        results = {"R2": R2, "R2Adj": R2Adj, "SSE": sse, "RMSE": rmse, "MAPE": mape}
    elif All==2:
        results = {"VSSE": sse, "VRMSE": rmse, "VRR": prr}
    else:
        results = {"PSSE": sse, "PRMSE": rmse, "PRR": prr, "PP": pp}

    return results

def Ytransform(prediction, y):
    Cumulative_Y = [y[0] + prediction[0]]
    for i in range(1, len(prediction)):
        aux = Cumulative_Y[i - 1] + prediction[i]
        Cumulative_Y.append(aux)
    Cumulative_Y = np.array(Cumulative_Y)
    return Cumulative_Y

def create_excel(path):
    workbook = Workbook()
    workbook.save(path)



data = pd.read_excel("dataset.xlsx", index_col=0, engine="openpyxl")

df = data.copy()
# df = df.loc[:, ['X19', 'Y', 'Delta Y']]
# df = df.loc[:, ['X19', 'X14', 'Y', 'Delta Y']]
# df = df.loc[:, ['X19', 'X14', 'X4', 'Y', 'Delta Y']]
df = df.loc[:, ['X19', 'X14', 'X4', 'X7', 'Y', 'Delta Y']]

m = df.shape[0]
n_covariates = df.shape[1]-2
seq_len = 2

valid_set_size_perc = 15
test_set_size_perc = 15

window_data = load_data(df, seq_len)

#Parameters
n_outputs = 1
n_epochs = 1000
Alpha = 0.01
total_size = window_data['X'].shape[0]
loss_function = 'mse'
optimizer = 'Adam'
algorithm = 'GRU'


parameters_dict = {"train_size": train_size,
                   "valid_set_size": valid_set_size, 
                   "test_set_size": test_set_size,
                   "Lags": seq_len,
                   "Max n_epochs": n_epochs,
                   "Learning Rate": Alpha,
                   "Model": algorithm,
                   "Covariates": df.columns.to_list()[:-2]}

def modelvalidation(nl, nepochs, alpha, NN):
    if NN == 1:
        inputs = keras.Input(shape=(1, n_covariates))
        flat = layers.Flatten()(inputs)
        ANNL1 = layers.Dense(nl[0], activation='relu')(flat)
        output = layers.Dense(n_outputs)(ANNL1)
        model = keras.Model(inputs=inputs, outputs=output, name="SingleANN")
    elif NN == 2:
        inputs = keras.Input(shape=(1, n_covariates))
        RNNL1 = layers.SimpleRNN(nl[0], activation='tanh')(inputs)
        dropout1 = layers.Dropout(0.2)(RNNL1)
        output = layers.Dense(n_outputs)(dropout1)
        model = keras.Model(inputs=inputs, outputs=output, name="SingleRNN")
    elif NN==3:
        inputs = keras.Input(shape=(1, n_covariates))
        LSTML1 = layers.LSTM(nl[0], activation='tanh')(inputs)
        output = layers.Dense(n_outputs, activation='linear')(LSTML1)
        model = keras.Model(inputs=inputs, outputs=output, name="SingleLSTM")
    else:
        inputs = keras.Input(shape=(1, n_covariates))
        GRU = layers.GRU(nl[0], activation='tanh')(inputs)
        output = layers.Dense(n_outputs, activation='linear')(GRU)
        model = keras.Model(inputs=inputs, outputs=output, name='SingleGRU')

    opt = keras.optimizers.legacy.Adam(learning_rate=alpha)
    model.compile(loss=loss_function, optimizer=opt, metrics=['mae'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='mae', min_delta=0.0001, patience=100)

    t_start = time.time()
    testmodel = model.fit(window_data['X_train'], window_data['Delta_Y_train'], epochs=nepochs,
                          validation_data=(window_data['X_valid'], window_data['Delta_Y_valid']), verbose=0, callbacks=[callback])
    t_end = time.time()

    lstm_time = t_end - t_start
    n_epochs_earlier = len(pd.DataFrame(testmodel.history).values[:, 0])
    Delta_Y_pred = model.predict(window_data['X'], verbose=0).reshape(total_size, )
    Y_cumulative = Ytransform(Delta_Y_pred, window_data['Y'])

    num_par = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    return [Delta_Y_pred, Y_cumulative, n_epochs_earlier, lstm_time, testmodel.history, num_par]

Y_pred_avg = pd.DataFrame()
Y_pred_std = pd.DataFrame()
DeltaY_pred_avg = pd.DataFrame()
DeltaY_pred_std = pd.DataFrame()
parameters_df = pd.DataFrame()
loss_df = pd.DataFrame()
val_loss_df = pd.DataFrame()
goodness_df = pd.DataFrame()
repetitions = 50
# vec = [1, 2]
vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15]


create_excel('AVGresults.xlsx')
writer = pd.ExcelWriter('AVGresults.xlsx', engine='xlsxwriter')



nn = 4
for neuron in vec:
    preds = pd.DataFrame()
    Deltapred = pd.DataFrame()
    loss = pd.DataFrame()
    val_loss = pd.DataFrame()
    EarlierEpochs = []
    running_time = []

    for i in range(repetitions):
        tf.keras.backend.clear_session()
        pred = modelvalidation([neuron], n_epochs, Alpha, nn)
        Deltapred.insert(i, '', pred[0], True)
        preds.insert(i, '', pred[1], True)
        EarlierEpochs.append(pred[2])
        running_time.append(pred[3])
        loss = pd.concat([loss, pd.DataFrame(pred[4]['loss'])], axis=1)
        val_loss = pd.concat([val_loss, pd.DataFrame(pred[4]['val_loss'])], axis=1)
        num_par = pred[5]

    parameters_dict['mean_running_epochs'] = float(np.mean(EarlierEpochs))
    parameters_dict['mean_running_time'] = float(np.mean(running_time))
    parameters_dict['num_par'] = int(num_par)

    Y_pred_avg[str(neuron)+ ' neuron'] = preds.mean(axis=1)
    Y_pred_std[str(neuron)+ ' neuron'] = preds.std(axis=1)

    DeltaY_pred_avg[str(neuron)+ ' neuron'] = Deltapred.mean(axis=1)
    DeltaY_pred_std[str(neuron)+ ' neuron'] = Deltapred.std(axis=1)

    loss_df['Avg '+ str(neuron)+ ' neuron'] = loss.mean(axis=1)
    loss_df['Std '+ str(neuron)+ ' neuron'] = loss.std(axis=1)

    val_loss_df['Avg '+ str(neuron)+ ' neuron'] = val_loss.mean(axis=1)
    val_loss_df['Std '+ str(neuron)+ ' neuron'] = val_loss.std(axis=1)

    prediction = Y_pred_avg[str(neuron)+ ' neuron']
    parameters_dict.update(goodness(prediction, window_data['Y'], num_par, All=1))
    parameters_dict.update(goodness(prediction[(total_size-valid_set_size-test_set_size):(total_size-test_set_size)], window_data['Y_valid'], num_par, All=2))
    parameters_dict.update(goodness(prediction[total_size-test_set_size:], window_data['Y_test'], num_par, All=3))

    parameters_df[str(neuron)+ ' neuron'] = parameters_dict

    print("Finished Neuron: ", neuron)

pd.DataFrame(parameters_df).to_excel(writer, sheet_name='Parameters')
Y_pred_avg.to_excel(writer, sheet_name='Y Predictions')
Y_pred_std.to_excel(writer, sheet_name='Y Predictions_std')
DeltaY_pred_avg.to_excel(writer, sheet_name='Delta Y predictions')
DeltaY_pred_std.to_excel(writer, sheet_name='Delta Y predictions_std')
loss_df.to_excel(writer, sheet_name='loss')
val_loss_df.to_excel(writer, sheet_name='val_loss')

writer.close()
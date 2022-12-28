import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import max_error, mean_absolute_error, r2_score
import mlflow
import mlflow.keras

def prepare_model(SOURCE_DATA_POWER,
                  data_column_name,
                  seq_size,
                  n_neurons,
                  drop,
                  epochs_num,
                  validation_split,
                  batch_size):
    df = pd.read_csv(SOURCE_DATA_POWER, sep = ';',parse_dates=[0])

    df['Time'] = pd.to_datetime(df['Time'])
    df.index = df['Time']

    for i in range(df.shape[0]):
      if np.isnan(df[data_column_name][i]):
        df[data_column_name][i] = 0

    load = list(df[data_column_name])
    min_load = min(load)
    max_load = max(load)
    if min_load >= 0:
      load[:] = [a - min_load for a in load]
      load[:] = [a / max_load for a in load]
    else:
      load[:] = [a + abs(min_load) for a in load]
      load[:] = [a / (max_load+ abs(min_load)) for a in load]

    month = list(df.index.month)
    min_month = min(month)
    max_month = max(month)
    month[:] = [a - min_month for a in month]
    month[:] = [a / max_month for a in month]

    week = list(df.index.isocalendar().week.astype(np.int64))
    min_week = min(week)
    max_week = max(week)
    week[:] = [a - min_week for a in week]
    week[:] = [a / max_week for a in week]

    day = list(df.index.dayofweek)
    min_day = min(day)
    max_day = max(day)
    day[:] = [a - min_day for a in day]
    day[:] = [a / max_day for a in day]

    data = [load, month, week, day]

    x = []
    y = []
    
    for i in range(0, len(data[0]) - len(data[0])%int(seq_size*3/2) - int(seq_size*3/2), 24):
      y.append(data[0][i + seq_size:i + int(seq_size*3/2)])
      for j in range(seq_size):
        tempx = []
        for k in range(0, 4):
          tempx.append(data[k][i+j])
        x.append(tempx)
        

    x = np.array(x)
    x = np.reshape(x, (int(x.shape[0]/seq_size), seq_size, x.shape[1]))

    y = np.array(y)
    y = np.reshape(y, (int(y.shape[0]), int(seq_size/2), 1))

    #print(x.shape, y.shape)

    x_train_size = int(len(x) * 0.85)
    y_train_size = int(len(y) * 0.85)

    x_train = x[0:x_train_size]
    x_train = np.array(x_train)

    x_test = x[x_train_size:len(x)]
    x_test = np.array(x_test)

    y_train = y[0:y_train_size]
    y_train = np.array(y_train)

    y_test = y[y_train_size:len(y)]
    y_test = np.array(y_test)

    model = Sequential()

    model.add(LSTM(n_neurons, input_shape=(96, 4), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(LSTM(n_neurons, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(LSTM(n_neurons))
    model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(Dense(int(seq_size/2), activation="sigmoid"))

    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size, epochs=epochs_num, verbose=2)

    return model, x_test, y_test, x, y, min_load, max_load, data

def predict_and_plot(k, hours_num):

  inp = x_test[k]

  inp = np.reshape(inp, (1, seq_size, x.shape[2]))
  y_pred = model.predict(inp, verbose=0).tolist()[0]

  if min_load >= 0:
    p = np.array(y_pred)
    p *= max_load
    p += min_load

    n = 0
    for n in range(200):

      real_num = int(len(data[0])*0.85) + 24*k - n

      real = data[0].copy()
      real = np.array(real[real_num - hours_num:real_num + int(seq_size/2)])
      real *= max_load
      real += min_load

      inp_copy = np.array([inp[0][i][0] for i in range(0, hours_num)])
      inp_copy *= max_load
      inp_copy += min_load
      inp_copy = inp_copy.tolist()

      n += 1

      if real[:hours_num].tolist() == inp_copy:
        #print(n)
        break

  else:
    p = np.array(y_pred)
    p *= (max_load + abs(min_load))
    p -= abs(min_load)

    n = 0
    for n in range(200):

      real_num = int(len(data[0])*0.85) + 24*k - n

      real = data[0].copy()
      real = np.array(real[real_num - hours_num:real_num + int(seq_size/2)])
      real *= (max_load + abs(min_load))
      real -= abs(min_load)

      inp_copy = np.array([inp[0][i][0] for i in range(0, hours_num)])
      inp_copy *= (max_load + abs(min_load))
      inp_copy -= abs(min_load)
      inp_copy = inp_copy.tolist()

      n += 1

      if real[:hours_num].tolist() == inp_copy:
        #print(n)
        break

  g = [i for i in range(hours_num, len(p)+hours_num)]
  s = [i for i in range(hours_num)]
  h = [i for i in range(len(real))]

  f = plt.figure()
  f.set_figwidth(30)
  plt.plot(s, inp_copy, color='b', label='Input LSTM', linestyle = 'dotted')
  plt.plot(g, p, color='r', label='LSTM')
  plt.plot(h, real, color='g', label='Real')

  real_copy = real[hours_num:]
  p_copy = p[:]

  me = max_error(real_copy, p_copy)
  mae = mean_absolute_error(real_copy, p_copy)
  r2s = r2_score(real_copy, p_copy)
  print('Max error - %.2f' % me)
  print('Absolute error - %.2f' % mae)
  print('R2 score - %.2f' % r2s)
  print('')

  plt.legend()
  #plt.show()
  plt.savefig(fname = plots_path + plot_name)

  return me, mae, r2s

mlflow.tensorflow.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_id = 571625146127493926

with mlflow.start_run(experiment_id=experiment_id) as run:
    SOURCE_DATA_POWER = './data/fp_archives FACT_PRED_P (Nura220_jan_novem22).csv'
    data_column_name = 'FACT_P_L'
    seq_size = 96
    n_neurons  = 150
    drop = 0.4
    epochs_num = 100
    validation_split=0.1
    batch_size=16

    model, x_test, y_test, x, y, min_load, max_load, data = prepare_model(SOURCE_DATA_POWER=SOURCE_DATA_POWER,
                                          data_column_name=data_column_name,
                                          seq_size=seq_size,
                                          n_neurons=n_neurons,
                                          drop=drop,
                                          epochs_num=epochs_num,
                                          validation_split=validation_split,
                                          batch_size=batch_size)

    print('')
    model.evaluate(x_test, y_test, verbose=2)
    print('')

    hours_num = 96
    max_k = 50
    step = 5
    plots_path = './mlruns/' + str(experiment_id) + '/' + str(run.info.run_id) + '/plots'
    os.mkdir(plots_path)
    for l in range(0, max_k, step):
      print('k = ' + str(l))
      plot_name = '/plot' + str(l)
      me, mae, r2s = predict_and_plot(l, hours_num)

      mlflow.log_artifact(plots_path)
      metrics = {'Max error': me, 'Absolute error': mae,'R2 score': r2s}
      mlflow.log_metrics(metrics)


#mlflow.keras.save_model(model, './models/1')

#model = mlflow.keras.load_model('./models/1')

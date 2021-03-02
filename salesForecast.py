import sys
import re
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, LSTM, GRU, Flatten
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dropout, Masking
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib.pyplot as plt
from tcn import TCN



# Diccionarios para guardar los resultados
res_pronos = defaultdict(lambda: {})
res_tiempos = defaultdict(lambda: {})

"""___

# Funciones

### Funciones Auxiliares
"""
def plot_line(df, ttl, path):
    plt.plot(df)
    plt.title(ttl, fontsize=12)
    plt.yticks(fontsize=10)
    plt.xticks(pd.date_range(df.index.min(),
                             df.index.max(),
                             freq='7D'),
               rotation='vertical',
               size=5)
    plt.xlabel(df.index.name, fontsize=10)
    plt.ylabel(df.columns[0], fontsize=10)
    plt.savefig(os.path.join(path, df.columns[0]+"-"+df.index.name+".pdf"))
    plt.show()

def plot_pronos_sarima(predic, test):
    pred_val = pred.conf_int()
    ax = test.plot()
    predic.predicted_mean.plot(ax=ax, label='One-step ahead Forecast',
                               alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_val.index,
                    pred_val.iloc[:, 0],
                    pred_val.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel(test.columns[0])
    plt.legend()
    plt.show()

def plot_pie(df, ttl, path):
    y = df.iloc[:,0]
    x = df.index
    total_cant_dia = y.sum()
    total_cant_dia = str(int(total_cant_dia))

    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{v:d}'.format(v=val)

        return my_format

    colors = ['#BC243C', '#FE840E', "#3324bc", "#24bc4d", '#C62168', "#70bc24", "#bcad24"]
    explode = tuple([0.05] * len(x))
    fig1, ax1 = plt.subplots()
    plt.title(ttl, fontsize=18)
    ax1.pie(y,
            colors=colors,
            labels=x,
            autopct=autopct_format(y),
            startangle=90,
            explode=explode)
    centre_circle = plt.Circle((0, 0), 0.82, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax1.axis('equal')

    label = ax1.annotate('Total \n' + str(total_cant_dia),
                         color='red',
                         xy=(0, 0),
                         fontsize=15,
                         ha="center")
    plt.savefig(os.path.join(path, df.columns[0]+"-"+df.index.name+"_pie.pdf"))
    plt.tight_layout()
    plt.show()


def plot_pie_cant(x, y):
    total_cant_dia = y.sum()
    total_cant_dia = str(int(total_cant_dia))

    plt.rcParams['font.size'] = 10.0
    plt.rcParams['font.weight'] = 6

    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return ' {v:d}'.format(v=val)

        return my_format

    colors = ['#BC243C', '#FE840E', '#C62168']
    explode = explode = tuple([0.05] * len(x))
    fig1, ax1 = plt.subplots()

    ax1.pie(y,
            colors=colors,
            labels=x,
            autopct=autopct_format(y),
            startangle=90,
            explode=explode)
    centre_circle = plt.Circle((0, 0), 0.82, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax1.axis('equal')

    label = ax1.annotate('Total \n' + str(total_cant_dia),
                         color='red',
                         xy=(0, 0),
                         fontsize=10,
                         ha="center")
    plt.tight_layout()
    plt.show()


def plot_pie_comi(x, y):
    total_cant_dia = y.sum()
    total_cant_dia = str(int(total_cant_dia))
    plt.rcParams["figure.figsize"] = (13, 5)
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.weight'] = 6

    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '${v:d}'.format(v=val)

        return my_format

    colors = ['#BC243C', '#FE840E', '#C62168']
    explode = tuple([0.05] * len(x))
    fig1, ax1 = plt.subplots()

    ax1.pie(y,
            colors=colors,
            labels=x,
            autopct=autopct_format(y),
            startangle=90,
            explode=explode)
    centre_circle = plt.Circle((0, 0), 0.82, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax1.axis('equal')

    label = ax1.annotate('Total \n' + '$' + str(total_cant_dia),
                         color='red',
                         xy=(0, 0),
                         fontsize=10,
                         ha="center")
    plt.tight_layout()
    plt.show()


def plot_bar(df, ttl, path):
    plt.title(ttl, fontsize=18)
    plt.bar(df.index, df.iloc[:,0], color='#99ff99', edgecolor='green', linewidth=1)
    plt.xlabel(df.index.name, fontsize=15)
    plt.ylabel(df.columns[0], fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for k, v in df.iloc[:, 0].items():
        plt.text(k,
                 v,
                 str(round(v)),
                 fontsize=12,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='bottom')

    plt.savefig(os.path.join(path, df.columns[0]+"-"+df.index.name+".pdf"))
    plt.show()


def plot_metrics(column, res_pronos_df, plot_mercado_path, RE):
    res_pronos_df = res_pronos_df.filter(regex=RE, axis=0)
    res_pronos_df.sort_values(by=['MAE'], ascending=False).plot.barh(
        y='MAE', legend=None, figsize=(30, len(res_pronos_df)))
    plt.title(column.replace("_", " ") + '\nError absoluto medio', size=18)
    mae_plot_path = os.path.join(plot_mercado_path, "MAE.png")
    plt.savefig(mae_plot_path)
    plt.show()
    res_pronos_df.sort_values(by=['MSE'], ascending=False).plot.barh(
        y='MSE', legend=None, figsize=(30, len(res_pronos_df)))
    plt.title(column.replace("_", " ") + '\nError cuadrático medio', size=18)
    mse_plot_path = os.path.join(plot_mercado_path, "MSE.png")
    plt.savefig(mse_plot_path)
    plt.show()
    res_pronos_df.sort_values(by=['R2'], ascending=True).plot.barh(
        y='R2', legend=None, figsize=(30, len(res_pronos_df)))
    plt.title(column.replace("_", " ") + '\nCoeficiente de determinación',
              size=18)
    r2_plot_path = os.path.join(plot_mercado_path, "R2.png")
    plt.savefig(r2_plot_path)
    plt.show()
    res_pronos_df.sort_values(by=['EVS'], ascending=True).plot.barh(
        y='EVS', legend=None, figsize=(30, len(res_pronos_df)))
    plt.title(column.replace("_", " ") + '\nVarianza explicada', size=18)
    evs_plot_path = os.path.join(plot_mercado_path, "EVS.png")
    plt.savefig(evs_plot_path)
    plt.show()
    res_pronos_df.sort_values(by=['ME'], ascending=False).plot.barh(
        y='ME', legend=None, figsize=(30, len(res_pronos_df)))
    plt.title(column.replace("_", " ") + '\nError máximo', size=18)
    me_plot_path = os.path.join(plot_mercado_path, "ME.png")
    plt.savefig(me_plot_path)
    plt.show()
    res_pronos_df.sort_values(by=['MAPE'], ascending=False).plot.barh(
        y='MAPE', legend=None, figsize=(30, len(res_pronos_df)))
    plt.title(column.replace("_", " ") + '\nError Porcentual Absoluto Medio',
              size=18)
    mape_plot_path = os.path.join(plot_mercado_path, "MAPE.png")
    plt.savefig(mape_plot_path)
    plt.show()


def plot_pronos(data, modelo, ruta, column):
    '''Plot de los pronósticos'''
    data.filter(items=[column, modelo], axis=1).plot(figsize=(30, 10))
    plt.ylabel(column)
    plt.xlabel('Fecha')
    plt.title(column.replace("_", " ") +
              "\nPronósticos de %i hs." % len(data) +
              "\n Modelo %s" % modelo.rstrip("-" + column).replace("_", " "),
              size=24)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.autoscale()
    plt.savefig(ruta, bbox_inches="tight")
    plt.show()


def mk_model_dirs(name, m_path, t_path, p_path):
    '''Crea las carpetas y los paths hacia las carpetas de los modelos'''
    modelo_path = os.path.join(m_path, name)
    tabla_path = os.path.join(t_path, name)
    plot_path = os.path.join(p_path, name)
    # log_path = os.path.join(logs_path, name)
    for model_dirs in [modelo_path, tabla_path, plot_path]:
        if not os.path.exists(model_dirs):
            try:
                os.makedirs(model_dirs, 0o700)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            print("Directorio creado: {}".format(model_dirs))
        else:
            print("Directorio ya existente: {}".format(model_dirs))
    return modelo_path, tabla_path, plot_path


def mk_model_mercado_dirs(mercado, mo_path, ta_path, pl_path):
    modelo_mercado_path = os.path.join(mo_path, mercado)
    tabla_mercado_path = os.path.join(ta_path, mercado)
    plot_mercado_path = os.path.join(pl_path, mercado)
    # log_mercado_path = os.path.join(log_path, mercado)
    for model_mercado_dirs in [
        modelo_mercado_path, tabla_mercado_path, plot_mercado_path
    ]:
        if not os.path.exists(model_mercado_dirs):
            try:
                os.makedirs(model_mercado_dirs, 0o700)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            print("Directorio creado: {}".format(model_mercado_dirs))
        else:
            print("Directorio ya existente: {}".format(model_mercado_dirs))
    return modelo_mercado_path, tabla_mercado_path, plot_mercado_path


def mk_dir(name):
    '''crear el directorio con el nombre que le pasas'''
    path = os.path.join(os.getcwd(), name)
    if not os.path.exists(path):
        try:
            os.makedirs(path, 0o700)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        print("Directorio creado: {}".format(path))
    else:
        print("Directorio ya existente: {}".format(path))
    return path


def traducir_dia(text):
    t = text.replace('Sunday', 'Domingo')
    t = t.replace('Monday', 'Lunes')
    t = t.replace('Tuesday', 'Martes')
    t = t.replace('Wednesday', 'Miércoles')
    t = t.replace('Thursday', 'Jueves')
    t = t.replace('Friday', 'Viernes')
    t = t.replace('Saturday', 'Sábado')
    return t


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def metrics_time_series(y_true, y_pred):
    me = round(max_error(y_true, y_pred), 3)
    r2 = round(r2_score(y_true, y_pred), 3)
    evs = round(explained_variance_score(y_true, y_pred), 3)
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    mse = round(mean_squared_error(y_true, y_pred), 3)
    mape = round(mean_absolute_percentage_error(y_true, y_pred), 3)
    print(f"Error máximo: {me}")
    print(f"Coeficiente de determinación (R2): {r2}")
    print(f"Varianza explicada: {evs}")
    print(f"Error cuadrático medio: {mse}")
    print(f"Error absoluto medio: {mae}")
    print(f"Error Porcentual Absoluto Medio: {mape}")
    return me, r2, evs, mae, mse, mape


def datos_supervisados(entrenamiento_esc, test_esc, pasado_historico):
    x_entrenamiento, y_entrenamiento = [], []
    for j in range(0, entrenamiento_esc.shape[0] - len(test_esc) + 1):
        indices = range(j - pasado_historico, j, 1)
        x_entrenamiento.append(
            np.reshape(entrenamiento_esc[indices], (pasado_historico, 1)))
        y_entrenamiento.append(entrenamiento_esc[j:j + len(test_esc)])
    x_train = np.asarray(x_entrenamiento)
    y_train = np.asarray(y_entrenamiento)
    x_test = np.reshape(entrenamiento_esc[-pasado_historico:],
                        (1, pasado_historico, 1))
    y_test = np.reshape(test_esc, (1, test_esc.shape[0]))
    return x_train, y_train, x_test, y_test


def plot_history(history, nombre, history_plot_path):
    plt.figure()
    plt.xlabel('Épocas')
    plt.ylabel('Error absoluto medio')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Entrenamiento', 'Validación'])
    plt.title(nombre, size=24)
    plt.savefig(history_plot_path)
    plt.show()


def mk_regexs(mercado):
    regex_AUTOARIMA = re.compile(r'^' + mercado + '|^AUTOARIMA.+' +
                                 mercado.replace("Precio_mercado_SPOT_", "") +
                                 "$")
    regex_AR = re.compile(r'^' + mercado + '|^AR_p.+' +
                          mercado.replace("Precio_mercado_SPOT_", "") + "$")
    regex_MA = re.compile(r'^' + mercado + '|^MA_q.+' +
                          mercado.replace("Precio_mercado_SPOT_", "") + "$")
    regex_ARMA = re.compile(r'^' + mercado + '|^ARMA.+' +
                            mercado.replace("Precio_mercado_SPOT_", "") + "$")
    regex_ARIMA = re.compile(r'^' + mercado + '|^ARIMA.+' +
                             mercado.replace("Precio_mercado_SPOT_", "") + "$")
    regex_SARIMA = re.compile(r'^' + mercado + '|^SARIMA.+' +
                              mercado.replace("Precio_mercado_SPOT_", "") +
                              "$")
    regex_BATS = re.compile(r'^' + mercado + '|^BATS.+' +
                            mercado.replace("Precio_mercado_SPOT_", "") + "$")
    regex_TBATS = re.compile(r'^' + mercado + '|^TBATS.+' +
                             mercado.replace("Precio_mercado_SPOT_", "") + "$")
    regex_lstm_densa = re.compile(r'^' + mercado + '|^LSTM\d+?_Densa\d+?.+' +
                                  mercado.replace('Precio_mercado_SPOT_', '') +
                                  '$')
    regex_lstm_densa_do = re.compile(
        r'^' + mercado + '|^LSTM\d+?_Densa\d+?_DO\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_lstm_lstm_densa = re.compile(
        r'^' + mercado + '|^LSTM\d+?_LSTM\d+?_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_lstm_lstm_densa_do = re.compile(
        r'^' + mercado + '|^LSTM\d+?_LSTM\d+?_Densa\d+?_DO\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_lstm_lstm_densa_densa = re.compile(
        r'^' + mercado + '|^LSTM\d+?_LSTM\d+?_Densa\d+?_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_lstm_lstm_densa_densa_do = re.compile(
        r'^' + mercado + '|^LSTM\d+?_LSTM\d+?_Densa\d+?_Densa\d+?_DO\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_lstm_lstm_lstm_densa = re.compile(
        r'^' + mercado + '|^LSTM\d+?_LSTM\d+?_LSTM\d+?_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_lstm_lstm_lstm_densa_do = re.compile(
        r'^' + mercado + '|^LSTM\d+?_LSTM\d+?_LSTM\d+?_Densa\d+?_DO\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_gru_flatten_densa = re.compile(
        r'^' + mercado + '|^GRU\d+?_Flatten_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_gru_do_flatten_densa = re.compile(
        r'^' + mercado + '|^GRU\d+?_DO\d+?_Flatten_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_gru_gru_flatten_densa = re.compile(
        r'^' + mercado + '|^GRU\d+?_GRU\d+?_Flatten_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_gru_do_gru_do_flatten_densa = re.compile(
        r'^' + mercado +
        '|^GRU\d+?_DO\d+?_GRU\d+?_DO\d+?_Flatten_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_gru_gru_flatten_densa_densa = re.compile(
        r'^' + mercado + '|^GRU\d+?_GRU\d+?_Flatten_Densa\d+?_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_gru_do_gru_do_flatten_densa_densa = re.compile(
        r'^' + mercado +
        '|^GRU\d+?_DO\d+?_GRU\d+?_DO\d+?_Flatten_Densa\d+?_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_gru_gru_gru_flatten_densa = re.compile(
        r'^' + mercado + '|^GRU\d+?_GRU\d+?_GRU\d+?_Flatten_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_gru_do_gru_do_gru_do_flatten_densa = re.compile(
        r'^' + mercado +
        '|^GRU\d+?_DO\d+?_GRU\d+?_DO\d+?_GRU\d+?_DO\d+?_Flatten_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_conv1d_maxpool1d_flatten_densa = re.compile(
        r'^' + mercado + '|^Conv1D\d+?_MaxPool1D\d+?_Flatten_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_conv1d_do_maxpool1d_flatten_densa = re.compile(
        r'^' + mercado +
        '|^Conv1D\d+?_DO\d+?_MaxPool1D\d+?_Flatten_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_conv1d_maxpool1d_conv1d_maxpool1d_flatten_densa = re.compile(
        r'^' + mercado +
        '|^Conv1D\d+?_MaxPool1D\d+?_Conv1D\d+?_MaxPool1D\d+?_Flatten_Densa\d+?.+'
        + mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_conv1d_do_maxpool1d_conv1d_do_maxpool1d_flatten_densa = re.compile(
        r'^' + mercado +
        '|^Conv1D\d+?_DO\d+?_MaxPool1D\d+?_Conv1D\d+?_DO\d+?_MaxPool1D\d+?_Flatten_Densa\d+?.+'
        + mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_TCN_densa = re.compile(r'^' + mercado + '|^TCN\d+?_Densa\d+?.+' +
                                 mercado.replace('Precio_mercado_SPOT_', '') +
                                 '$')
    regex_TCN_do_densa = re.compile(
        r'^' + mercado + '|^TCN\d+?_DO\d+?_Densa\d+?.+' +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_lstm_densa = re.compile(
        r"^" + mercado + "|^Masking_LSTM\d+?_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_lstm_densa_do = re.compile(
        r"^" + mercado + "|^Masking_LSTM\d+?_Densa\d+?_DO\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_lstm_lstm_densa = re.compile(
        r"^" + mercado + "|^Masking_LSTM\d+?_LSTM\d+?_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_lstm_lstm_densa_do = re.compile(
        r"^" + mercado + "|^Masking_LSTM\d+?_LSTM\d+?_Densa\d+?_DO\d+?" +
        mercado.replace("Perecio_mercado_SPOT_", '') + "$")
    regex_masking_lstm_lstm_densa_densa = re.compile(
        r"^" + mercado + "|^Masking_LSTM\d+?_LSTM\d+?_Densa\d+?_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_lstm_lstm_densa_densa_do = re.compile(
        r"^" + mercado +
        "|^Masking_LSTM\d+?_LSTM\d+?_Densa\d+?_Densa\d+?_DO\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_lstm_lstm_lstm_densa = re.compile(
        r"^" + mercado + "|^Masking_LSTM\d+?_LSTM\d+?_LSTM\d+?_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_lstm_lstm_lstm_densa_do = re.compile(
        r"^" + mercado +
        "|^Masking_LSTM\d+?_LSTM\d+?_LSTM\d+?_Densa\d+?_DO\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_gru_flatten_densa = re.compile(
        r"^" + mercado + "|^Masking_GRU\d+?_Flatten_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_gru_do_flatten_densa = re.compile(
        r"^" + mercado + "|^Masking_GRU\d+?_DO\d+?_Flatten_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_gru_gru_flatten_densa = re.compile(
        r"^" + mercado + "|^Masking_GRU\d+?_GRU\d+?_Flatten_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_gru_do_gru_do_flatten_densa = re.compile(
        r"^" + mercado +
        "|^Masking_GRU\d+?_DO\d+?_GRU\d+?_DO\d+?_Flatten_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_gru_gru_flatten_densa_densa = re.compile(
        r"^" + mercado +
        "|^Masking_GRU\d+?_GRU\d+?_Flatten_Densa\d+?_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_gru_do_gru_do_flatten_densa_densa = re.compile(
        r"^" + mercado +
        "|^Masking_GRU\d+?_DO\d+?_GRU\d+?_DO\d+?_Flatten_Densa\d+?_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_gru_gru_gru_flatten_densa = re.compile(
        r"^" + mercado +
        "|^Masking_GRU\d+?_GRU\d+?_GRU\d+?_Flatten_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_gru_do_gru_do_gru_do_flatten_densa = re.compile(
        r"^" + mercado +
        "|^Masking_GRU\d+?_DO\d+?_GRU\d+?_DO\d+?_GRU\d+?_DO\d+?_Flatten_Densa\d+?"
        + mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_conv1d_maxpool1d_flatten_densa = re.compile(
        r"^" + mercado +
        "|^Masking_Conv1D\d+?_MaxPool1D\d+?_Flatten_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_conv1d_do_maxpool1d_flatten_densa = re.compile(
        r"^" + mercado +
        "|^Masking_Conv1D\d+?_DO\d+?_MaxPool1D\d+?_Flatten_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_conv1d_maxpool1d_conv1d_maxpool1d_flatten_densa = re.compile(
        r"^" + mercado +
        "|^Masking_Conv1D\d+?_MaxPool1D\d+?_Conv1D\d+?_MaxPool1D\d+?_Flatten_Densa\d+?"
        + mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_conv1d_do_maxpool1d_conv1d_do_maxpool1d_flatten_densa = re.compile(
        r"^" + mercado +
        "|^Masking_Conv1D\d+?_DO\d+?_MaxPool1D\d+?_Conv1D\d+?_DO\d+?_MaxPool1D\d+?_Flatten_Densa\d+?"
        + mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_TCN_densa = re.compile(
        r"^" + mercado + "|^Masking_TCN\d+?_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    regex_masking_TCN_do_densa = re.compile(
        r"^" + mercado + "|^Masking_TCN\d+?_DO\d+?_Densa\d+?" +
        mercado.replace('Precio_mercado_SPOT_', '') + '$')
    return regex_AUTOARIMA, regex_AR, regex_MA, regex_ARMA, regex_ARIMA, regex_SARIMA, regex_BATS, regex_TBATS, regex_lstm_densa, regex_lstm_densa_do, regex_lstm_lstm_densa, regex_lstm_lstm_densa_do, regex_lstm_lstm_densa_densa, regex_lstm_lstm_densa_densa_do, regex_lstm_lstm_lstm_densa, regex_lstm_lstm_lstm_densa_do, regex_gru_flatten_densa, regex_gru_do_flatten_densa, regex_gru_gru_flatten_densa, regex_gru_do_gru_do_flatten_densa, regex_gru_gru_flatten_densa_densa, regex_gru_do_gru_do_flatten_densa_densa, regex_gru_gru_gru_flatten_densa, regex_gru_do_gru_do_gru_do_flatten_densa, regex_conv1d_maxpool1d_flatten_densa, regex_conv1d_do_maxpool1d_flatten_densa, regex_conv1d_maxpool1d_conv1d_maxpool1d_flatten_densa, regex_conv1d_do_maxpool1d_conv1d_do_maxpool1d_flatten_densa, regex_TCN_densa, regex_TCN_do_densa, regex_masking_lstm_densa, regex_masking_lstm_densa_do, regex_masking_lstm_lstm_densa, regex_masking_lstm_lstm_densa_do, regex_masking_lstm_lstm_densa_densa, regex_masking_lstm_lstm_densa_densa_do, regex_masking_lstm_lstm_lstm_densa, regex_masking_lstm_lstm_lstm_densa_do, regex_masking_gru_flatten_densa, regex_masking_gru_do_flatten_densa, regex_masking_gru_gru_flatten_densa, regex_masking_gru_do_gru_do_flatten_densa, regex_masking_gru_gru_flatten_densa_densa, regex_masking_gru_do_gru_do_flatten_densa_densa, regex_masking_gru_gru_gru_flatten_densa, regex_masking_gru_do_gru_do_gru_do_flatten_densa, regex_masking_conv1d_maxpool1d_flatten_densa, regex_masking_conv1d_do_maxpool1d_flatten_densa, regex_masking_conv1d_maxpool1d_conv1d_maxpool1d_flatten_densa, regex_masking_conv1d_do_maxpool1d_conv1d_do_maxpool1d_flatten_densa, regex_masking_TCN_densa, regex_masking_TCN_do_densa


"""### Funciones: Redes Neuronales"""


# 08- LSTM Densa
# crear_lstm_densa
def crear_lstm_densa(unidades,
                     pasado_historico,
                     horizonte_prevision,
                     n_artributos=1):
    """crear_lstm_densa"""
    model = Sequential(
        name="LSTM{}_Densa{}".format(unidades, horizonte_prevision))
    model.add(
        LSTM(units=unidades, input_shape=(pasado_historico, n_artributos)))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_lstm_densa
def crear_masking_lstm_densa(unidades,
                             pasado_historico,
                             horizonte_prevision,
                             n_artributos=1):
    """crear_masking_lstm_densa"""
    model = Sequential(
        name="Masking_LSTM{}_Densa{}".format(unidades, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 09- LSTM Densa DO
# crear_lstm_densa_do
def crear_lstm_densa_do(unidades,
                        pasado_historico,
                        horizonte_prevision,
                        dropout=0.1,
                        n_artributos=1):
    """crear_lstm_densa_do"""
    model = Sequential(name="LSTM{}_Densa{}_DO{}".format(
        unidades, horizonte_prevision, int(dropout * 100)))
    model.add(
        LSTM(units=unidades,
             dropout=dropout,
             input_shape=(pasado_historico, n_artributos)))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_lstm_densa_do
def crear_masking_lstm_densa_do(unidades,
                                pasado_historico,
                                horizonte_prevision,
                                dropout=0.1,
                                n_artributos=1):
    """crear_masking_lstm_densa_do"""
    model = Sequential(name="Masking_LSTM{}_Densa{}_DO{}".format(
        unidades, horizonte_prevision, int(dropout * 100)))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades, dropout=dropout))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 10- LSTM LSTM Densa
# crear_lstm_lstm_densa
def crear_lstm_lstm_densa(unidades_1,
                          unidades_2,
                          pasado_historico,
                          horizonte_prevision,
                          n_artributos=1):
    """crear_lstm_lstm_densa"""
    model = Sequential(name="LSTM{}_LSTM{}_Densa{}".format(
        unidades_1, unidades_2, horizonte_prevision))
    model.add(
        LSTM(units=unidades_1,
             return_sequences=True,
             input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_2))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_lstm_lstm_densa
def crear_masking_lstm_lstm_densa(unidades_1,
                                  unidades_2,
                                  pasado_historico,
                                  horizonte_prevision,
                                  n_artributos=1):
    model = Sequential(name="Masking_LSTM{}_LSTM{}_Densa{}".format(
        unidades_1, unidades_2, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_1, return_sequences=True))
    model.add(LSTM(units=unidades_2))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 11- LSTM LSTM Densa_DO
# crear_lstm_lstm_densa_do
def crear_lstm_lstm_densa_do(unidades_1,
                             unidades_2,
                             pasado_historico,
                             horizonte_prevision,
                             dropout=0.1,
                             n_artributos=1):
    model = Sequential(name="LSTM{}_LSTM{}_Densa{}_DO{}".format(
        unidades_1, unidades_2, horizonte_prevision, int(dropout * 100)))
    model.add(
        LSTM(units=unidades_1,
             dropout=dropout,
             return_sequences=True,
             input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_2, dropout=dropout))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_lstm_lstm_densa_do
def crear_masking_lstm_lstm_densa_do(unidades_1,
                                     unidades_2,
                                     pasado_historico,
                                     horizonte_prevision,
                                     dropout=0.1,
                                     n_artributos=1):
    model = Sequential(name="Masking_LSTM{}_LSTM{}_Densa{}_DO{}".format(
        unidades_1, unidades_2, horizonte_prevision, int(dropout * 100)))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_1, dropout=dropout, return_sequences=True))
    model.add(LSTM(units=unidades_2, dropout=dropout))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 12- LSTM LSTM Densa Densa
# crear_lstm_lstm_densa_densa
def crear_lstm_lstm_densa_densa(unidades_1,
                                unidades_2,
                                pasado_historico,
                                horizonte_prevision,
                                n_artributos=1):
    model = Sequential(name="LSTM{}_LSTM{}_Densa{}_Densa{}".format(
        unidades_1, unidades_2, unidades_1, horizonte_prevision))
    model.add(
        LSTM(units=unidades_1,
             return_sequences=True,
             input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_2))
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_lstm_lstm_densa_densa
def crear_masking_lstm_lstm_densa_densa(unidades_1,
                                        unidades_2,
                                        pasado_historico,
                                        horizonte_prevision,
                                        n_artributos=1):
    model = Sequential(name="Masking_LSTM{}_LSTM{}_Densa{}_Densa{}".format(
        unidades_1, unidades_2, unidades_1, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_1, return_sequences=True))
    model.add(LSTM(units=unidades_2))
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 13- LSTM LSTM Densa Densa_DO
# crear_lstm_lstm_densa_densa_do
def crear_lstm_lstm_densa_densa_do(unidades_1,
                                   unidades_2,
                                   pasado_historico,
                                   horizonte_prevision,
                                   dropout=0.1,
                                   n_artributos=1):
    model = Sequential(name="LSTM{}_LSTM{}_Densa{}_Densa{}_DO{}".format(
        unidades_1, unidades_2, unidades_1, horizonte_prevision,
        int(dropout * 100)))
    model.add(
        LSTM(units=unidades_1,
             dropout=dropout,
             return_sequences=True,
             input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_2, dropout=dropout))
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_lstm_lstm_densa_densa_do
def crear_masking_lstm_lstm_densa_densa_do(unidades_1,
                                           unidades_2,
                                           pasado_historico,
                                           horizonte_prevision,
                                           dropout=0.1,
                                           n_artributos=1):
    model = Sequential(
        name="Masking_LSTM{}_LSTM{}_Densa{}_Densa{}_DO{}".format(
            unidades_1, unidades_2, unidades_1, horizonte_prevision,
            int(dropout * 100)))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_1, dropout=dropout, return_sequences=True))
    model.add(LSTM(units=unidades_2, dropout=dropout))
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 14- LSTM LSTM LSTM_Densa
# crear_lstm_lstm_lstm_densa
def crear_lstm_lstm_lstm_densa(unidades_1,
                               unidades_2,
                               unidades_3,
                               pasado_historico,
                               horizonte_prevision,
                               n_artributos=1):
    model = Sequential(name="LSTM{}_LSTM{}_LSTM{}_Densa{}".format(
        unidades_1, unidades_2, unidades_3, horizonte_prevision))
    model.add(
        LSTM(units=unidades_1,
             return_sequences=True,
             input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_2, return_sequences=True))
    model.add(LSTM(units=unidades_3))
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_lstm_lstm_lstm_densa
def crear_masking_lstm_lstm_lstm_densa(unidades_1,
                                       unidades_2,
                                       unidades_3,
                                       pasado_historico,
                                       horizonte_prevision,
                                       n_artributos=1):
    model = Sequential(name="Masking_LSTM{}_LSTM{}_LSTM{}_Densa{}".format(
        unidades_1, unidades_2, unidades_3, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_1, return_sequences=True))
    model.add(LSTM(units=unidades_2, return_sequences=True))
    model.add(LSTM(units=unidades_3))
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 15- LSTM LSTM LSTM_Densa_DO
# crear_lstm_lstm_lstm_densa_do
def crear_lstm_lstm_lstm_densa_do(unidades_1,
                                  unidades_2,
                                  unidades_3,
                                  pasado_historico,
                                  horizonte_prevision,
                                  dropout=0.1,
                                  n_artributos=1):
    model = Sequential(name="LSTM{}_LSTM{}_LSTM{}_Densa{}_DO{}".format(
        unidades_1, unidades_2, unidades_3, horizonte_prevision,
        int(dropout * 100)))
    model.add(
        LSTM(units=unidades_1,
             dropout=dropout,
             return_sequences=True,
             input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_2, dropout=dropout, return_sequences=True))
    model.add(LSTM(units=unidades_3, dropout=dropout))
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_lstm_lstm_lstm_densa_do
def crear_masking_lstm_lstm_lstm_densa_do(unidades_1,
                                          unidades_2,
                                          unidades_3,
                                          pasado_historico,
                                          horizonte_prevision,
                                          dropout=0.1,
                                          n_artributos=1):
    model = Sequential(name="Masking_LSTM{}_LSTM{}_LSTM{}_Densa{}_DO{}".format(
        unidades_1, unidades_2, unidades_3, horizonte_prevision,
        int(dropout * 100)))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(LSTM(units=unidades_1, dropout=dropout, return_sequences=True))
    model.add(LSTM(units=unidades_2, dropout=dropout, return_sequences=True))
    model.add(LSTM(units=unidades_3, dropout=dropout))
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 16- GRU Flatten Densa
# crear_gru_flatten_densa
def crear_gru_flatten_densa(unidades,
                            pasado_historico,
                            horizonte_prevision,
                            n_artributos=1):
    model = Sequential(
        name="GRU{}_Flatten_Densa{}".format(unidades, horizonte_prevision))
    model.add(GRU(units=unidades,
                  input_shape=(pasado_historico, n_artributos)))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_gru_flatten_densa
def crear_masking_gru_flatten_densa(unidades,
                                    pasado_historico,
                                    horizonte_prevision,
                                    n_artributos=1):
    model = Sequential(name="Masking_GRU{}_Flatten_Densa{}".format(
        unidades, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(GRU(units=unidades))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 17- GRU DO Flatten Densa
# crear_gru_do_flatten_densa
def crear_gru_do_flatten_densa(unidades,
                               pasado_historico,
                               horizonte_prevision,
                               dropout=0.1,
                               n_artributos=1):
    model = Sequential(name="GRU{}_DO{}_Flatten_Densa{}".format(
        unidades, int(dropout * 100), horizonte_prevision))
    model.add(GRU(units=unidades,
                  input_shape=(pasado_historico, n_artributos)))
    model.add(Dropout(rate=dropout, seed=1))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_gru_do_flatten_densa
def crear_masking_gru_do_flatten_densa(unidades,
                                       pasado_historico,
                                       horizonte_prevision,
                                       dropout=0.1,
                                       n_artributos=1):
    model = Sequential(name="Masking_GRU{}_DO{}_Flatten_Densa{}".format(
        unidades, horizonte_prevision, int(dropout * 100)))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(GRU(units=unidades, dropout=dropout))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 18- GRU GRU Flatten Densa
# crear_gru_gru_flatten_densa
def crear_gru_gru_flatten_densa(unidades_1,
                                unidades_2,
                                pasado_historico,
                                horizonte_prevision,
                                n_artributos=1):
    model = Sequential(name="GRU{}_GRU{}_Flatten_Densa{}".format(
        unidades_1, unidades_2, horizonte_prevision))
    model.add(
        GRU(units=unidades_1,
            return_sequences=True,
            input_shape=(pasado_historico, n_artributos)))
    model.add(GRU(units=unidades_2))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_gru_gru_flatten_densa
def crear_masking_gru_gru_flatten_densa(unidades_1,
                                        unidades_2,
                                        pasado_historico,
                                        horizonte_prevision,
                                        n_artributos=1):
    model = Sequential(name="Masking_GRU{}_GRU{}_Flatten_Densa{}".format(
        unidades_1, unidades_2, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(GRU(units=unidades_1, return_sequences=True))
    model.add(GRU(units=unidades_2))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 19- GRU DO GRU DO Flatten Densa
# crear_gru_do_gru_do_flatten_densa
def crear_gru_do_gru_do_flatten_densa(unidades_1,
                                      unidades_2,
                                      pasado_historico,
                                      horizonte_prevision,
                                      dropout=0.1,
                                      n_artributos=1):
    model = Sequential(name="GRU{}_DO{}_GRU{}_DO{}_Flatten_Densa{}".format(
        unidades_1, int(dropout * 100),
        unidades_2, int(dropout * 100),
        horizonte_prevision))
    model.add(GRU(units=unidades_1,
                  return_sequences=True,
                  input_shape=(pasado_historico, n_artributos)))
    model.add(Dropout(rate=dropout, seed=1))
    model.add(GRU(units=unidades_2))
    model.add(Dropout(rate=dropout, seed=1))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_gru_gru_flatten_densa_do
def crear_masking_gru_do_gru_do_flatten_densa(unidades_1,
                                              unidades_2,
                                              pasado_historico,
                                              horizonte_prevision,
                                              dropout=0.1,
                                              n_artributos=1):
    model = Sequential(name="Masking_GRU{}_DO{}_GRU{}_DO{}_Flatten_Densa{}".format(
        unidades_1, int(dropout * 100),
        unidades_2, int(dropout * 100),
        horizonte_prevision))
    model.add(Masking(mask_value=-1,
                      input_shape=(pasado_historico, n_artributos)))
    model.add(GRU(units=unidades_1, dropout=dropout, return_sequences=True))
    model.add(GRU(units=unidades_2, dropout=dropout))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 20- GRU GRU Flatten Densa_Densa
# crear_gru_gru_flatten_densa_densa
def crear_gru_gru_flatten_densa_densa(unidades_1,
                                      unidades_2,
                                      pasado_historico,
                                      horizonte_prevision,
                                      n_artributos=1):
    model = Sequential(name="GRU{}_GRU{}_Flatten_Densa{}_Densa{}".format(
        unidades_1, unidades_2, unidades_1, horizonte_prevision))
    model.add(
        GRU(units=unidades_1,
            return_sequences=True,
            input_shape=(pasado_historico, n_artributos)))
    model.add(GRU(units=unidades_2))
    model.add(Flatten())
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_gru_gru_flatten_densa_densa
def crear_masking_gru_gru_flatten_densa_densa(unidades_1,
                                              unidades_2,
                                              pasado_historico,
                                              horizonte_prevision,
                                              n_artributos=1):
    model = Sequential(
        name="Masking_GRU{}_GRU{}_Flatten_Densa{}_Densa{}".format(
            unidades_1, unidades_2, unidades_1, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(GRU(units=unidades_1, return_sequences=True))
    model.add(GRU(units=unidades_2))
    model.add(Flatten())
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 21- GRU GRU Flatten Densa_Densa_DO
# crear_gru_do_gru_do_flatten_densa_densa
def crear_gru_do_gru_do_flatten_densa_densa(unidades_1,
                                            unidades_2,
                                            pasado_historico,
                                            horizonte_prevision,
                                            dropout=0.1,
                                            n_artributos=1):
    model = Sequential(
        name="GRU{}_DO{}_GRU{}_DO{}_Flatten_Densa{}_Densa{}".format(
            unidades_1, int(dropout * 100), unidades_2, int(dropout * 100),
            unidades_1, horizonte_prevision))
    model.add(
        GRU(units=unidades_1,
            return_sequences=True,
            input_shape=(pasado_historico, n_artributos)))
    model.add(Dropout(rate=dropout, seed=1))
    model.add(GRU(units=unidades_2))
    model.add(Dropout(rate=dropout, seed=1))
    model.add(Flatten())
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_gru_gru_flatten_densa_densa_do
def crear_masking_gru_do_gru_do_flatten_densa_densa(unidades_1,
                                                    unidades_2,
                                                    pasado_historico,
                                                    horizonte_prevision,
                                                    dropout=0.1,
                                                    n_artributos=1):
    model = Sequential(
        name="Masking_GRU{}_DO{}_GRU{}_DO{}_Flatten_Densa{}_Densa{}".format(
            unidades_1, int(dropout * 100), unidades_2, int(dropout * 100),
            unidades_1, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(GRU(units=unidades_1, dropout=dropout, return_sequences=True))
    model.add(GRU(units=unidades_2, dropout=dropout))
    model.add(Flatten())
    model.add(Dense(units=unidades_1))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 22- GRU GRU GRU Flatten Densa
# crear_gru_gru_gru_flatten_densa
def crear_gru_gru_gru_flatten_densa(unidades_1,
                                    unidades_2,
                                    unidades_3,
                                    pasado_historico,
                                    horizonte_prevision,
                                    n_artributos=1):
    model = Sequential(name="GRU{}_GRU{}_GRU{}_Flatten_Densa{}".format(
        unidades_1, unidades_2, unidades_3, horizonte_prevision))
    model.add(
        GRU(units=unidades_1,
            return_sequences=True,
            input_shape=(pasado_historico, n_artributos)))
    model.add(GRU(units=unidades_2, return_sequences=True))
    model.add(GRU(units=unidades_3))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_gru_gru_gru_flatten_densa
def crear_masking_gru_gru_gru_flatten_densa(unidades_1,
                                            unidades_2,
                                            unidades_3,
                                            pasado_historico,
                                            horizonte_prevision,
                                            n_artributos=1):
    model = Sequential(name="Masking_GRU{}_GRU{}_GRU{}_Flatten_Densa{}".format(
        unidades_1, unidades_2, unidades_3, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(GRU(units=unidades_1, return_sequences=True))
    model.add(GRU(units=unidades_2, return_sequences=True))
    model.add(GRU(units=unidades_3))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 23- GRU GRU GRU Flatten Densa_DO
# crear_gru_do_gru_do_gru_do_flatten_densa
def crear_gru_do_gru_do_gru_do_flatten_densa(unidades_1,
                                             unidades_2,
                                             unidades_3,
                                             pasado_historico,
                                             horizonte_prevision,
                                             dropout=0.1,
                                             n_artributos=1):
    model = Sequential(
        name="GRU{}_DO{}_GRU{}_DO{}_GRU{}_DO{}_Flatten_Densa{}".format(
            unidades_1, int(dropout * 100), unidades_2, int(dropout * 100),
            unidades_3, int(dropout * 100), horizonte_prevision))
    model.add(
        GRU(units=unidades_1,
            return_sequences=True,
            input_shape=(pasado_historico, n_artributos)))
    model.add(Dropout(rate=dropout, seed=1))
    model.add(GRU(units=unidades_2, return_sequences=True))
    model.add(Dropout(rate=dropout, seed=1))
    model.add(GRU(units=unidades_3))
    model.add(Dropout(rate=dropout, seed=1))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_gru_do_gru_do_gru_do_flatten_densa
def crear_masking_gru_do_gru_do_gru_do_flatten_densa(unidades_1,
                                                     unidades_2,
                                                     unidades_3,
                                                     pasado_historico,
                                                     horizonte_prevision,
                                                     dropout=0.1,
                                                     n_artributos=1):
    model = Sequential(
        name="Masking_GRU{}_DO{}_GRU{}_DO{}_GRU{}_DO{}_Flatten_Densa{}".format(
            unidades_1, int(dropout * 100), unidades_2, int(dropout * 100),
            unidades_3, int(dropout * 100), horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(GRU(units=unidades_1, dropout=dropout, return_sequences=True))
    model.add(GRU(units=unidades_2, dropout=dropout, return_sequences=True))
    model.add(GRU(units=unidades_3, dropout=dropout))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 24- Conv1D MaxPool1D Flatten Densa
# crear_conv1d_maxpool1d_flatten_densa
def crear_conv1d_maxpool1d_flatten_densa(unidades_1,
                                         pasado_historico,
                                         horizonte_prevision,
                                         n_artributos=1):
    model = Sequential(name="Conv1D{}_MaxPool1D{}_Flatten_Densa{}".format(
        unidades_1, 2, horizonte_prevision))
    model.add(
        Conv1D(filters=unidades_1,
               kernel_size=7,
               padding='same',
               activation='relu',
               input_shape=(pasado_historico, n_artributos)))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_conv1d_maxpool1d_flatten_densa
def crear_masking_conv1d_maxpool1d_flatten_densa(unidades_1,
                                                 pasado_historico,
                                                 horizonte_prevision,
                                                 n_artributos=1):
    model = Sequential(
        name="Masking_Conv1D{}_MaxPool1D{}_Flatten_Densa{}".format(
            unidades_1, 2, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(
        Conv1D(filters=unidades_1,
               kernel_size=7,
               padding='same',
               activation='relu',
               input_shape=(pasado_historico, n_artributos)))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


#### 25- Conv1D_DO_MaxPool1D_Flatten_Densa
# crear_conv1d_DO_maxpool1d_flatten_densa
def crear_conv1d_do_maxpool1d_flatten_densa(unidades_1,
                                            pasado_historico,
                                            horizonte_prevision,
                                            dropout=0.1,
                                            n_artributos=1):
    model = Sequential(name="Conv1D{}_DO{}_MaxPool1D{}_Flatten_Densa{}".format(
        unidades_1, int(dropout * 100), 2, horizonte_prevision))
    model.add(
        Conv1D(filters=unidades_1,
               kernel_size=7,
               padding='same',
               activation='relu',
               input_shape=(pasado_historico, n_artributos)))
    model.add(Dropout(rate=dropout, seed=1))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_conv1d_DO_maxpool1d_flatten_densa
def crear_masking_conv1d_do_maxpool1d_flatten_densa(unidades_1,
                                                    pasado_historico,
                                                    horizonte_prevision,
                                                    dropout=0.1,
                                                    n_artributos=1):
    model = Sequential(
        name="Masking_Conv1D{}_DO{}_MaxPool1D{}_Flatten_Densa{}".format(
            unidades_1, int(dropout * 100), 2, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(
        Conv1D(filters=unidades_1,
               kernel_size=7,
               padding='same',
               activation='relu',
               input_shape=(pasado_historico, n_artributos)))
    model.add(Dropout(rate=dropout, seed=1))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 26- Conv1D MaxPool1D Conv1D MaxPool1D Flatten Densa
# crear_conv1d_maxpool1d_conv1d_maxpool1d_flatten_densa
def crear_conv1d_maxpool1d_conv1d_maxpool1d_flatten_densa(
        unidades_1,
        unidades_2,
        pasado_historico,
        horizonte_prevision,
        n_artributos=1):
    model = Sequential(
        name="Conv1D{}_MaxPool1D{}_Conv1D{}_MaxPool1D{}_Flatten_Densa{}".
            format(unidades_1, 2, unidades_2, 2, horizonte_prevision))
    model.add(
        Conv1D(filters=unidades_1,
               kernel_size=7,
               padding='same',
               activation='relu',
               input_shape=(pasado_historico, n_artributos)))
    model.add(MaxPool1D(pool_size=2))
    model.add(
        Conv1D(filters=unidades_2,
               kernel_size=5,
               padding='same',
               activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_conv1d_maxpool1d_conv1d_maxpool1d_flatten_densa
def crear_masking_conv1d_maxpool1d_conv1d_maxpool1d_flatten_densa(
        unidades_1,
        unidades_2,
        pasado_historico,
        horizonte_prevision,
        n_artributos=1):
    model = Sequential(
        name="Masking_Conv1D{}_MaxPool1D{}_Conv1D{}_MaxPool1D{}_Flatten_Densa{}"
            .format(unidades_1, 2, unidades_2, 2, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(
        Conv1D(filters=unidades_1,
               kernel_size=7,
               padding='same',
               activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(
        Conv1D(filters=unidades_2,
               kernel_size=5,
               padding='same',
               activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 27- Conv1D DO MaxPool1D Conv1D DO MaxPool1D Flatten Densa
# crear_conv1d_do_maxpool1d_conv1d_do_maxpool1d_flatten_densa
def crear_conv1d_do_maxpool1d_conv1d_do_maxpool1d_flatten_densa(
        unidades_1,
        unidades_2,
        pasado_historico,
        horizonte_prevision,
        dropout=0.1,
        n_artributos=1):
    model = Sequential(
        name=
        "Conv1D{}_DO{}_MaxPool1D{}_Conv1D{}_DO{}_MaxPool1D{}_Flatten_Densa{}".
            format(unidades_1, int(dropout * 100), 2, unidades_2, int(
            dropout * 100), 2, horizonte_prevision))
    model.add(
        Conv1D(filters=unidades_1,
               kernel_size=7,
               padding='same',
               activation='relu',
               input_shape=(pasado_historico, n_artributos)))
    model.add(Dropout(rate=dropout))
    model.add(MaxPool1D(pool_size=2))
    model.add(
        Conv1D(filters=unidades_2,
               kernel_size=5,
               padding='same',
               activation='relu'))
    model.add(Dropout(rate=dropout))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_conv1d_do_maxpool1d_conv1d_do_maxpool1d_flatten_densa
def crear_masking_conv1d_do_maxpool1d_conv1d_do_maxpool1d_flatten_densa(
        unidades_1,
        unidades_2,
        pasado_historico,
        horizonte_prevision,
        dropout=0.1,
        n_artributos=1):
    model = Sequential(
        name=
        "Masking_Conv1D{}_DO{}_MaxPool1D{}_Conv1D{}_DO{}_MaxPool1D{}_Flatten_Densa{}"
            .format(unidades_1, int(dropout * 100), 2, unidades_2,
                    int(dropout * 100), 2, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(
        Conv1D(filters=unidades_1,
               kernel_size=7,
               padding='same',
               activation='relu'))
    model.add(Dropout(rate=dropout))
    model.add(MaxPool1D(pool_size=2))
    model.add(
        Conv1D(filters=unidades_2,
               kernel_size=5,
               padding='same',
               activation='relu'))
    model.add(Dropout(rate=dropout))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


#### 28- TCN_Densa
# crear_TCN_densa
def crear_TCN_densa(unidades,
                    pasado_historico,
                    horizonte_prevision,
                    n_artributos=1):
    model = Sequential(
        name="TCN{}_Densa{}".format(unidades, horizonte_prevision))
    model.add(InputLayer(input_shape=(pasado_historico, n_artributos)))
    model.add(
        TCN(nb_filters=unidades,
            kernel_size=3,
            nb_stacks=1,
            dilations=[1, 2, 4, 8, 16, 32, 64]))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_TCN_densa
def crear_masking_TCN_densa(unidades,
                            pasado_historico,
                            horizonte_prevision,
                            n_artributos=1):
    model = Sequential(
        name="Masking_TCN{}_Densa{}".format(unidades, horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(
        TCN(nb_filters=unidades,
            kernel_size=3,
            nb_stacks=1,
            dilations=[1, 2, 4, 8, 16, 32, 64]))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# 29- TCN DO Densa
# crear_TCN_do_densa
def crear_TCN_do_densa(unidades,
                       pasado_historico,
                       horizonte_prevision,
                       dropout=0.1,
                       n_artributos=1):
    model = Sequential(name="TCN{}_DO{}_Densa{}".format(
        unidades, int(dropout * 100), horizonte_prevision))
    model.add(InputLayer(input_shape=(pasado_historico, n_artributos)))
    model.add(
        TCN(nb_filters=unidades,
            kernel_size=3,
            nb_stacks=1,
            dilations=[1, 2, 4, 8, 16, 32, 64]))
    model.add(Dropout(rate=dropout))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model


# crear_masking_TCN_do_densa
def crear_masking_TCN_do_densa(unidades,
                               pasado_historico,
                               horizonte_prevision,
                               dropout=0.1,
                               n_artributos=1):
    model = Sequential(name="Masking_TCN{}_DO{}_Densa{}".format(
        unidades, int(dropout * 100), horizonte_prevision))
    model.add(
        Masking(mask_value=-1, input_shape=(pasado_historico, n_artributos)))
    model.add(
        TCN(nb_filters=unidades,
            kernel_size=3,
            nb_stacks=1,
            dilations=[1, 2, 4, 8, 16, 32, 64]))
    model.add(Dropout(rate=dropout))
    model.add(Dense(units=horizonte_prevision))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    print(model.summary())
    return model

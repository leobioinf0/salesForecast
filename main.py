from salesForecast import *
import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import pickle
import gc
import datetime
import pprint

plt.style.use('fast')
plt.rcParams["figure.figsize"] = (10, 5)
#sns.set_context("paper", rc={"lines.linewidth": 1})
tf.random.set_seed(1)
tf.get_logger().setLevel('ERROR')
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.set_option("display.max_rows", 100)
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)
sns.set(style='whitegrid', palette='muted')



if __name__ == '__main__':

    """# Directorios"""

    # Definir el directorio de trabajo
    cwd = "/home/leo/salesForecast"
    os.chdir(cwd)

    # Crear directorios
    data_path = mk_dir("00-data")
    plots_path = mk_dir("01-plots")
    modelos_path = mk_dir("02-modelos")
    tablas_path = mk_dir("03-tablas")
    sumario_path = mk_dir("04-sumario")

    """# Importar datos"""

    # Importing dataset
    df = pd.read_excel('/home/leo/salesForecast/DEALS_DETAIL-DUMMY_NUMBERS.xlsx')
    df.drop('Negocio - Título', axis=1, inplace=True)
    df.columns = [
        'Creado_el', 'Fecha_ganado', 'Origen', 'CAC', 'Vehículo', 'Contratación',
        'Tipo_cliente', 'Kilometraje', 'Plazo', 'PFF', 'Cuota', 'Comisión_fija',
        'Comision_variable', 'Comsión_total']
    df['Vehículo'] = df['Vehículo'].str.replace('Modelo ', '')
    # convertir las 'Negocio - Negocio creado el' al formato 'datetime' de pandas
    df['Creado_el'] = pd.to_datetime(df['Creado_el'], format='%Y-%m-%d %H:%M:%S')
    # convertir las 'Negocio - Fecha de ganado' al formato 'datetime' de pandas
    df['Fecha_ganado'] = pd.to_datetime(df['Fecha_ganado'],
                                        format='%Y-%m-%d %H:%M:%S')
    # ordenar datos por 'Fecha_creado'
    df.sort_values(by=['Creado_el'], inplace=True, ascending=True)
    df.reset_index(drop=True, inplace=True)


    # """# CANTIDAD DE VENTAS POR FECHA"""
    #
    df_fecha_cant = df.resample("D", on='Creado_el').count()['Comsión_total'].to_frame(name="Num_ventas")
    # title = "CANTIDAD DE VENTAS POR FECHA"
    # plot_line(df_fecha_cant, title, plots_path)
    #
    # """# CANTIDAD DE VENTAS POR DÍA DE LA SEMANA"""
    #
    # df_fecha_cant['Día_de_la_semana'] = df_fecha_cant.index.day_name()
    # df_fecha_cant['Día_de_la_semana'] = df_fecha_cant['Día_de_la_semana'].apply(lambda x: traducir_dia(x))
    # dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    # df_dia_cant = df_fecha_cant.groupby('Día_de_la_semana').sum()['Num_ventas'].to_frame(name="Num_ventas").reindex(
    #     dias)
    # title = "CANTIDAD DE VENTAS POR DÍA DE LA SEMANA"
    # plot_bar(df_dia_cant, title, plots_path)
    # plot_pie_cant(df_dia_cant.index, df_dia_cant["Num_ventas"])
    #
    # title = "PASTEL CANTIDAD DE VENTAS POR DÍA DE LA SEMANA"
    # plot_pie(df_dia_cant, title, plots_path)
    #
    # """# CANTIDAD DE VENTAS POR VEHÍCULO"""
    #
    # df_veh_cant = df.groupby('Vehículo').count()['Comsión_total'].sort_values(ascending=False).to_frame(
    #     name="Num_ventas")
    #
    # title = "CANTIDAD DE VENTAS POR VEHÍCULO"
    # plot_bar(df_veh_cant, title, plots_path)
    # plot_pie(df_veh_cant, title, plots_path)
    #
    # """# CANTIDAD DE VENTAS POR ORIGEN"""
    #
    # df_orig_cant = df.groupby('Origen').count()['Comsión_total'].sort_values(ascending=False).to_frame(
    #     name="Num_ventas")
    #
    # title = "CANTIDAD DE VENTAS POR ORIGEN"
    # plot_bar(df_orig_cant, title, plots_path)
    # plot_pie(df_orig_cant, title, plots_path)
    #
    # """# CANTIDAD DE VENTAS POR PROCESO DE CONTRATACIÓN"""
    # df_contrat_cant = df.groupby('Contratación').count()['Comsión_total'].sort_values(ascending=False).to_frame(
    #     name="Num_ventas")
    # title = "CANTIDAD DE VENTAS POR PROCESO DE CONTRATACIÓN"
    # plot_bar(df_contrat_cant, title, plots_path)
    # plot_pie(df_contrat_cant, title, plots_path)
    #
    # """# CANTIDAD DE VENTAS POR TIPO DE CLIENTE"""
    #
    # df_cliente_cant = df.groupby('Tipo_cliente').count()['Comsión_total'].sort_values(ascending=False).to_frame(
    #     name="Num_ventas")
    # title = "CANTIDAD DE VENTAS POR TIPO DE CLIENTE"
    # plot_bar(df_cliente_cant, title, plots_path)
    # plot_pie(df_cliente_cant, title, plots_path)
    #
    # """# COMISIONES POR FECHA"""
    #
    # df_fecha_com = df.resample("D", on='Creado_el').sum()['Comsión_total'].to_frame(name="Comisiones_dia")
    # title = "COMISIONES POR FECHA"
    # plot_line(df_fecha_com, title, plots_path)
    #
    # """# COMISIONES POR DÍA DE LA SEMANA"""
    #
    # df_fecha_com['Día_de_la_semana'] = df_fecha_com.index.day_name()
    # df_fecha_com['Día_de_la_semana'] = df_fecha_com['Día_de_la_semana'].apply(lambda x: traducir_dia(x))
    # df_dia_com = df_fecha_com.groupby('Día_de_la_semana').sum()['Comisiones_dia'].to_frame(
    #     name="Comisiones_dia").reindex(dias)
    #
    # title = "COMISIONES POR DÍA DE LA SEMANA"
    # plot_bar(df_dia_com, title, plots_path)
    # plot_pie(df_dia_com, title, plots_path)
    #
    # """# COMISIONES TOTALES POR VEHÍCULO"""
    # df_veh_com = df.groupby('Vehículo').sum()['Comsión_total'].sort_values(ascending=False).to_frame(
    #     name="Comisiones_veh")
    # title = "COMISIONES TOTALES POR VEHÍCULO"
    # plot_bar(df_veh_com, title, plots_path)
    # plot_pie(df_veh_com, title, plots_path)
    #
    # """# COMISIONES TOTALES POR ORIGEN"""
    # df_orig_com = df.groupby('Origen').sum()['Comsión_total'].sort_values(ascending=False).to_frame(
    #     name="Comisiones_orig")
    # title = "COMISIONES TOTALES POR ORIGEN"
    # plot_bar(df_orig_com, title, plots_path)
    # plot_pie(df_orig_com, title, plots_path)
    #
    # """# COMISIONES TOTALES POR PROCESO DE CONTRATACIÓN"""
    # df_contrat_com = df.groupby('Contratación').sum()['Comsión_total'].sort_values(ascending=False).to_frame(
    #     name="Comisiones_contrat")
    # title = "COMISIONES TOTALES POR PROCESO DE CONTRATACIÓN"
    # plot_bar(df_contrat_com, title, plots_path)
    # plot_pie(df_contrat_com, title, plots_path)
    #
    # """# COMISIONES TOTALES POR TIPO DE CLIENTE"""
    # df_cliente_com = df.groupby('Tipo_cliente').sum()['Comsión_total'].sort_values(ascending=False).to_frame(
    #     name="Comisiones_cliente")
    # title = "COMISIONES TOTALES POR TIPO DE CLIENTE"
    # plot_bar(df_cliente_com, title, plots_path)
    # plot_pie(df_cliente_com, title, plots_path)
    #
    # """# COMPROBAR ESTACIONARIEDAD 'Num_ventas'"""
    # new_data = df_fecha_cant.loc[(df_fecha_cant.index > "2020-06-01"), ["Num_ventas"]]
    #
    # # Augmented Dicky Fuller Test
    # adf = adfuller(new_data['Num_ventas'])
    # print('ADF = ', str(adf[0]))
    # print('p-value = ', str(adf[1]))
    # print('\nCritical Values:')
    # for key, val in adf[4].items():
    #     print(key, ':', val)
    #     if adf[0] < val:
    #         print('Null Hypothesis Rejected. Time Series is Stationary\n')
    #     else:
    #         print('Null Hypothesis Accepted. Time Series is not Stationary\n')
    #
    # # Seasonal Decompose
    # decomposition = sm.tsa.seasonal_decompose(new_data, model='additive')
    # fig = decomposition.plot()
    # plt.savefig(os.path.join(plots_path, "Num_ventas-seasonal_decompose.pdf"))
    # plt.show()

    """
    # **Predicciones**
    """
    horizonte_prevision = 7
    df = df_fecha_cant.loc[(df_fecha_cant.index > "2020-06-17"), ["Num_ventas"]]
    """
    # MODELOS TRADICIONALES
    """
    """
    ## 00- AUTOARIMA
    """
    dir_modelo = "00-AUTOARIMA"
    modelo_path, tabla_path, plot_path = mk_model_dirs(dir_modelo, modelos_path, tablas_path, plots_path)
    for column in tqdm(["Num_ventas"]):
        # Regex para filtrar los resultados
        RE = re.compile(r'^AUTOARIMA.+' + column + "$")

        # Directorios de modelo y mercado/s
        modelo_mercado_path, tabla_mercado_path, plot_mercado_path = mk_model_mercado_dirs(column, modelo_path, tabla_path, plot_path)

        # datos de entrenamiento y testeo
        df_train = df[[column]].iloc[:-horizonte_prevision].copy()
        df_test = df[[column]].iloc[-horizonte_prevision:].copy()
        tiempo_inicio_fit = datetime.datetime.now()

        # Nombre de modelo
        nombre_modelo = "AUTOARIMA-" + column
        print("\n\n{}: Modelo \t{}".format(tiempo_inicio_fit, nombre_modelo))

        # Ruta al modelo
        path_modelo = os.path.join(modelo_mercado_path, nombre_modelo + ".pkl")
        if os.path.isfile(path_modelo):  # Si el modelo existe
            # Cargar modelo
            print("\n{}: Cargando modelo:\t{}".format(datetime.datetime.now(),
                                                      path_modelo))
            AUTOARIMA = pickle.load(open(path_modelo, 'rb'))
        else:
            # Crear modelo
            print("\n{}: Inicio entrenamiento".format(datetime.datetime.now()))
            AUTOARIMA = auto_arima(df_train[column].values)
            # Guardar modelo
            print("\n{}: Guardando modelo:\t{}".format(datetime.datetime.now(),
                                                       path_modelo))
            pickle.dump(AUTOARIMA, open(path_modelo, 'wb'))
            # Tiempo de entrenamiento
            t_total = datetime.datetime.now() - tiempo_inicio_fit
            print("\n{}: Tiempo total:\t{}".format(datetime.datetime.now(),
                                                   t_total))
            res_tiempos['t_total'][nombre_modelo] = t_total.total_seconds()
            time_tabla_path = os.path.join(tabla_mercado_path,
                                           nombre_modelo + "_t.csv")
            res_tiempos_df = pd.DataFrame(res_tiempos)
            res_tiempos_df.to_csv(time_tabla_path)
            print("\n{}: Guardando tiempo total en:\t{}\n\n".format(
                datetime.datetime.now(), time_tabla_path))

        # Sumario
        print("\n{}: SUMARIO: ".format(datetime.datetime.now()))
        pprint.pprint(AUTOARIMA.summary())

        # Diagnósticos
        print("\n{}: plot_diagnostics: ".format(datetime.datetime.now()))
        diagnostics_plot_path = os.path.join(plot_mercado_path,
                                             nombre_modelo + "_d.png")
        AUTOARIMA.plot_diagnostics(figsize=(20, 20))
        plt.savefig(diagnostics_plot_path)
        plt.show()

        # Parámetros
        print("\n{}: Parámetros: ".format(datetime.datetime.now()))
        pprint.pprint(AUTOARIMA.get_params())

        # Pronóstico
        print("\n{}: Pronosticando".format(datetime.datetime.now()))
        fc = AUTOARIMA.predict(horizonte_prevision)
        df_test[nombre_modelo] = fc
        pronos_tabla_path = os.path.join(tabla_mercado_path,
                                         nombre_modelo + "_p.csv")
        print("\n{}: Guardando pronósticos en:\t{}".format(datetime.datetime.now(),
                                                           pronos_tabla_path))
        df_test.to_csv(pronos_tabla_path)

        # Plot pronóstico
        print("\n{}: Plot pronóstico modelo:\t{}".format(datetime.datetime.now(),
                                                         nombre_modelo))
        prono_plot_path = os.path.join(plot_mercado_path, nombre_modelo + "_p.png")
        plot_pronos(df_test, nombre_modelo, prono_plot_path, column)

        # Métricas
        print("\n{}: Evaluando pronósticos".format(datetime.datetime.now()))
        metrics = metrics_time_series(df_test[column], fc)
        res_pronos['ME'][nombre_modelo] = metrics[0]
        res_pronos['R2'][nombre_modelo] = metrics[1]
        res_pronos['EVS'][nombre_modelo] = metrics[2]
        res_pronos['MAE'][nombre_modelo] = metrics[3]
        res_pronos['MSE'][nombre_modelo] = metrics[4]
        res_pronos['MAPE'][nombre_modelo] = metrics[5]
        metric_tabla_path = os.path.join(tabla_mercado_path,
                                         nombre_modelo + "_m.csv")
        res_pronos_df = pd.DataFrame(res_pronos)
        res_pronos_df.filter(regex=RE, axis=0).to_csv(metric_tabla_path)
        print("\n{}: Guardando métricas en:\t{}".format(datetime.datetime.now(),
                                                        metric_tabla_path))

        gc.collect()
        # Plot Métricas
        plot_metrics(column, res_pronos_df, plot_mercado_path, RE)

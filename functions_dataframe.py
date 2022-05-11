# Funciones para los DataFrames de pandas
import pandas as pd


def segment_feature(df, region_name, feature):
    """
    Con esta función filtraremos el segmento específico y la característica a evaluar.
    :param df: dataframe
    :param region_name: región
    :param feature: característica
    :return: diccionario separado ctrl y desn, y lista.
    """
    ctrl = df[df["Region Name"] == region_name]
    ctrl = list(ctrl[df["Label"] == False][feature])
    desn = df[df["Region Name"] == region_name]
    desn = list(desn[df["Label"] == True][feature])
    dict_ctrl_desn = {"ctrl": ctrl, "desn": desn}
    list_ctrl_desn = dict_ctrl_desn["ctrl"] + dict_ctrl_desn["desn"]
    return dict_ctrl_desn, list_ctrl_desn


def data_target(df, region_name):
    """
    Función para separar datos de los targets o labels.
    :param df: dataframe
    :param region_name: región
    :return: datos con target, datos sin target, target
    """
    data = df[df["Region Name"] == region_name]
    target = data["Label"]
    return data.drop(columns=["Region Name"]), data.drop(columns=["Label", 'Region Name']), target


def df_features_extraction(df, df_features, label_list):
    """
    Función para crear columnas que concatenen REGIÓN + CARACTERÍSTICA. Se extrae del df completo
    únicamente las características seleccionadas como mejores.

    :param df: DataFrame completo con columna de etiquetas de nombre 'Labels'.
    :param df_features: Dataframe de características principales.
    :param label_list: lista de las etiquetas de manera ordenada.
    :return: DataFrame con las columnas con nombres REGIÓN + CARACTERÍSTICA
    """
    df_features_svm = pd.DataFrame()
    df.sort_values('Label')
    for i in range(0, len(df_features)):
        df_features_svm[df_features['Feature_Region'].iloc[i]] = list \
            (df[df_features['Feature'].iloc[i]][df['Region Name'] == df_features['Region'].iloc[i]])

    df_features_svm['Label'] = label_list

    return df_features_svm


def join_feature_region(df):
    """
    Función para unir el nombre de la región con el nombre de la característica correspondiente.
    :param df: DataFrame.
    :return: DataFrame con columna 'Feature_Region'.
    """
    list_f_r = []
    for i in range(0, len(df)):
        list_f_r.append(df["Feature"].iloc[i] + "_" + df["Region"].iloc[i])
    df["Feature_Region"] = list_f_r
    return df


def division_ctrl_des(df, df_features, i):
    """
    Función para obtener las poblaciones a comparar en la diferencia de medias de sus
    respectivas distribuciones.
    :param df: DataFrame con todas las características
    :param df_features: DataFrame de las características
    :param i: Número de característica
    :return: dos DF que separa los sujetos control y desnutrición.
    """
    sample1 = pd.DataFrame()
    sample2 = pd.DataFrame()
    sample1[df_features['Feature_Region'].iloc[i]] = \
        df[df['Region Name'] == df_features['Region'].iloc[i]][df["Label"] == 0][df_features['Feature'].iloc[i]]
    sample2[df_features['Feature_Region'].iloc[i]] = \
        df[df['Region Name'] == df_features['Region'].iloc[i]][df["Label"] == 1][df_features['Feature'].iloc[i]]
    return sample1, sample2


def save_csv(df, file_name):
    """
    Función para guardar los DataFrames.
    :param df: DF a guardar.
    :param file_name: string con nombre del archivo.
    :return: Se guarda en la carpeta /databases.
    """

    return df.to_csv('databases/' + file_name + '.csv')


def X_y(df_feat, column_y):
    """
    Función para separar los conjuntos de datos (columnas) de su respectiva etiqueta.
    :param df_feat: df con los valores de sus características en cada columna.
    :param column_y: columna con las etiquetas.
    :return: X -> Data sin etiquetas , y -> Etiquetas.
    """
    y = df_feat[column_y]
    X = df_feat.drop(columns=column_y, axis=1)
    return X, y

import numpy as np

def X_y_X_names(df_feat,labels,dimensiones):
    """
    Función para separar data, etiquetas y nombres.
    :param df_feat:
    :param labels:
    :param dimensiones:
    :return:
    """
    training_set_label = df_feat[0:len(labels)]
    training_set_label['Label'] = labels

    X_training,y_training = X_y(training_set_label,'Label')
    return np.array(X_training.iloc[:,0:dimensiones]), np.array(y_training), list(X_training.iloc[:,0:dimensiones])
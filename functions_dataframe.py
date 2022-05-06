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



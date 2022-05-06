import scipy.stats as stats
import pandas as pd
from functions_dataframe import data_target


# _____________________________________
# _____________  ANOVA  _______________
# _____________________________________


def value_anova(df, feature, value):
    """
    Función para obtener f value y p value.
    :param df: dataframe
    :param feature: característica para filtrar dataframe
    :param value: "p" o "f"
    :return: p_value ó f_value
    """
    fvalue, pvalue = stats.f_oneway(df[df['Label'] == 0][feature], df[df['Label'] == 1][feature])
    if value == "f":
        return fvalue
    if value == "p":
        return pvalue


def all_values_anova(df, lista_region_name, features, p_value):
    """
    Función para obtener el DataFrame de las características con un p_value menor al que se declara en la variable p_value.

    Columnas del DataFrame:
    | Región | Característica | F-Value | P-Value |
    :param df: DataFrame con todas las características
    :param lista_region_name: lista de las regiones.
    :param features: lista de las características.
    :param p_value: valor P que no debe rebasar
    :return: DataFrame ordenado en orden ascendente al p-value.
    """
    p = p_value
    lista_nivel0 = []
    for reg in lista_region_name:
        lista_nivel1 = []
        for feat in features:
            data_and_target, data, target = data_target(df, reg)
            p_value = value_anova(data_and_target, feat, "p")
            f_value = value_anova(data_and_target, feat, "f")
            lista_nivel2 = [reg, feat, f_value, p_value]
            lista_nivel1.append(lista_nivel2)
        lista_nivel0 = lista_nivel0 + lista_nivel1

    df_values = pd.DataFrame(lista_nivel0, columns=["Region", "Feature", "F-value", "P-value"])
    df_values = df_values.sort_values('P-value', ascending=True)[df_values['P-value'] < p]

    return df_values


# _____________________________________
# _________MUTUAL INFORMATION _________
# _____________________________________


def all_values_mi(df, lista_region_name, mi_score):
    """
    Función para obtener el DataFrame de las características con un mi_score mayor al que se declara en la variable mi_score.

    Columnas del DataFrame:
    | Característica | MI score | Region |

    :param df: DataFrame del set original.
    :param lista_region_name: lista de las regiones
    :return: DataFrame con el MI score > mi_score
    """
    mi_scores_all = pd.DataFrame()
    for reg in lista_region_name:
        X_y, X, y = data_target(df, reg)
        for colname in X.select_dtypes("object"):
            X[colname], _ = X[colname].factorize()

        # All discrete features should now have integer dtypes (double-check this before using MI!)
        discrete_features = X.dtypes == int

        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        def make_mi_scores(X, y, discrete_features):
            mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
            mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
            mi_scores = mi_scores.sort_values(ascending=False)
            return mi_scores

        mi_scores = make_mi_scores(X, y, discrete_features)

        mi_scores = pd.DataFrame(mi_scores)
        mi_scores = mi_scores.reset_index()

        list_region = [reg for i in range(len(mi_scores))]
        mi_scores["Region"] = list_region
        mi_scores.columns = ["Feature", "MI score", "Region"]

        mi_scores_all = pd.concat([mi_scores_all, mi_scores])
    mi_scores_all = mi_scores_all.sort_values("MI score", ascending=False)
    return mi_scores_all[mi_scores_all["MI score"] > mi_score]


# _____________________________________
# _______________  KDE  _______________
# _____________________________________
import seaborn as sns
import matplotlib.pyplot as plt


def pairplot_kde(df_features_col, title = 'Título'):
    """
    Función para plotear los resultados de la densidad de kernel y scatter plot.

    :param df_features_col: DataFrame con las columnas como características
    :param title: título del plot
    :return: pairplot
    """
    X = df_features_col
    y = []
    label = list(df_features_col['Label'])
    for lab in label:

        if lab == 0:
            y.append('Control')
        else:
            y.append('Desnutrición')

    X['Label'] = y
    g = sns.pairplot(X, hue="Label", palette="bright", diag_kind="kde")
    g.map_lower(sns.kdeplot, levels=4, color=".2")

    for ax in g.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), rotation=90)
        ax.set_ylabel(ax.get_ylabel(), rotation=0)
        ax.yaxis.get_label().set_horizontalalignment('right')

    plt.suptitle(title, y=1.05)
    plt.show()
    plt.close()
    df_features_col['Label'] = label


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


def pairplot_kde(df_features_col, title='Título'):
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


# _____________________________________
# _____Diferencia de Medias____________
# _____________________________________
import numpy as np
from scipy.stats import t
from numpy import mean, sqrt, std


def difference_means_confidence_interval(sample1, sample2, alpha):
    """
    Función para calcular el intervalo de confianza de la diferencia de medias.
    :param sample1: Muestra 1.
    :param sample2: Muestra 2.
    :param alpha: Nivel de significancia.
    :return: Intervalos de confianza: mínimo, diferencia y máximo.
    """
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    # sample data
    n1, m1, s1 = len(sample1), mean(sample1), std(sample1)
    n2, m2, s2 = len(sample2), mean(sample2), std(sample2)

    # statistical stuff
    K = sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    v = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    d = K * t.ppf(1. - alpha / 2, v)

    # confidence interval
    diff = m1 - m2
    low = diff - d
    high = diff + d

    # print results and return
    print(f'Diferencia = {diff:.3f}')
    print(f'Intervalos de confianza ({1 - alpha:.0%}): [{low:.3f},{diff:.3f}, {high:.3f}]')
    return low, diff, high


# _____________________________________
# __________ Bootstrapping ____________
# _____________________________________


def bootstrap_fn(sample1, sample2, plot, title, iterations, alpha):
    """
    Función para hacer bootstrapping
    :param sample1: distribución 1 con .squeeze().
    :param sample2: distribución 2 con .squeeze().
    :param plot: False default, True para plotear gráfica.
    :param title: título del gráfico
    :param fn: Función
    :param iterations: Número de iteraciones
    :param alpha: Nivel de significancia.
    :return: Intervalos de confianza: mínimo, diferencia y máximo.
    """
    fn = lambda x1, x2: mean(x1) - mean(x2)

    # muestreo original y estadísticas
    n1 = len(sample1)
    n2 = len(sample2)
    sample_fn = fn(sample1, sample2)

    # algoritmo de boostrap
    bootstrap_dist = []
    for i in range(iterations):
        resample1 = np.random.choice(sample1, n1, replace=True)
        resample2 = np.random.choice(sample2, n2, replace=True)
        estimator = fn(resample1, resample2)
        bootstrap_dist.append(estimator)

    # media, std, bias
    bootstrap_mean = np.mean(bootstrap_dist)
    bootstrap_se = np.std(bootstrap_dist)
    bias = sample_fn - bootstrap_mean

    # calculamos el intervalo de confianza por percentil boostrap
    high = np.quantile(bootstrap_dist, 1 - alpha / 2)
    median = np.quantile(bootstrap_dist, .5)
    low = np.quantile(bootstrap_dist, alpha / 2)

    # plot distribución bootstrap
    if plot == True:
        sns.histplot(bootstrap_dist, stat='density')
        sns.kdeplot(bootstrap_dist)
        plt.title(title)
        plt.show()
        plt.close()

        print(f'{n1=} {n2=} {iterations=}')
        print(f'{sample_fn=:.3f} {bootstrap_mean=:.3f} {bias=:.3f} {bootstrap_se=:.3f}')
        print(f'Intervalos de Confianza ({1 - alpha:.0%}): [{low:.3f} ... {median:.3f} ... {high:.3f}]')

    return low, median, high


from functions_dataframe import division_ctrl_des


def lista_feats_diff_0(df, df_feats):
    """
    Con esta función podemos comparar característica por característica, ambas distribuciones
    y ver con un 95% de significancia si las medias se parecen.
    :param df: DataFrame completo.
    :param df_feats: DF de características principales
    :return: lista de características en las que la población podrían parecerse.
    """
    lista_feats_dif_0 = []
    for i in range(0, len(df_feats)):
        sample1, sample2 = division_ctrl_des(df, df_feats, i)
        low, diff, high = bootstrap_fn(sample1.squeeze(), sample2.squeeze(), False, list(sample1)[0], iterations=10000,
                                       alpha=0.05)
        if low < 0 < high:
            lista_feats_dif_0.append(list(sample1)[0])

    return lista_feats_dif_0



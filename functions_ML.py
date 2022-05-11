from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np


def svm_2d(X, y, X_names, alg_name):
    """
    Función para plotear los márgenes de los vectores soportes y su hiperplano separador.
    :param X: Array con data 2D sin etiquetas.
    :param y: Array de las etiquetas.
    :param X_names: Lista de nombres de las características
    :param alg_name: Nombre del algoritmo.
    :return:
    """
    plt.rcParams["figure.figsize"] = (15, 9)
    plt.rcParams.update({'font.size': 22})

    clf = svm.SVC(kernel='poly', C=10)

    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
    )
    # plot support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.xlabel(X_names[0])
    plt.ylabel(X_names[1])
    plt.title("SVM Primeras dos características " + alg_name)
    plt.show()
    plt.close()


def svm_3d(X, y, X_names):
    """
    Función para plotear 3 dimensiones o características.
    :param X: Array con data 2D sin etiquetas.
    :param y: Array de las etiquetas.
    :param X_names: Lista de nombres de las características.
    :return:
    """
    # make it binary classification problem
    Y = y
    X = X[np.logical_or(Y == 0, Y == 1)]
    Y = Y[np.logical_or(Y == 0, Y == 1)]

    model = svm.SVC(kernel='linear')
    clf = model.fit(X, Y)

    # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
    # Solve for w3 (z)
    z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]

    tmp = np.linspace(0, 0, 0)
    x, y = np.meshgrid(tmp, tmp)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], 'ob')
    ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], 'sr')
    ax.plot_surface(x, y, z(x, y))
    ax.view_init(30, 60)

    ax.set_xlabel(X_names[0], fontsize=15, loc='left')
    ax.set_ylabel(X_names[1], fontsize=15)
    ax.set_zlabel(X_names[2], fontsize=15)
    plt.tick_params(labelsize=9)
    plt.show()
    plt.close()


import functions_dataframe
from functions_dataframe import X_y_X_names


def svm_n_dimensiones(df_feat_training, labels_training, df_validation_set, labels_validation, n):
    """
    Función para entrenar y validar.
    :param df_feat_training: Dataframe de conjunto de entrenamiento.
    :param labels_training: Lista de etiquetas conjunto de entrenamiento.
    :param df_validation_set: Dataframe de conjunto de validación.
    :param labels_validation: Lista de etiquetas validación.
    :param n: Dimensión.
    :return: Accuracy.
    """
    X, y, X_names = X_y_X_names(df_feat_training, labels_training, n)
    clf = svm.SVC(gamma='auto')
    clf.fit(X, y)

    X_val, y_val, X_val_names = X_y_X_names(df_validation_set, labels_validation, n)

    lista_tf = []
    for i in range(0, len(y_val)):
        if clf.predict([X_val[i]])[0] == y_val[i]:
            lista_tf.append(1)
        else:
            lista_tf.append(0)

    print('Accuracy: ' + str(np.mean(lista_tf)))
    return np.mean(lista_tf)

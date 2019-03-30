# Author: llw
import sys
import pickle

import cv2
import numpy as np
from sklearn.decomposition import PCA


def get_pca_decomposition(data, n=3):
    pca = PCA(n_components=n)
    return pca.fit_transform(data)


def plot_3d_data(data, labels):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    plt.axes(projection='3d')
    for cor in set(labels):
        mask = (labels == cor)
        plt.scatter(
            data[mask, 0],
            data[mask, 1],
            data[mask, 2],
            label="{}".format(cor),
            facecolor="C{}".format(cor),
            linewidths=0.001
        )
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_2d_data(data, labels):
    import matplotlib.pyplot as plt
    for cor in set(labels):
        mask = (labels == cor)
        if cor == 9:
            plt.scatter(
                data[mask, 0],
                data[mask, 1],
                label="{}".format(cor),
                facecolor="C{}".format(cor),
                linewidths=0.001
            )
    plt.tight_layout()
    plt.legend()
    plt.show()


def sigma_normalize(array):
    return (array - np.mean(array, axis=0)) / np.std(array, axis=0)


if __name__ == '__main__':
    data_path = "../data/"
    train_dicts = []
    for i in range(1):
        with open(data_path + "data_batch_{}".format(i + 1), "rb") as file:
            dic = pickle.load(file, encoding='bytes')
            train_dicts.append(dic)

    def get_data(dic):
        return [np.array(dic[b'data']), np.array(dic[b'labels'])]

    dim = 2
    select_dict = train_dicts[0]
    X, y = get_data(select_dict)
    X = get_pca_decomposition(X, dim)
    X = sigma_normalize(X)
    if dim == 2:
        plot_2d_data(X, y)
    else:
        plot_3d_data(X, y)

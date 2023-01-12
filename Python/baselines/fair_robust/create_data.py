from numpy import genfromtxt
import numpy as np
import random
import collections
from math import sqrt
from sklearn.cluster import KMeans
# filename = 'adult_norm.csv'
# numpy_data = genfromtxt(filename, delimiter=',')
# numpy_data = numpy_data[np.argsort(numpy_data[:,5])]



def weighted_adult_data_with_unlabel(numpy_data):
    train_size = 35000
    numpy_data_train = numpy_data[:train_size]
    weight = [1.0]*len(numpy_data_train)
    bound = [1.0]*len(numpy_data_train)
    numpy_data_test = numpy_data[train_size:]
    maps_train = collections.defaultdict(list)
    maps_test = {}
    feature_size = 36

    for i in range(len(numpy_data_train)):
        maps_train[tuple(numpy_data_train[i][1:feature_size])].append(i)

    index_list = []
    presum = 0
    random.seed(1)
    np.random.seed(1)
    count = 0
    sigma = 0.1
    p_0 = 0.01
    cluster = []
    real_upper = []
    real_lower = []
    for key, val in maps_train.items():
        if len(val) > 10:
            n = random.random()
            while 0.35 < n < 0.65 or n > 0.85 or n < 0.15:
                n = random.random()
            n = int(len(val)*n)
            index = np.random.choice(val, n, replace=False)
            index_list.extend(index)
            if len(index) > 0:
                cluster.append(len(index))
            for ele in index:
                weight[ele] = n / len(val)
                temp = (np.log2(2*train_size/2) + np.log2(1.0/sigma)) / (p_0*train_size)
                bound[ele] = sqrt(temp)
            if weight[ele] - bound[ele] < 0:
                real_lower.append(0.1)
            else:
                real_lower.append(weight[ele] - bound[ele])
            real_upper.append(weight[ele] + bound[ele])

    sen = numpy_data_train[:, 0]
    sen = np.array(sen)
    weight = np.array(weight)
    weight = weight.reshape(len(weight), 1)
    bound = np.array(bound)
    bound = bound.reshape(len(bound), 1)
    label = np.take(numpy_data_train[:, -1], index_list, axis=0)
    numpy_data_train = numpy_data_train[:, 1:feature_size]
    numpy_data_train = np.take(numpy_data_train, index_list, axis=0)

    weight = np.take(weight, index_list, axis=0)


    upper_bound = []
    lower_bound = []
    for i in range(len(real_lower)):
        upper_bound.append(1.0 / real_lower[i])
        lower_bound.append(-1.0 / real_upper[i])
    weight_data = np.concatenate((numpy_data_train, weight), axis=1)



    # print(weight)
    sen = sen.reshape(len(sen), 1)
    sen = np.take(sen, index_list, axis=0)

    return weight_data, label, lower_bound, upper_bound, sen, cluster


def weighted_adult_data_kmeans_wihtout_unlabel(numpy_data):
    train_size = 35000
    numpy_data_train = numpy_data[:train_size]
    maps_train = collections.defaultdict(list)
    maps = {}
    feature_size = 36

    for i in range(len(numpy_data_train)):
        maps_train[tuple(numpy_data_train[i][1:feature_size])].append(i)

    index_list = []
    random.seed(1)
    np.random.seed(1)
    cluster_size = 300
    cluster = []
    real_upper = []
    real_lower = []
    rho = 0.02

    for index in range(train_size):
        if numpy_data_train[index][37] >= 0.625:
            seed = random.random()
            # if seed > 0.3:
            index_list.append(index)
        else:
            seed = random.random()
            # if seed > 0.7:
            index_list.append(index)




    sen = numpy_data_train[:, 0]
    sen = np.array(sen)


    label = np.take(numpy_data_train[:, -1], index_list, axis=0)


    numpy_data_train = numpy_data_train[:, 1:feature_size]
    numpy_data_train = np.take(numpy_data_train, index_list, axis=0)
    kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(numpy_data_train)
    cluster_list = kmeans.labels_
    cluster_list = np.array(cluster_list)

    for class_label in cluster_list:
        maps[class_label] = maps.get(class_label, 0) + 1

    weight = [0.0]*len(cluster_list)
    index = 0


    for key in sorted(maps.keys()):
        value = maps[key]
        for i in range(index, index + value):
            weight[i] = value / len(weight)
        lambda_hat = value / len(weight)

        if lambda_hat - rho < 0.01:
            real_lower.append(0.02)
        else:
            real_lower.append(lambda_hat - rho)

        real_upper.append(lambda_hat + rho)


        cluster.append(value)
        index += value



    upper_bound = []
    lower_bound = []

    for i in range(len(real_lower)):
        upper_bound.append(1.8)
        lower_bound.append(-0.2)

    # print(upper_bound)
    # print(lower_bound)


    sen = sen.reshape(len(sen), 1)
    cluster_list = cluster_list.reshape(len(cluster_list), 1)
    sen = np.take(sen, index_list, axis=0)
    sen_label_list = np.concatenate((sen, cluster_list), axis=1)

    sen_label_list = sen_label_list[np.argsort(sen_label_list[:, -1])]


    weight_data = np.concatenate((numpy_data_train, cluster_list), axis=1)
    weight_data = weight_data[np.argsort(weight_data[:, -1])]
    weight_data = weight_data[:, 0: -1]
    weight = np.array(weight)
    weight = weight.reshape(len(weight), 1)
    weight_data = np.concatenate((weight_data, weight), axis=1)



    label = label.reshape(len(label), 1)
    # temp = label
    # print(sum(label))
    label = np.concatenate((label, cluster_list), axis=1)
    label = label[np.argsort(label[:, -1])]
    label = label[:, 0]

    sen = sen_label_list[:, 0]
    sen = sen.reshape(len(sen), 1)



    return weight_data[:,:-1], label, lower_bound, upper_bound, sen, cluster


def weighted_dutch_data_with_unlabel(numpy_data):
    train_size = 45000
    numpy_data_train = numpy_data[:train_size]
    weight = [1.0] * len(numpy_data_train)
    bound = [1.0] * len(numpy_data_train)
    numpy_data_test = numpy_data[train_size:]
    maps_train = collections.defaultdict(list)
    maps = collections.defaultdict(list)
    feature_size = 35

    for i in range(len(numpy_data_train)):
        maps_train[tuple(numpy_data_train[i][1:feature_size - 1])].append(i)

    old_index_list = []
    index_list = []
    presum = 0
    random.seed(1)
    np.random.seed(1)
    count = 0
    sigma = 0.1
    p_0 = 0.01
    cluster = []
    real_upper = []
    real_lower = []

    for index in range(train_size):
        if numpy_data_train[index][33] >= 1.0:
            seed = random.random()
            if seed > 0.25:
                old_index_list.append(index)
        else:

            seed = random.random()
            if seed > 0.75:
                old_index_list.append(index)



    for index in old_index_list:
        maps[tuple(numpy_data_train[index][1:feature_size - 1])].append(index)

    for key, val in maps.items():
        if len(val) >= 20:
            index_list.extend(val)

    right_term = 0.33
    for key, val in maps.items():
        if len(val) < 20: continue
        cluster.append(len(val))
        for ele in val:
            weight[ele] = len(val) / len(maps_train[key])

        if weight[ele] - right_term < 0.2:
            real_lower.append(0.2)
        else:
            real_lower.append(weight[ele] - right_term)

        if weight[ele] + right_term >= 1:
            real_upper.append(1)
        else:
            real_upper.append(weight[ele] + right_term)

    sen = numpy_data_train[:, 0]
    sen = np.array(sen)
    weight = np.array(weight)
    weight = weight.reshape(len(weight), 1)

    label = np.take(numpy_data_train[:, -1], index_list, axis=0)
    numpy_data_train = numpy_data_train[:, 1:feature_size]
    numpy_data_train = np.take(numpy_data_train, index_list, axis=0)

    weight = np.take(weight, index_list, axis=0)
    #
    #
    upper_bound = []
    lower_bound = []
    for i in range(len(real_lower)):
        upper_bound.append(1.0 / real_lower[i])
        lower_bound.append(-1.0 / real_upper[i])
    weight_data = np.concatenate((numpy_data_train, weight), axis=1)

    sen = sen.reshape(len(sen), 1)
    sen = np.take(sen, index_list, axis=0)


    # # print(len(weight_data))
    # return weight_data, label, lower_bound, upper_bound, sen, cluster


def weighted_dutch_data_kmeans_wihtout_unlabel(numpy_data):
    train_size = 45000
    numpy_data_train = numpy_data[:train_size]
    weight = [1.0] * len(numpy_data_train)
    bound = [1.0] * len(numpy_data_train)
    numpy_data_test = numpy_data[train_size:]
    maps_train = collections.defaultdict(list)
    maps = collections.defaultdict(list)
    feature_size = 35

    for i in range(len(numpy_data_train)):
        maps_train[tuple(numpy_data_train[i][1:feature_size - 1])].append(i)

    index_list = []
    random.seed(1)
    np.random.seed(1)
    cluster_size = 300
    cluster = []
    real_upper = []
    real_lower = []
    rho = 0.02

    for index in range(train_size):
        if numpy_data_train[index][33] >= 1.0:
            seed = random.random()
            if seed > 0.25:
                index_list.append(index)
        else:

            seed = random.random()
            if seed > 0.75:
                index_list.append(index)




    sen = numpy_data_train[:, 0]
    sen = np.array(sen)


    label = np.take(numpy_data_train[:, -1], index_list, axis=0)


    numpy_data_train = numpy_data_train[:, 1:feature_size]
    numpy_data_train = np.take(numpy_data_train, index_list, axis=0)
    kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(numpy_data_train)
    cluster_list = kmeans.labels_
    cluster_list = np.array(cluster_list)

    for class_label in cluster_list:
        maps[class_label] = maps.get(class_label, 0) + 1

    weight = [0.0]*len(cluster_list)
    index = 0


    for key in sorted(maps.keys()):
        value = maps[key]
        for i in range(index, index + value):
            weight[i] = value / len(weight)
        lambda_hat = value / len(weight)

        if lambda_hat - rho < 0.01:
            real_lower.append(0.02)
        else:
            real_lower.append(lambda_hat - rho)

        real_upper.append(lambda_hat + rho)


        cluster.append(value)
        index += value



    upper_bound = []
    lower_bound = []

    for i in range(len(real_lower)):
        upper_bound.append(1.8)
        lower_bound.append(-0.2)

    # print(upper_bound)
    # print(lower_bound)


    sen = sen.reshape(len(sen), 1)
    cluster_list = cluster_list.reshape(len(cluster_list), 1)
    sen = np.take(sen, index_list, axis=0)
    sen_label_list = np.concatenate((sen, cluster_list), axis=1)

    sen_label_list = sen_label_list[np.argsort(sen_label_list[:, -1])]


    weight_data = np.concatenate((numpy_data_train, cluster_list), axis=1)
    weight_data = weight_data[np.argsort(weight_data[:, -1])]
    weight_data = weight_data[:, 0: -1]
    weight = np.array(weight)
    weight = weight.reshape(len(weight), 1)
    weight_data = np.concatenate((weight_data, weight), axis=1)



    label = label.reshape(len(label), 1)
    # temp = label
    # print(sum(label))
    label = np.concatenate((label, cluster_list), axis=1)
    label = label[np.argsort(label[:, -1])]
    label = label[:, 0]

    sen = sen_label_list[:, 0]
    sen = sen.reshape(len(sen), 1)



    return weight_data, label, lower_bound, upper_bound, sen, cluster




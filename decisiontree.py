import numpy as np
import pandas as pd
import math as math
import node as nd
import operator


def main():
    print("decision tree")
    pass


def test():
    print("test")
    df = pd.DataFrame({
        'col1': ['A', 'A', 'B', np.nan, 'D', 'C'],
        'col2': [2, 1, 9, 8, 7, 4],
        'col3': [0, 1, 9, 4, 2, 3],
        'col4': ['a', 'B', 'c', 'D', 'e', 'F']
    })
    print(df)
    print("================")
    result = df.sort_values(by=['col1'])
    print(df)
    print("++++++++++++")
    print(result)
    l = [0, 1, 2, 3, 4, 5]
    l_dict = dict.fromkeys(l, 0)
    print(l_dict)
    l_dict[2] += 1
    print(l_dict)
    test_d = dict()
    print(test_d)

    data = pd.read_csv('data/iris.data', header=None)
    print(data.head())
    listofclusters = data[4].unique()
    print(listofclusters)
    eval_numeric_attr(data, 0, 3, 4, listofclusters)

    eval_numeric_attr(data, 1, 3, 4, listofclusters)


def test2():
    data = pd.read_csv('data/iris.data', header=None)
    training = pd.read_csv('data/iris.test.data', header=None)
    print(data)
    listofclusters = data[4].unique()
    node = createdecisionTree(data, 5, 0.9, [0, 1, 2, 3], 4, listofclusters)
    print(node)
    l = node.predictlabel(training.iloc[0].to_numpy())
    print(l)
    node.predict_data_set(training)
    print(training)


def createdecisionTree(data, neta, phi, listofattributes, labelattribute, listofclusters):
    # create tree and initialize it to empty
    initialnode = nd.Node()
    result = decisiontree(initialnode, data, neta, phi, listofattributes, labelattribute, listofclusters)
    return result


def decisiontree(node, data, neta, phi, listofattributes, labelattribute, listofclusters):
    n = data.shape[0]
    freq_of_classes_dict = dict.fromkeys(listofclusters, 0)
    # calculate the frequency
    for i in range(n):
        freq_of_classes_dict[data.iloc[i][labelattribute]] += 1

    max_purity_class = max(freq_of_classes_dict.items(), key=operator.itemgetter(1))[0]
    purity_D = freq_of_classes_dict[max_purity_class] / n
    # print(max_purity_class)
    # print(purity_D)

    if n <= neta or purity_D >= phi:
        c_star = max_purity_class
        node.createleafnode(c_star, n)
        return node

    split_point = np.nan
    best_score = 0
    split_attribute = np.nan
    ny = 0
    nn = 0

    for attribute in listofattributes:
        v, score, ny, nn = eval_numeric_attr(data, attribute, labelattribute, listofclusters)
        if score > best_score:
            split_point = v
            best_score = score
            split_attribute = attribute

    sorted_data = data.sort_values(split_attribute)

    sliced_dy = sorted_data.iloc[0:ny]
    sliced_dn = sorted_data.iloc[ny:]
    # print("sliced_dy\n", sliced_dy)
    # print("%%%%%%%")
    # print("sliced_dn\n", sliced_dn)
    nodeleft, noderight = node.createinternalndoesplit(split_attribute, split_point, best_score, ny, nn)
    decisiontree(nodeleft, sliced_dy, neta, phi, listofattributes, labelattribute, listofclusters)
    decisiontree(noderight, sliced_dn, neta, phi, listofattributes, labelattribute, listofclusters)
    return node


def eval_numeric_attr(data, attribute, labelattribute, listofclusters):
    n = data.shape[0]  # number of points
    midpoints = []
    sorted_data = data.sort_values(attribute)
    # print("__+_+_+_+_+_+_+")
    # print(sorted_data)
    # frequency of classes set to zero, kept as a dictionary against the cluster label
    freq_of_classes_dict = dict.fromkeys(listofclusters, 0)
    N_vi = dict()

    for j in range(n - 1):
        freq_of_classes_dict[sorted_data.iloc[j][labelattribute]] += 1

        x_j = sorted_data.iloc[j][attribute]
        x_j_1 = sorted_data.iloc[j + 1][attribute]
        if x_j_1 != x_j:
            v = x_j + (x_j_1 - x_j) / 2
            midpoints.append(v)
            if v not in N_vi.keys():
                N_vi[v] = dict.fromkeys(listofclusters, 0)
            for i in listofclusters:
                N_vi.get(v)[i] = freq_of_classes_dict[i]

    freq_of_classes_dict[sorted_data.iloc[n - 1][labelattribute]] += 1

    # print(freq_of_classes_dict)
    # print(N_vi)

    # print(sum(list(N_vi.get(5.25).values())))

    # evaluating split points
    v_best = np.nan
    best_score = 0
    ny = 0
    nn = 0
    for v in midpoints:
        p_ci_Dy = dict.fromkeys(listofclusters, 0.0)
        p_ci_Dn = dict.fromkeys(listofclusters, 0.0)
        sigma_N_vj = sum(list(N_vi.get(v).values()))
        sigma_nj_Nvj = 0
        # for key in N_vi.get(v).keys():
        #     sigma_nj_Nvj = sigma_nj_Nvj + (freq_of_classes_dict[key] - N_vi.get(v)[key])
        sigma_nj_Nvj = n - sigma_N_vj
        for i in listofclusters:
            p_ci_Dy[i] = N_vi.get(v).get(i) / sigma_N_vj
            p_ci_Dn[i] = (freq_of_classes_dict[i] - N_vi.get(v).get(i)) / sigma_nj_Nvj

        # print(p_ci_Dy)
        # print(p_ci_Dn)
        score_x_less_or_eq_v = gain(n, freq_of_classes_dict, p_ci_Dy, p_ci_Dn, sigma_N_vj, sigma_nj_Nvj)
        if score_x_less_or_eq_v > best_score:
            v_best = v
            best_score = score_x_less_or_eq_v
            ny = sigma_N_vj
            nn = sigma_nj_Nvj

    print("Best v: ", v_best, " best score: ", best_score)
    return v_best, best_score, ny, nn


def gain(n, freq_of_classes_dict, p_ci_Dy, p_ci_Dn, ny, nn):
    # H(D)
    HD = 0
    for value in freq_of_classes_dict.values():
        temp = (value / n) if value != 0 else 0
        if temp == 0: continue
        HD = HD - temp * math.log2(temp)
    # minus it

    HD_Y = 0
    for p_ci_Dy_ele in p_ci_Dy.values():
        temp = p_ci_Dy_ele * math.log2(p_ci_Dy_ele) if p_ci_Dy_ele != 0 else 0
        HD_Y = HD_Y - (temp)

    HD_N = 0
    for p_ci_Dn_ele in p_ci_Dn.values():
        temp = p_ci_Dn_ele * math.log2(p_ci_Dn_ele) if p_ci_Dn_ele != 0 else 0
        HD_N = HD_N - (temp)

    HD_Y_N = (ny / n) * HD_Y + (nn / n) * HD_N

    result = HD - HD_Y_N
    print("gain: ", result)
    return result


if __name__ == '__main__':
    # test()
    test2()

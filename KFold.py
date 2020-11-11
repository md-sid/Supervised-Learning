import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sympy.strategies.core import switch
import decisiontree as dt
import Performance as perf
import knn

def crossValidation(D, K, classifier):
    N = len(D.index)  # Number of entries
    print(str(K)+'-fold cross validation on '+str(N)+' data points')

    listofclusters = D.iloc[:, classifier.gTruthCol].unique()
    listofclusters.sort()

    D = pd.DataFrame(shuffle(D.values))     # Shuffling the dataset

    Theta = np.zeros(K)
    Prec = np.zeros([K, len(listofclusters)])
    Rec = np.zeros([K, len(listofclusters)])

    for i in range(K):

        print('From '+str(i*np.floor(N/K)) + ', to ' + str((i+1)*np.floor(N/K)-1) + ' as test set')
        D_test = D.iloc[int(i*np.floor(N/K)):int((i+1)*np.floor(N/K)-1), :].copy().reset_index(drop=True)
        D_train = D[~D.index.isin(range(int(i*np.floor(N/K)), int((i+1)*np.floor(N/K)-1)))].copy().reset_index(drop=True)

        if classifier.name == 'Dtree':

            node = dt.createdecisionTree(D_train, classifier.neta, classifier.phi, classifier.listofattributes, classifier.gTruthCol, listofclusters)
            result = node.predict_data_set(D_test)
            # print(result)

            Theta[i] = perf.F_measure(result, classifier.gTruthCol, classifier.predCol, listofclusters)
            Prec[i, :] = perf.precision(result, classifier.gTruthCol, classifier.predCol, listofclusters)
            Rec[i, :] = perf.recall(result, classifier.gTruthCol, classifier.predCol, listofclusters)

        elif classifier.name == 'KNN':
            predictions = []

            for j in range(len(D_test)):
                tmp = knn.k_nearest_neighbors(D_train.iloc[:, 0:4].to_numpy(), D_train.iloc[:, classifier.gTruthCol].to_numpy(), D_test.iloc[j, 0:4].to_numpy(), classifier.k)
                predictions.append(tmp)

            result = pd.DataFrame(D_test)
            result.insert(classifier.predCol, 'Results', predictions)

            Theta[i] = perf.F_measure(result, classifier.gTruthCol, classifier.predCol, listofclusters)
            Prec[i, :] = perf.precision(result, classifier.gTruthCol, classifier.predCol, listofclusters)
            Rec[i, :] = perf.recall(result, classifier.gTruthCol, classifier.predCol, listofclusters)


    print('************************************')
    print('***********-FINAL-*********')

    print('Class labels')
    print(listofclusters)

    print('Performance measure in each fold')
    print(str(Theta))

    mu_Theta = np.mean(Theta)
    var_Theta = np.var(Theta)

    print('Mean performance measure = ' + str(mu_Theta))
    print('Variance performance measure = ' + str(mu_Theta))

    print('Precision = ')
    print(Prec)
    print('Mean precision = ' + str(np.mean(Prec)))
    print('Variance Precision = '+ str(np.var(Prec)))

    print('Recall = ')
    print(Rec)
    print('Mean Recall = ' + str(np.mean(Rec)))
    print('Variance Recall = ' + str(np.var(Rec)))

    return mu_Theta, var_Theta

class ModelMethod:
    def __init__(self, name, gTruthCol, predCol,  phi, neta, k, listofattributes):
        self.name = name
        self.gTruthCol = gTruthCol
        self.predCol = predCol
        self.phi = phi
        self.neta = neta
        self.k = k
        self.listofattributes = listofattributes

def main():
    print("K-Fold cross validation")

    dataName = 'iris'
    print('Data = ' + dataName)

    # Can include other datasets and their parameters in the elif structure
    if dataName == 'shuttle':
        data = pd.read_csv('data/shuttle/shuttle.trn', header=None)
        gTruthCol = 9
        predCol = 10

        listofattributes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        neta = 500
        phi = 0.85
        k = 3
    elif dataName == 'satellite':
        data = pd.read_csv('data/satellite/sat.trn', header=None)
        gTruthCol = 9
        predCol = 10

        listofattributes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        neta = 600
        phi = 0.85
        k = 3
    elif dataName == 'iris':
        data = pd.read_csv('data/iris.data', header=None)
        gTruthCol = 4
        predCol = 5

        listofattributes = [0, 1, 2, 3]
        neta = 5
        phi = 0.9
        k = 3


    N = len(data.index)  # Number of entries

    # 'Dtree' or 'KNN' for Decition tree and K nearest neighbour
    name = 'KNN'


    Model = ModelMethod( name, gTruthCol, predCol,  phi, neta, k, listofattributes)
    print('Classification method = ' + Model.name)
    print(Model)

    K = 3  # number of folds in K-fold

    [theta_u, theta_var] = crossValidation(data.copy(), K, Model)

if __name__ == "__main__":
    main()
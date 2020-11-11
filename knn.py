from sklearn.model_selection import train_test_split
import random
from math import sqrt
import pandas as pd
import numpy as np


random.seed(42)

def euclidean_distance(value1, value2):
    distance = 0
    for i in range(len(value1)):
        distance += (value1[i] - value2[i])**2
    return sqrt(distance)
    
def k_nearest_neighbors(data, label, test_instance, k):
    distances = []
    for i in range(len(data)):
        tmp_dist = euclidean_distance(test_instance, data[i])
        distances.append((data[i], tmp_dist, label[i]))
    distances.sort(key=lambda tmp: tmp[1])
    neighbors = distances[:k]
    prediction = []
    for tmp in range(len(neighbors)):
        prediction.append(neighbors[tmp][-1])
    return(max(prediction))


def test_iris():
    # load the data
    print('KNN with Iris Dataset')
    dataset = pd.read_csv('data/iris.data', header=None)
    dataset = np.array(dataset)
    data = dataset[:, 0:4]
    label = dataset[:, 4]
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, test_size=0.2, random_state=12)
    
    k = 6
    
    print('Example Test with ' + str(k) + ' nearest neighbors :')
    prediction = k_nearest_neighbors(data_train, label_train, 
                        data_test[5], k)
    print('True Label : ' + label_test[5], '\nPredicted Label : ' + prediction)
    
    predictions = []
    for i in range(len(data_test)):
        tmp = k_nearest_neighbors(data_train, label_train, 
                        data_test[i], k)
        predictions.append(tmp)
    
    acc = (label_test == predictions).sum()/len(predictions)
    print('Accuracy : ', + acc)
    
    df = pd.DataFrame(data_test)
    df.insert(4, '4', label_test)
    df.insert(5, '5', predictions)
    df.to_csv('data/KNN_6_iris_test_true_predict.csv', index=False, header=False)
    print('Task Complete')

def test_satellite():
    # load the data
    print('KNN with Satellite Dataset')
    dataset = pd.read_csv('data/satellite/sat.trn', header=None)
    dataset = np.array(dataset)
    data_train = dataset[:, 0:36]
    label_train = dataset[:, 36]
    
    test_set = pd.read_csv('data/satellite/sat.tst', header=None)
    test_set = np.array(test_set)
    data_test = test_set[:, 0:36]
    label_test = test_set[:, 36]
    
    k = 2
    
    print('Example Test with ' + str(k) + ' nearest neighbors :')
    prediction = k_nearest_neighbors(data_train, label_train, 
                        data_test[5], k)
    print('True Label : ' + str(label_test[5]), '\nPredicted Label : ' + 
          str(prediction))
    
    predictions = []
    for i in range(len(data_test)):
        tmp = k_nearest_neighbors(data_train, label_train, 
                        data_test[i], k)
        predictions.append(tmp)
    
    acc = (label_test == predictions).sum()/len(predictions)
    print('Accuracy : ', + acc)
    
    df = pd.DataFrame(data_test)
    df.insert(36, '36', label_test)
    df.insert(37, '37', predictions)
    df.to_csv('data/satellite/KNN_2_sat_test_true_predict.csv', index=False, 
              header=False)
    print('Task Complete')


def test_shuttle():
    # load the data
    print('KNN with Shuttle Dataset')
    dataset = pd.read_csv('data/shuttle/shuttle.trn', header=None)
    dataset = np.array(dataset)
    data_train = dataset[:, 0:9]
    label_train = dataset[:, 9]
    
    test_set = pd.read_csv('data/shuttle/shuttle.tst', header=None)
    test_set = np.array(test_set)
    data_test = test_set[:, 0:9]
    label_test = test_set[:, 9]
    
    k = 4
    
    print('Example Test with ' + str(k) + ' nearest neighbors :')
    prediction = k_nearest_neighbors(data_train, label_train, 
                        data_test[5], k)
    print('True Label : ' + str(label_test[5]), '\nPredicted Label : ' + 
          str(prediction))
    
    predictions = []
    for i in range(len(data_test)):
        tmp = k_nearest_neighbors(data_train, label_train, 
                        data_test[i], k)
        predictions.append(tmp)
    
    acc = (label_test == predictions).sum()/len(predictions)
    print('Accuracy : ', + acc)
    
    df = pd.DataFrame(data_test)
    df.insert(9, '9', label_test)
    df.insert(10, '10', predictions)
    df.to_csv('data/shuttle/KNN_4_shut_test_true_predict.csv', index=False, 
              header=False)
    print('Task Complete')


if __name__ == "__main__":
    # uncomment whichever you want to run
    # note: shuttle dataset takes too much time
    test_iris()
    # test_satellite()
    # test_shuttle()




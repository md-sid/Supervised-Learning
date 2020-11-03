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


def main():
    # load the data
    dataset = pd.read_csv('iris.data', header=None)
    dataset = np.array(dataset)
    data = dataset[:, 0:4]
    label = dataset[:, 4]
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, test_size=0.2, random_state=12)
    
    k = 3
    
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
    df.to_csv('KNN_test_true_predict.csv', index=False, header=False)

if __name__ == "__main__":
    main()





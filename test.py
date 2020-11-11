# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 12:36:22 2020

@author: SID
"""

from Performance import F_measure
import pandas as pd

result = pd.read_csv('data/KNN_2_iris_test_true_predict.csv', header=None)
F = F_measure(result, 4, 5)
print('Iris Dataset with k = 2 gives F : ', F)

result = pd.read_csv('data/KNN_3_iris_test_true_predict.csv', header=None)
F = F_measure(result, 4, 5)
print('Iris Dataset with k = 3 gives F : ', F)

result = pd.read_csv('data/KNN_4_iris_test_true_predict.csv', header=None)
F = F_measure(result, 4, 5)
print('Iris Dataset with k = 4 gives F : ', F)

result = pd.read_csv('data/KNN_5_iris_test_true_predict.csv', header=None)
F = F_measure(result, 4, 5)
print('Iris Dataset with k = 5 gives F : ', F)


result = pd.read_csv('data/KNN_6_iris_test_true_predict.csv', header=None)
F = F_measure(result, 4, 5)
print('Iris Dataset with k = 6 gives F : ', F)

result = pd.read_csv('data/satellite/KNN_2_sat_test_true_predict.csv', header=None)
F = F_measure(result, 36, 37)
print('Satellite Dataset with k = 2 gives F : ', F)

result = pd.read_csv('data/satellite/KNN_3_sat_test_true_predict.csv', header=None)
F = F_measure(result, 36, 37)
print('Satellite Dataset with k = 3 gives F : ', F)

result = pd.read_csv('data/satellite/KNN_4_sat_test_true_predict.csv', header=None)
F = F_measure(result, 36, 37)
print('Satellite Dataset with k = 4 gives F : ', F)

result = pd.read_csv('data/satellite/KNN_5_sat_test_true_predict.csv', header=None)
F = F_measure(result, 36, 37)
print('Satellite Dataset with k = 5 gives F : ', F)

result = pd.read_csv('data/satellite/KNN_6_sat_test_true_predict.csv', header=None)
F = F_measure(result, 36, 37)
print('Satellite Dataset with k = 6 gives F : ', F)


result = pd.read_csv('data/shuttle/KNN_4_shut_test_true_predict.csv', header=None)
F = F_measure(result, 9, 10)
print('Shuttle Dataset with k = 4 gives F : ', F)



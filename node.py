import numpy as np
import pandas as pd


class Node():
    def __init__(self):
        self.left = np.nan
        self.right = np.nan
        self.isleafnode = False
        self.attribute = np.nan
        self.attribute_split_value = np.nan
        self.label = np.nan
        self.score = np.nan
        self.no_of_data_points = np.nan

    def createinternalndoesplit(self, split_attribute, split_point, best_score, ny, nn):
        self.isleafnode = False
        self.attribute = split_attribute
        self.attribute_split_value = split_point
        self.score = best_score
        self.no_of_data_points = nn + ny
        self.left = Node()
        self.left.no_of_data_points = ny
        self.right = Node()
        self.right.no_of_data_points = nn
        return self.left, self.right

    def createleafnode(self, label, n):
        self.isleafnode = True
        self.label = label
        self.no_of_data_points = n

    def predictlabel(self, data):
        search_node = self
        while search_node.isleafnode is False:
            if data[search_node.attribute] < search_node.attribute_split_value:
                search_node = search_node.left
            else:
                search_node = search_node.right
        label = search_node.label
        return label

    def predict_data_set(self, data):
        result_array = []
        for index, row in data.iterrows():
            label = self.predictlabel(row)
            result_array.append(label)
        data['result'] = np.asarray(result_array)
        return data

def main():
    pass


if __name__ == '__main__':
    pass

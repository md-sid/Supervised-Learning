import numpy as np
import pandas as pd

def accuracy(result, gTruthCol, predCol):
    print("Starting accuracy calc")
    acc = 1 - errorRate(result, gTruthCol, predCol)
    print("Accuracy = "+str(acc))
    return(acc)

def errorRate(result, gTruthCol, predCol):
    print("Starting error rate calc")
    n = len(result.index)       # Number of entries

    comp = result.iloc[:, gTruthCol].eq(result.iloc[:, predCol])    # Comparison
    n_TF = comp.value_counts()      # Count of True/ False

    errRate = n_TF[False]/n         #
    print("Error Rate = "+str(errRate))
    return errRate


def contingencyTable(result, gTruthCol, predCol):
    print("Contingency table")

    tlist = result.iloc[:, gTruthCol].unique()  # Truth label list
    llist = result.iloc[:, predCol].unique()    # Predicted label list

    cT = np.zeros((len(tlist), len(tlist)))

    count = result[result.columns[gTruthCol: predCol+1]].value_counts(sort=False)

    for i in range(len(count.index)):
        idx = count.index[i]
        # print("Truth = "+str(idx[0])+", Predicted ="+ str(idx[1]))
        cT[idx[1], idx[0]] = count[count.index[i]]
        # print(count.index[i])
        # print(count[count.index[i]])
    print(cT)
    return cT

def precision(result, gTruthCol, predCol):
    print("Starting Precision calc")
    cT = contingencyTable(result, gTruthCol, predCol)

    with np.errstate(divide='ignore', invalid='ignore'):
        prec_i = cT.diagonal()/cT.sum(axis=1)     # Maybe 0/0 check
    prec_i[np.isnan(prec_i)] = 0
    print("Class-specific precision")
    print(prec_i)

    prec = cT.diagonal().sum()/len(result.index)
    print("Overall precision")
    print(prec)
    return prec_i

def recall(result, gTruthCol, predCol):
    print("Starting Recall calc")
    cT = contingencyTable(result, gTruthCol, predCol)

    recall_i = cT.diagonal() / cT.sum(axis=0)  # Maybe 0/0 check
    print("Class-specific recall")
    print(recall_i)

    return recall_i


def F_measure(result, gTruthCol, predCol):
    print("Starting F measure calc")
    cT = contingencyTable(result, gTruthCol, predCol)

    m_i = cT.sum(axis=1)
    n_i = cT.sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        F_i = 2 * cT.diagonal() / (n_i + m_i)
    F_i[np.isnan(F_i)] = 0
    print("Class-specific F measure")
    print(F_i)

    F = F_i.sum()/len(F_i)
    print("Total F measure")
    print(F)

    return(F)



def main():
    result = pd.read_csv("sysnthetic.data.result.k_means.csv")
    # print(result.describe())
    gTruthCol = 2   # Column index of the ground truth
    predCol = 3     # Column index of the prediction

    # errR = errorRate(result, gTruthCol, predCol)
    # acc = accuracy(result, gTruthCol, predCol)

    # prec_i  = precision(result, gTruthCol, predCol)
    # rec_i = recall(result, gTruthCol, predCol)

    F_i = F_measure(result, gTruthCol, predCol)


if __name__ == "__main__":
    main()
# -*- encoding: utf-8 -*-

# 2017-7-27 by xuanyuan14
# The template for calculating Kappa coefficient and Fleiss Kappa coefficient
# Diversity 0.0~0.20 very low consistency (slight), 0.21~0.40 general consistency (fair), 0.41~0.60 medium consistency (moderate)
# 0.61~0.80 is highly consistent (substantial) and 0.81~1 is almost completely consistent (almost perfect)

import numpy as np


def kappa(testData, k):  # testData represents the data to be calculated, k represents the data matrix is k*k
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i] * 1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    # xsum is a vector of k rows and 1 column, and ysum is a vector of 1 row and k columns
    Pe = float(ysum * xsum) / k ** 2
    P0 = float(P0 / k * 1.0)
    cohens_coefficient = float((P0 - Pe) / (1 - Pe))
    return cohens_coefficient


def fleiss_kappa(testData, N, k, n):  # testData represents the data to be calculated, (N, k) represents the shape of the matrix, indicating that the data is N rows and j columns, and there are a total of n labellers
    dataMat = np.mat(testData, float)
    oneMat = np.ones((k, 1))
    sum = 0.0
    P0 = 0.0
    for i in range(N):
        temp = 0.0
        for j in range(k):
            sum += dataMat[i, j]
            temp += 1.0 * dataMat[i, j] ** 2
        temp -= n
        temp /= (n ) * n
        P0 += temp
    P0 = 1.0 * P0 / N
    ysum = np.sum(dataMat, axis=0)
    for i in range(k):
        ysum[0, i] = (ysum[0, i] / sum) ** 2
    Pe = ysum * oneMat * 1.0
    ans = (P0 - Pe) / (1 - Pe)
    return ans[0, 0]


if __name__ == "__main__":
    dataArr1 = [[54, 43, 103], [85, 47, 68], [49, 39, 112], [94, 43, 63]]
    dataArr2 = [[0, 0, 0, 0, 14],
                [0, 2, 6, 4, 2],                                                                                                        
                [0, 0, 3, 5, 6],
                [0, 3, 9, 2, 0],
                [2, 2, 8, 1, 1],
                [7, 7, 0, 0, 0],
                [3, 2, 6, 3, 0],
                [2, 5, 3, 2, 2],
                [6, 5, 2, 1, 0],
                [0, 2, 2, 3, 7]]
    res2 = fleiss_kappa(dataArr1, 4, 3, 200)
    print(res2)


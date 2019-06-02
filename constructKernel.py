import numpy as np


def constructKernel(fea_a,fea_b,options):
    options = [];
    t = 1  
    num = fea_a.shape[0]  
    D = np.zeros((num,num))

    for i in range(num):
        for j in range(num):
            D[i,j] = np.power(np.linalg.norm(np.subtract(fea_a[i],fea_a[j])),2)
    K = np.exp(-D/(2*t**2))

    return K

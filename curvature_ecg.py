
import numpy as np
# import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib pt
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft,fftfreq
from scipy.spatial.distance import pdist, squareform
from teaspoon.SP.tsa_tools import takens
from numba import jit
from tqdm import tqdm
import time

def k_neighbors(data, K):
    """ K近邻算法，用于选取局部点云
        @param data: 数据坐标矩阵，默认一行代表一个样本
        @param K: 选取K个近邻
        @return: list, 长度为样本个数（即data的行数）
                第n个元素为k个最靠近第n个样本的样本下标
    """
    rslt_list = []
    dist_matrix = squareform(pdist(data, metric='euclidean'))       # 求距离矩阵
    for idx, line in enumerate(dist_matrix):                        # 第k行即为第k个数据点到其他点的距离
        k_n_idx = sorted(range(len(line)), key=lambda k:line[k])    # 按下标排序
        k_n_idx = k_n_idx[:K]                                       # 选择最靠近的k个，注意此处需排除自身
        rslt_list.append(k_n_idx)
    return rslt_list

def base_coefficient(signal,n,dim=3):
    N=len(signal)
    # ts_max=np.max(signal)
    # ts_min=np.min(signal)
    # signal=(signal-ts_min)/(ts_max-ts_min)
    # signal=(signal-ts_min)/10
    window_length=n
    window_num=N-n
    sig_seg=np.zeros([window_num,window_length])
    coefficient_matrix= np.zeros([window_num,dim])
    for i in range(window_num):
        sig_seg[i:]=signal[i:i+n]
        sig_seg_fft=fft(sig_seg[i,:])
        sig_seg_fft_real=2*sig_seg_fft.real/len(sig_seg_fft)
        sig_seg_fft_imag=2*sig_seg_fft.imag/len(sig_seg_fft)
        coefficient_matrix[i,0]= sig_seg_fft_real[0]
        coefficient_matrix[i,1]= sig_seg_fft_real[1] 
        coefficient_matrix[i,2]= sig_seg_fft_imag[1]
    return coefficient_matrix

def run_base_coefficient(data, n=10, dim=3):
    """ 将数据点正定化
        @param data: 数据坐标矩阵，默认一行代表一个样本
        @param n, tau: takens参数
        @return: np.ndarray, 正定化的数据点
    """
    return [base_coefficient(_, n=n, dim=dim) for _ in data]

def curvaturecurve(data, data_neighbor_idx_list):
    """ 求解某一原始数据点对应正定阵的曲率
        @param data: 数据坐标矩阵，默认一行代表一个样本
        @param data_neighbor_idx_list: 包含邻域信息的列表
        @return: list, 求得的曲率
    """       
    curve = []
    for i in range(len(data)):
        cache = data[data_neighbor_idx_list[i]]
        u = np.mean(cache,axis=0)
        std = cache - u                                             
        m,n = std.shape
        sigma = np.sum(std.reshape(m,n,1)@std.reshape(m,1,n), axis=0) / (len(data_neighbor_idx_list)-1)   # magic
        A = sigma
        eigenvalue,_ =np.linalg.eig(A)
        U = np.zeros_like(A)
        for y in range(len(eigenvalue)):
            for x in range(len(eigenvalue)):
                if y > x:
                    U[x,y] = 1.0/(eigenvalue[x]+eigenvalue[y])
        gamma=np.diag(eigenvalue)
        uu = U+U.T
        # curvature = 3*np.sum(np.diagonal(np.asmatrix(U)*np.asmatrix(gamma)*(np.asmatrix(U)+np.asmatrix(U).T)\
        #     +(np.asmatrix(U)+np.asmatrix(U).T)*np.asmatrix(gamma)*np.asmatrix(U) +(np.asmatrix(U)+np.asmatrix(U).T)\
        #         *np.asmatrix(gamma)*np.asmatrix(U)*np.asmatrix(gamma)*(np.asmatrix(U)+np.asmatrix(U).T)))
        curvature = 3*np.sum(np.diagonal(U@gamma@uu + uu@gamma@U + uu@gamma@U@gamma@uu))
        curve.append(curvature)
    return curve

# 

def count_curves(curves, width=1, begin=0, end=200):
    """ 计算某一数据点不同区间的曲率数
        @param curves: @func curvaturecurve返回值
        @param width: 区间宽度
        @param begin: 区间起始位置
        @param end:
        @return: dict, 包括不同的区间及分布情况（左闭右开，key为左区间）
    """
    c = np.array(curves)
    rslt = {}
    leftbound = begin
    while leftbound < end:
        rightbound = leftbound + width
        rslt[f"{leftbound}"] = len(np.where(c<rightbound)[0]) \
            - len(np.where(c<rightbound-width)[0])
        leftbound += width
    return rslt

def map2vec(curve_counts, curvaturecurvelist,threshold=40, round=2):
    """ 将曲率直方图映射为二维向量
        @param curve_counts: @func count_curves返回值
        @param threshold: 小于阈值时系数乘5
        @param round: 保留小数位数，default=2
        @return: tuple, 第一维为横向离散度，第二维为纵向离散度
    """
    # x = np.array(list(curve_counts.keys()),dtype='int')
    # Yme = 0.5 * np.sum(y)
    # sumY = np.array([np.sum(y[:k+1]) for k in range(len(y))])
    # j = np.where(Yme > sumY)[0][-1] if np.where(Yme > sumY)[0].any() else -1
    # X_dicret = x[j+1] + (x[1]-x[0])/2
    
    curvaturecurvelist_sort=np.array(sorted(curvaturecurvelist))
    length=np.where(curvaturecurvelist_sort<=200)[0][-1]
    X_dicret=curvaturecurvelist_sort[length//2]
    
    y = np.array(list(curve_counts.values()))
    Ynz_idx = np.where(y>0)
    u = np.sum(y[Ynz_idx]) / len(Ynz_idx[0])
    Y_dicret = 1/(len(y)-1) * (np.linalg.norm(y-u) ** 2)

    # print(y)
    # print(Yme)
    # print(sumY)
    # print(j, X_dicret, Y_dicret)

    return np.round(X_dicret, round), np.round(Y_dicret, round)

k = 20
n_samples = 5000
# 109398
# 112599
data_points = np.load('ecg.npy')[0:n_samples+1]
tags = np.load('label.npy')[1:n_samples+1]

print("Running takens...")
d_takened_list_new = run_base_coefficient(data_points)                                

print("Running k_neighbors...")
d_takened_neighbors_list_new = []
for _ in tqdm(d_takened_list_new, desc="k_neighbors"):
    d_takened_neighbors_list_new.append(k_neighbors(_, k))

print("Calculating curves...")
curve_list_new = []
for i in tqdm(range(len(d_takened_list_new)), desc="curvaturecurve"):
    curve_list_new.append(curvaturecurve(d_takened_list_new[i], d_takened_neighbors_list_new[i]))

print("Mapping curves to 2d vector...")
result_list_new = []
for n in tqdm(range(n_samples), desc="map2vec"):
    result_list_new.append(map2vec(count_curves(curve_list_new[n]),curve_list_new[n]))

filename = time.strftime("%Y_%m_%d_%H_%M", time.localtime()) + '.csv'
print(f"Writing result to {filename}...")
result_pd = pd.DataFrame(np.array(result_list_new), columns=['X', 'Y'])
result_pd.to_csv(filename)
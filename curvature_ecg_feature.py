
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

    rslt_list = []
    dist_matrix = squareform(pdist(data, metric='euclidean'))      
    for idx, line in enumerate(dist_matrix):                        
        k_n_idx = sorted(range(len(line)), key=lambda k:line[k])    
        k_n_idx = k_n_idx[:K]                                       
        rslt_list.append(k_n_idx)
    return rslt_list

def base_coefficient(signal,n,dim=3):
    N=len(signal)

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
 
    return [base_coefficient(_, n=n, dim=dim) for _ in data]

def curvaturecurve(data, data_neighbor_idx_list):
  
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

        curvature = 3*np.sum(np.diagonal(U@gamma@uu + uu@gamma@U + uu@gamma@U@gamma@uu))
        curve.append(curvature)
    return curve

# 

def count_curves(curves, width=1, begin=0, end=200):
   
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
   
    
    curvaturecurvelist_sort=np.array(sorted(curvaturecurvelist))
    length=np.where(curvaturecurvelist_sort<=200)[0][-1]
    X_dicret=curvaturecurvelist_sort[length//2]
    
    y = np.array(list(curve_counts.values()))
    Ynz_idx = np.where(y>0)
    u = np.sum(y[Ynz_idx]) / len(Ynz_idx[0])
    Y_dicret = 1/(len(y)-1) * (np.linalg.norm(y-u) ** 2)



    return np.round(X_dicret, round), np.round(Y_dicret, round)

k = 20
n_samples = 5000

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
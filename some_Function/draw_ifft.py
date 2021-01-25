import numpy as np
import matplotlib.pyplot as plt
import javabin
import os
filepath = r'D:\Na2Ti2AsO\LHe2\LHe_atomic_phase\794-847\rawdata_823_sxm\Z_forward.bin'
file_folder,dirname = os.path.split(filepath)
layer_topo = javabin.read_java_bin_single_layer(filepath)

len_x,len_y = np.shape(layer_topo.data)

PointList = np.array([[220,156],[179,244],[243,221],[157,178]])
fft_data = np.fft.fftshift(np.fft.fft2(layer_topo.data))
fft_choose = np.zeros((len_x,len_y),dtype=complex)
fft_choose_1 = np.zeros((len_x,len_y),dtype=complex)

length_we_choose = 4
for i in range(2):
    for m in range(length_we_choose):
        for n in range(length_we_choose):
            fft_choose[n+PointList[i][1]][m+PointList[i][0]]=fft_data[n+PointList[i][1]][m+PointList[i][0]]
            fft_choose[n-PointList[i][1]][m-PointList[i][0]]=fft_data[n-PointList[i][1]][m-PointList[i][0]]
            fft_choose_1[n + PointList[i+2][1]][m + PointList[i+2][0]] = fft_data[n + PointList[i+2][1]][m + PointList[i+2][0]]
            fft_choose_1[n - PointList[i+2][1]][m - PointList[i+2][0]] = fft_data[n - PointList[i+2][1]][m - PointList[i+2][0]]

#fft_data = np.log10(np.abs(np.fft.fftshift(np.fft.fft2())))
index_x,index_y = np.meshgrid(layer_topo.new_x,layer_topo.new_y)
fft_choose_return = np.fft.fftshift(fft_choose)
fft_choose_return_1 = np.fft.fftshift(fft_choose_1)
ifft_topo = np.real(np.fft.ifft2(fft_choose_return))
ifft_topo_phase = np.angle(np.fft.ifft2(fft_choose_return))
ifft_topo_1 = np.real(np.fft.ifft2(fft_choose_return_1))
ifft_topo_phase_1 = np.angle(np.fft.ifft2(fft_choose_return_1))
fig,ax=plt.subplots(figsize=(10,10))
ax.pcolormesh(ifft_topo_phase,cmap=plt.cm.RdBu)
plt.show()
fig,ax=plt.subplots(figsize=(10,10))
ax.pcolormesh(ifft_topo_phase_1,cmap=plt.cm.RdBu)
plt.show()

"""
fig,ax=plt.subplots(figsize=(10,10))
ax.pcolormesh(np.log10(np.abs(fft_data)),cmap=plt.cm.RdBu)
plt.show()
fig,ax=plt.subplots(figsize=(10,10))
ax.pcolormesh(index_x,index_y,layer_topo.data,cmap=plt.cm.RdBu)
plt.show()
fig,ax=plt.subplots(figsize=(10,10))
ax.pcolormesh(np.log10(np.abs(fft_choose)),cmap=plt.cm.RdBu)
plt.show()
fig,ax=plt.subplots(figsize=(10,10))
ax.pcolormesh(index_x,index_y,ifft_topo,cmap=plt.cm.RdBu)
plt.show()
"""

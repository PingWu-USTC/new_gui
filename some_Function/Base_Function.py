import numpy as np
import matplotlib.pyplot as plt
import javabin
import layers

def fast_fourier_transform(layer_data:layers.singlelayer):
    fft_layer =layers.singlelayer(np.fft.fft2(layer_data.data),layer_data.nx,layer_data.ny,layer_data.new_x,layer_data.new_y,layer_data.bias,layer_data.current)
    fft_logmag = layers.singlelayer(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(layer_data.data)))),layer_data.nx,layer_data.ny,layer_data.new_x,layer_data.new_y,layer_data.bias,layer_data.current)
    return fft_layer,fft_logmag
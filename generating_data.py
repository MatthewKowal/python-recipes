# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:21:20 2024

@author: user1
"""


'''####################################
   ___                          _       
  / _ \___ _ __   ___ _ __ __ _| |_ ___ 
 / /_\/ _ \ '_ \ / _ \ '__/ _` | __/ _ \
/ /_\\  __/ | | |  __/ | | (_| | ||  __/
\____/\___|_| |_|\___|_|  \__,_|\__\___|
                                        
     _     _       _       _            
    / | __| |   __| | __ _| |_ __ _     
    | |/ _` |  / _` |/ _` | __/ _` |    
    | | (_| | | (_| | (_| | || (_| |    
    |_|\__,_|  \__,_|\__,_|\__\__,_|    
                                      
#######################################'''


# make a straight line with noise

import numpy as np
import matplotlib.pyplot as plt

#make an array with linearly spaced values. this will be the x axis
X = np.linspace(0,31,32)
print(X)

# now, build a formula for a line and feed the x-axis and the parameters into it
m = 1
b = 2
y = m*X + b
#then plot it
plt.scatter(X, y, color='black')

#add some normally distributed noise to the line, and plot it
noise = 5
y_n = y + noise * np.random.randn(*X.shape)
plt.scatter(X, y_n, color='red')
plt.show()


#%%

# make a gaussian function with noise

x_dim = 32
X = np.linspace(0,x_dim-1,x_dim)
print(X)

amp = -0.01
mean = x_dim/2
std = 5
offset = 1
y = offset + amp * np.exp( -((X-mean)/std)**2)
plt.scatter(X, y, color='black')

#add noise
noise = 0.002
y_n = y + noise * np.random.randn(*X.shape)
plt.scatter(X, y_n, color='red')
plt.show()


#%%

#make a (pporly) simulated Raman spectrum

import numpy as np
import matplotlib.pyplot as plt

#lets make a function that returns some normally distributed data with noise
def gendata_1dgauss(x_dim, amp, mean, std, offset, noise):
    X = np.linspace(0,x_dim-1,x_dim)
    y = offset + amp * np.exp( -((X-mean)/std)**2)
    #add noise
    y_n = y + noise * np.random.randn(*X.shape)
    #plot
    plot=False
    if plot:
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.figure(dpi=150)
        #plt.scatter(X, y, color='black', s=4)
        plt.scatter(X, y_n, color='red', s=1)
        plt.show()
    return y_n

data = gendata_1dgauss(2000, 1, 1000, 50, 0, 0.01)
#plt.plot(data)



#now lets simulate Raman data by overlaying a few gaussians with random parameters
#then add noise and add a large, wide backgroun fluoresence signal with another gaussian
def gendata_raman(num_peaks, plot=True):
    x_dim = 2000
    X = np.linspace(0,x_dim-1,x_dim)
    y = np.zeros(*X.shape)
    for i in range(num_peaks):    
        amp = np.random.rand()
        mean = np.random.randint(0,x_dim)
        std = np.random.rand()*40
        y += gendata_1dgauss(x_dim, amp, mean, std, offset=0, noise=0)
    #add noise
    noise = 0.01
    y += noise * np.random.randn(*X.shape)
    #add fluoresence
    amp = np.random.rand()*3
    mean = np.random.randint(0,x_dim/2)
    std = np.random.rand()*40+1000
    y += gendata_1dgauss(x_dim, amp, mean, std, offset=0, noise=0)
    #plot
    if plot:
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.figure(dpi=150)
        plt.scatter(X, y, color='red', s=1)
        plt.show()
    
    return y
        
num_peaks = 5
raman_spec = gendata_raman(num_peaks)



#%%


'''
   ___                          _       
  / _ \___ _ __   ___ _ __ __ _| |_ ___ 
 / /_\/ _ \ '_ \ / _ \ '__/ _` | __/ _ \
/ /_\\  __/ | | |  __/ | | (_| | ||  __/
\____/\___|_| |_|\___|_|  \__,_|\__\___|
                                        
     ____     _       _       _         
    |___ \ __| |   __| | __ _| |_ __ _  
      __) / _` |  / _` |/ _` | __/ _` | 
     / __/ (_| | | (_| | (_| | || (_| | 
    |_____\__,_|  \__,_|\__,_|\__\__,_| 
                                       

'''



import numpy as np
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import cv2

# from scipy import optimize
# import matplotlib.cm as cm
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# import math

#this function will simulate and return an iscat point spread function (scattering signal)
#please supply the square image dimension, the gaussian amplitude, and standard deviation (width)
def generate_psf(dim=30, amp=0.005, std=10, noise=0.002, plot=True):
    # The two-dimensional domain of the fit.
    xmin, xmax, nx = 0, dim-1, dim
    ymin, ymax, ny = 0, dim-1, dim
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # make gaussian
    #           x0, y0, xalpha, yalpha,  A
    initguess = (dim/2, dim/2,   std,    std,   amp, 1)
    Z = np.zeros(X.shape)
    #Z += gaussian(X, Y, *initguess)
    mean = dim/2
    offset = 1
    Z += offset + amp * np.exp( -((X-mean)/std)**2 -((Y-mean)/std)**2)

    #add noise
    Z += noise * np.random.randn(*Z.shape)
    
    if plot:
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.figure(dpi=150)
        plot3d(Z)
    return Z

def plot3d(Z, title=""):
    ''' make a neat 3d plot '''
    x=Z.shape[0]
    y=Z.shape[1]
    xmin, xmax, nx = 0, x-1, x
    ymin, ymax, ny = 0, y-1, y
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    
    # Plot the 3D figure of the fitted function and the residuals.
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.figure(dpi=300)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')#, alpha = 0.8)
    zlim = (np.max(Z) - np.min(Z))
    #ax.set_zlim(1-1.2*zlim, 1+zlim)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_zticks([np.min(Z), 1, np.max(Z)])
    ax.view_init(elev=10, azim=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()



dim=30
amp=-0.01
std=5
noise=0.001
generate_psf(dim, amp, std, noise)


#%%


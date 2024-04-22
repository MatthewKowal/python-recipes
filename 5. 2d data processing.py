# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:36:58 2024

@author: user1
"""


# #this lesson covers 2d data manipulation. we will need from data to work with
# #so we will import some functions from the "generating_data" script.
# import os
# import sys
# folder, filename = os.path.split(__file__)  # get folder and filename of this script
# #modfolder = os.path.join(folder)            # I cant remember why this is here
# sys.path.insert(0, folder)               # add the current folder to the system path
# import generating_data as generating_data

import numpy as np
import matplotlib.pyplot as plt

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
    ax.view_init(elev=10, azim=20)
    plt.title(title)
    plt.tight_layout()
    plt.show()

#make a noisy PSF image
image = generate_psf(dim=50, amp=0.05, std=6, noise=0.002, plot=False)
plot3d(image)



#gaussian blur an image
import cv2
import matplotlib.pyplot as plt
size = 9
std  = 3
blurred_image = cv2.GaussianBlur(image, (size,size), std)
plot3d(blurred_image)





#%%

#fitting a 2d gaussian to some data

from scipy import optimize
import numpy as np

#this is a lambda function which defines the 2d gaussian function
def gaussian(height, center_x, center_y, width_x, width_y, z_offset): #gaussian lamda function generator
    """Returns a gaussian function with the given parameters"""
    # width_x = float(width_x)
    # width_y = float(width_y)
    return lambda x,y: z_offset + height*np.exp(-(((center_x-x)/width_x)**2 + ((center_y-y)/width_x)**2)/2)

#this function finds best fit gaussian parameters for some experimental data
def fit_gaussian_parameters(data): #find optimized gaussian fit for a particle
    #make a good initial guess at the gaussian parameters
    height   = -0.1
    x        = data.shape[0]/2
    y        = data.shape[1]/2
    width_x  = 2.0
    width_y  = 2.0
    z_offset = 1.0
    params = height, x, y, width_x, width_y, z_offset
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p





#fit a gaussian to the data
paramsG       = fit_gaussian_parameters(image)
Xin, Yin      = np.mgrid[0:(image.shape[0]), 0:(image.shape[1])]         #emtpy grid to fit the parameters to. must be the same size as the particle iamge
fitG          = gaussian(*paramsG)(Xin, Yin)
plot3d(image, "raw data")
plot3d(fitG, "best fit gaussian")

#%%



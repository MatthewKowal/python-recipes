# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:36:58 2024

@author: user1
"""


#this lesson covers 2d data manipulation. we will need from data to work with
#so we will import some functions from the "generating_data" script.
import os
import sys
folder, filename = os.path.split(__file__)  # get folder and filename of this script
#modfolder = os.path.join(folder)            # I cant remember why this is here
sys.path.insert(0, folder)               # add the current folder to the system path
import generating_data as generating_data




#make a noisy PSF image
image = generating_data.generate_psf(dim=50, amp=0.05, std=6, noise=0.002, plot=False)
generating_data.plot3d(image)



#gaussian blur an image
import cv2
import matplotlib.pyplot as plt
size = 9
std  = 3
blurred_image = cv2.GaussianBlur(image, (size,size), std)
generating_data.plot3d(blurred_image)





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
generating_data.plot3d(image, "raw data")
generating_data.plot3d(fitG, "best fit gaussian")

#%%



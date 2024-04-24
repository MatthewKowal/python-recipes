# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:36:58 2024

@author: user1
"""

'''
# #this lesson covers 2d data manipulation.
# #we will need data to work with so lets first define some methods
# #that will generate data that looks like what we see with an iscat microscope
'''


import numpy as np
import matplotlib.pyplot as plt
import math


'''this is a nice 3d plotting function that makes an image look 3d'''
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
    return 1




''' GAUSSIAN 
#this function will simulates and return an iscat point spread function (scattering signal)
#modelled as a gaussian function
#please supply the square image dimension, the gaussian amplitude, and standard deviation (width)
#this is a lambda function which defines the 2d gaussian function '''

def gaussian(height, center_x, center_y, width_x, width_y, z_offset): #gaussian lamda function generator
    """Returns a gaussian function with the given parameters"""
    # width_x = float(width_x)
    # width_y = float(width_y)
    return lambda x,y: z_offset + height*np.exp(-(((center_x-x)/width_x)**2 + ((center_y-y)/width_x)**2)/2)

def generate_psf(noise=0.002, dim=30, amp=-0.03, sig=6, offset=1, plot=False):
    # The two-dimensional domain of the fit.
    xmin, xmax, nx = 0, dim-1, dim
    ymin, ymax, ny = 0, dim-1, dim
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # make gaussian
    #           x0, y0, xalpha, yalpha,  A
    #initguess = (dim/2, dim/2,   sig,    std,   amp, 1)
    Z = np.zeros(X.shape)
    #Z += gaussian(X, Y, *initguess)
    mean = dim/2
    #offset = 1
    #Z += offset + amp * np.exp( -((X-mean)/sig)**2 -((Y-mean)/std)**2)
    params = [amp, dim/2, dim/2, sig, sig, offset]
    Z += gaussian(*params)(X,Y)
    #add noise
    Z += noise * np.random.randn(*Z.shape)
    
    if plot:
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.figure(dpi=150)
        plot3d(Z)
    return Z






'''Laplacian of Gaussian
#this is a more accurate representation of a point spread function than a gaussian '''

def LoG(height, center_x, center_y, sigma):
    ox = lambda x: center_x - x
    oy = lambda y: center_y - y
    return lambda x, y: -(height*1000)/(math.pi*sigma**4)*(1-((ox(x)**2+oy(y)**2)/(2*sigma**2)))*np.exp(-((ox(x)**2+oy(y)**2)/(2*sigma**2)))+1

def generate_log(noise=0.002, dim=30, ampmin=0.005, ampmax=0.01, sig=4, plot=False):
    
    xmin, xmax, nx = 0, dim-1, dim
    ymin, ymax, ny = 0, dim-1, dim
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    amp = ampmin + np.random.rand()*ampmax

    Z = LoG(amp,dim/2,dim/2, sig)(X,Y)

    #add noise
    Z += noise * np.random.randn(*Z.shape)
    
    return Z




'''Difference of Gaussian
# generate a point spread function based on the shape of the difference of two gaussian '''
def DoG(height1, height2, center_x, center_y, sigma1, sigma2):
    #throw away height2, using it causes the optimize function to run past the max number of iterations its willing to. using the same height for each gaussian seems to work well anyway
    ox = lambda x: center_x - x
    oy = lambda y: center_y - y
    return lambda x, y: -2*height1*np.exp(-((ox(x)/sigma1)**2 + ((oy(y))/sigma1)**2)/2) + height1*np.exp(-((ox(x)/sigma2)**2 + ((oy(y))/sigma2)**2)/2) + 1

def generate_dog(noise=0.002, dim=30, amp1=0.02, amp2=0.01, sig1 = 4, sig2 = 6):
    
    xmin, xmax, nx = 0, dim-1, dim
    ymin, ymax, ny = 0, dim-1, dim
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    #   height1, height2, center_x, center_y, sigma1, sigma2
    # height1  = 0.02
    # height2  = 0.01
    # center_x = dim/2
    # center_y = dim/2
    # sigma1   = 4
    # sigma2   = 6
    # params = [height1, height2, center_x, center_y, sigma1, sigma2]
    params = [amp1, amp2, dim/2, dim/2, sig1, sig2]
    Z = DoG(*params)(X,Y)
    
    #add noise
    #noise = 0.002
    Z += noise * np.random.randn(*Z.shape)
    return Z





#make a noisy PSF image
#use gaussian method
image = generate_psf()
plot3d(image, title="Gaussian")

#use difference of gaussian method
img = generate_dog()
plot3d(img, title="Dif. of Gauss")

#use laplacian of gaussian method
img= generate_log()
plot3d(img, title="Laplacian of Gauss")






#%%
''' data generation part II: GENERATE A POORLY SIMULATED ISCAT IMAGE ''' 

from PIL import Image

def generate_iscat_image(imgxy, n, method):
    bbox    = 15
    pdim    = 30
    pstd    = 6
    pampmax = -0.01
    
    image = np.ones([imgxy,imgxy])
    
    pl = []
    
    #add each particle
    for i in range(n):
        pamp = np.random.rand()*pampmax
        if method=='gauss': psf = generate_psf(noise=0)
        if method=='log': psf = generate_log(noise=0)-1
        if method=='dog': psf = generate_dog(noise=0)-1
        
        x = bbox + np.random.randint(imgxy-3*bbox)
        # print(bbox, x-bbox)
        y = bbox + np.random.randint(imgxy-3*bbox)
        #plot3d(psf)
        # print(bbox, y-bbox)
        # print(i, "amp:", pamp)
        # print(i, "x:  ", x)
        # print(i, "y:  ", y, "\n")
        image[y:(y+(2*bbox)), x:(x+(2*bbox))] += psf
        pl.append([x,y])
    
    #add noise
    noise=0.002
    image += noise * np.random.randn(*image.shape)
    
    return image, pl

image, pl = generate_iscat_image(256,5, 'log')
plt.imshow(image)
plt.colorbar()
plt.show()
# plot3d(image)
print(pl)

#%%
''' DATA MANIPULATION '''

#now that we have some 2d data we can work with, we can try some manipulations


import cv2
import matplotlib.pyplot as plt

image = generate_log()
plot3d(image)

'''gaussian blur an image'''
size = 9
std  = 3
blurred_image = cv2.GaussianBlur(image, (size,size), std)
plot3d(blurred_image)
# plt.imshow(blurred_image)
# plt.show()





#%%

''' FITTING DATA ''' 

from scipy import optimize
import numpy as np



''' GAUSSIAN '''

#this function finds best fit gaussian parameters for some experimental data
def fit_gaussian_parameters(data): #find optimized gaussian fit for a particle
    #this is a lambda function which defines the 2d gaussian function
    # def gaussian(height, center_x, center_y, width_x, width_y, z_offset): #gaussian lamda function generator
    #     """Returns a gaussian function with the given parameters"""
    #     # width_x = float(width_x)
    #     # width_y = float(width_y)
    #     return lambda x,y: z_offset + height*np.exp(-(((center_x-x)/width_x)**2 + ((center_y-y)/width_x)**2)/2)

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


def fitLoGaussian(data, pimage_dim): #find optimized gaussian fit for a particle
    #first guess parameters
    height   = 0.7
    center_x = pimage_dim
    center_y = pimage_dim
    sigma    = 1.0
    params = height, center_x, center_y, sigma
    errorfunction = lambda p: np.ravel(LoG(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def fitDoGaussian(data, pimage_dim): #find optimized gaussian fit for a particle
    #first guess parameters
    height1   = 0.25#0.3
    height2   = 0.2#0.4
    center_x = pimage_dim
    center_y = pimage_dim
    sigma1   = 3.0#4.2
    sigma2   = 5.0#5.7
    
    params = height1, height2, center_x, center_y, sigma1, sigma2
    errorfunction = lambda p: np.ravel(DoG(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p



#generate an image using one of the methods
image = generate_psf()
image = generate_log()
image = generate_dog()



#fit a Gaussian to the data
paramsG       = fit_gaussian_parameters(image)
Xin, Yin      = np.mgrid[0:(image.shape[0]), 0:(image.shape[1])]         #emtpy grid to fit the parameters to. must be the same size as the particle iamge
fitG          = gaussian(*paramsG)(Xin, Yin)
plot3d(image, "raw data")
plot3d(fitG, "best fit gaussian")

#fit a Laplacian of Gaussian to the data
paramsLoG       = fitLoGaussian(image, 15)
Xin, Yin        = np.mgrid[0:(image.shape[0]), 0:(image.shape[1])]         #emtpy grid to fit the parameters to. must be the same size as the particle iamge
fitLoG          = LoG(*paramsLoG)(Xin, Yin)
plot3d(image, "raw data")
plot3d(fitLoG, "best fit Laplacian of gaussian")

#fit a Difference of Gaussian to the data
paramsDoG       = fitDoGaussian(image, 15)
Xin, Yin        = np.mgrid[0:(image.shape[0]), 0:(image.shape[1])]         #emtpy grid to fit the parameters to. must be the same size as the particle iamge
fitDoG          = DoG(*paramsDoG)(Xin, Yin)
plot3d(image, "raw data")
plot3d(fitDoG, "best fit Difference of gaussian")



#%%





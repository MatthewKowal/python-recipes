# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:36:58 2024

@author: user1
"""



#fitting functions to data

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Our function to fit is going to be a sum of two-dimensional Gaussians
def gaussian(x, y, x0, y0, xalpha, yalpha, A, offset):
    return offset + A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2)

# A list of the Gaussian parameters:

def generate_psf(dim, amp, std, plot=False, add_noise=True,):
    # The two-dimensional domain of the fit.
    xmin, xmax, nx = 0, dim-1, dim
    ymin, ymax, ny = 0, dim-1, dim
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # make gaussian
    #           x0, y0, xalpha, yalpha,  A
    initguess = (dim/2, dim/2,   std,    std,   amp, 1)
    Z = np.zeros(X.shape)
    Z += gaussian(X, Y, *initguess)

    #add noise
    if add_noise:
        noise_sigma = 0.002
        Z += noise_sigma * np.random.randn(*Z.shape)
    
    if plot:
        plot3d(Z)
    return Z

def plot3d(Z):
    
    x=Z.shape[0]
    y=Z.shape[1]
    xmin, xmax, nx = 0, x-1, x
    ymin, ymax, ny = 0, y-1, y
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    
    # Plot the 3D figure of the fitted function and the residuals.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma', alpha = 0.5)
    zlim = (np.max(Z) - np.min(Z))
    #ax.set_zlim(1-1.2*zlim, 1+zlim)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_zticks([np.min(Z), 1, np.max(Z)])
    ax.view_init(elev=0, azim=23)
    plt.show()
    

image = generate_psf(dim=50, amp=-0.005, std=10, plot=True)

#try blurring the image
import cv2
size = 5
std  = 3
blurred_image = cv2.GaussianBlur(image, (size,size), std)

plot3d(blurred_image)

#%%
#use scipy.optimize.curve_fit to fit some data to a function

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#define a function
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

#generate some experimental data:
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
rng = np.random.default_rng()
y_noise = 0.2 * rng.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')


#Fit for the parameters a, b, c of the function func:
popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

#Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
plt.plot(xdata, func(xdata, *popt), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


#%%

# import matplotlib.cm as cm
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# import math

# def gaussian2(x, y, x0, y0, xalpha, yalpha, A, offset):
#     print(type(x), x.shape)
#     print(type(y), y.shape)
#     print(type(x0))
#     print(type(y0))
#     print(type(xalpha))
#     print(type(yalpha))
#     print(type(A))
#     print(type(offset))
#     return offset + A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2)


# import numpy as np
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import cv2

# from scipy import optimize
# import matplotlib.cm as cm
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# import math

# #this function will simulate and return an iscat point spread function (scattering signal)
# #please supply the square image dimension, the gaussian amplitude, and standard deviation (width)
# def generate_psf(dim=30, amp=0.005, std=10, noise=0.002, plot=False):
#     # def gaussian(x, y, x0, y0, xalpha, yalpha, A, offset):
#     #     print(type(x), x.shape)
#     #     print(type(y), y.shape)
#     #     print(type(x0))
#     #     print(type(y0))
#     #     print(type(xalpha))
#     #     print(type(yalpha))
#     #     print(type(A))
#     #     print(type(offset))
#     #     return offset + A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2)

#     # The two-dimensional domain of the fit.
#     xmin, xmax, nx = 0, dim-1, dim
#     ymin, ymax, ny = 0, dim-1, dim
#     x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
#     X, Y = np.meshgrid(x, y)

#     # make gaussian
#     #           x0, y0, xalpha, yalpha,  A
#     initguess = (dim/2, dim/2,   std,    std,   amp, 1)
#     Z = np.zeros(X.shape)
#     #Z += gaussian(X, Y, *initguess)
#     mean = dim/2
#     offset = 1
#     Z += offset + amp * np.exp( -((X-mean)/std)**2 -((Y-mean)/std)**2)

#     #add noise
#     Z += noise * np.random.randn(*Z.shape)
    
#     if plot:
#         plot3d(Z)
#     return Z

# def plot3d(Z):
    
#     x=Z.shape[0]
#     y=Z.shape[1]
#     xmin, xmax, nx = 0, x-1, x
#     ymin, ymax, ny = 0, y-1, y
#     x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
#     X, Y = np.meshgrid(x, y)
    
#     # Plot the 3D figure of the fitted function and the residuals.
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z, cmap='plasma')#, alpha = 0.8)
#     zlim = (np.max(Z) - np.min(Z))
#     #ax.set_zlim(1-1.2*zlim, 1+zlim)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     #ax.set_zticks([np.min(Z), 1, np.max(Z)])
#     ax.view_init(elev=10, azim=0)
#     plt.show()

















# def gaussian2(x, y, x0, y0, xalpha, yalpha, A, offset):
#     print(type(x), x.shape)
#     print(type(y), y.shape)
#     print(type(x0))
#     print(type(y0))
#     print(type(xalpha))
#     print(type(yalpha))
#     print(type(A))
#     print(type(offset))
#     return offset + A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2)



#%%
def remove_non_gaussian_particles(particle_list, pimage_dim, constants):
    
    particle_list_out = []
    new_pID = 1
    
    print("\nFitting: ", len(particle_list), " Particles to approximations of Zero Order Bessel Functions of the First Kind...")
    #print("pID  \t   drkpxl \t loc \t   len \t %  \t   RMSE  \t  peak contrast")
    for p in tqdm.tqdm(particle_list):  
        #print("working on particle: ", new_pID)
        darkest_frame = np.argmin(p.drkpxl_vec)
        darkest_image = p.pimage_vec[darkest_frame]
        
        #generate a space the same shape as the image to use for the fit
        Xin, Yin = np.mgrid[0:(2*pimage_dim), 0:(2*pimage_dim)]         #emtpy grid to fit the parameters to. must be the same size as the particle iamge
        
        #generate an optimized parameters for gaussian fit
        paramsG          = fitgaussian(darkest_image, pimage_dim)
        paramsLoG        = fitLoGaussian(darkest_image, pimage_dim)
        paramsDoG        = fitDoGaussian(darkest_image, pimage_dim)
        
        # Plot the fit function on a surface 
        fitG             = gaussian(*paramsG)(Xin, Yin)        
        fitLoG           = LoG(*paramsLoG)(Xin, Yin)        
        fitDoG           = DoG(*paramsDoG)(Xin, Yin)        
        
        # Calculate the PEAK CONTRAST based on the fit
        peak_contrastG   = 1 + paramsG[0]
        peak_contrastLoG = np.min(fitLoG)#1 + params[0]
        peak_contrastDoG = np.min(fitDoG)#1 + params[0]
        
        #calculate RMSE for the fit
        rmseG   = sqrt(mean_squared_error(normfit(darkest_image), normfit(fitG)))
        rmseLoG = sqrt(mean_squared_error(normfit(darkest_image), normfit(fitLoG)))
        rmseDoG = sqrt(mean_squared_error(normfit(darkest_image), normfit(fitDoG)))

        #update current particle
        p.pID              = new_pID
        p.rmseG            = rmseG
        p.rmseLoG          = rmseLoG
        p.rmseDoG          = rmseDoG
        p.peak_contrastG   = peak_contrastG
        p.peak_contrastLoG = peak_contrastLoG
        p.peak_contrastDoG = peak_contrastDoG
        p.paramsG          = paramsG
        p.paramsLoG        = paramsLoG
        p.paramsDoG        = paramsDoG
        
        
        ''' Remove Particles that do not fit the model well '''
        if 'DoG' in constants:
                
            if rmseDoG < constants['DoG']:
                #populate new particle list   
                particle_list_out.append(p)
                new_pID += 1
         
    particle_list_out = np.asarray(particle_list_out)
    
    return particle_list_out



def LoG(height, center_x, center_y, sigma):
    ox = lambda x: center_x - x
    oy = lambda y: center_y - y
    return lambda x, y: -(height*1000)/(math.pi*sigma**4)*(1-((ox(x)**2+oy(y)**2)/(2*sigma**2)))*np.exp(-((ox(x)**2+oy(y)**2)/(2*sigma**2)))+1


def DoG(height1, height2, center_x, center_y, sigma1, sigma2):
    #throw away height2, using it causes the optimize function to run past the max number of iterations its willing to. using the same height for each gaussian seems to work well anyway
    ox = lambda x: center_x - x
    oy = lambda y: center_y - y
    return lambda x, y: -2*height1*np.exp(-((ox(x)/sigma1)**2 + ((oy(y))/sigma1)**2)/2) + height1*np.exp(-((ox(x)/sigma2)**2 + ((oy(y))/sigma2)**2)/2) + 1




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

def normfit(image):
    return image/np.sum(np.square(image-1))








#%%


def save_3d_contour(data, fit, rmse, pID, functype, basepath, name, pimage_dim): #generate a plot for the particle and its gaussian fit
    
    #print("Making Particle Library Spreadsheets .csv file...") #create files
    if not os.path.exists(os.path.join(basepath, "output", "random sampling")): os.makedirs(os.path.join(basepath, "output", "random sampling"))
    image_filename  = os.path.join(basepath, "output", "random sampling", (name+"-pID"+str(pID)+functype+".png"))
    
    #x_dim, y_dim, x_steps, y_steps = (2*pimage_dim), (2*pimage_dim), (2*pimage_dim), (2*pimage_dim)
    fig = plt.figure()

    x, y = np.mgrid[0:(2*pimage_dim), 0:(2*pimage_dim)] #numpy.mgrid[-x_dim/2:x_dim/2:x_steps*1j, -y_dim/2:y_dim/2:y_steps*1j]
    v_min = np.min(fit)#0.7 #numpy.min(0)
    v_max = 1.05#np.max(data)#1.3 #numpy.max(255)

    ax = fig.gca(projection='3d')

    ax.contourf(x, y, data, zdir='z', levels=256, offset=v_min, cmap=cm.gray)
    
    #cset = ax.contourf(x, y, data, zdir='x', offset=-x_dim/2-1, cmap=cm.coolwarm)
    #cset = ax.contourf(x, y, data, zdir='y', offset=0, cmap=cm.coolwarm)

    ax.plot_wireframe(x, y, fit, rstride=5, cstride=5, alpha=0.5, color='blue', linewidth=1)
    ax.plot_surface(x, y, fit, rstride=2, cstride=2, alpha=0.3, cmap=cm.jet, linewidth=1)

    #ax.plot_wireframe(x, y, data, rstride=5, cstride=5, alpha=0.2, color='blue', linewidth=1)
    #ax.plot_surface(x, y, data, rstride=2, cstride=2, alpha=0.1, cmap=cm.jet, linewidth=1)


    ax.set_xlabel('X')
    #ax.set_xlim(-x_dim/2-1, x_dim/2+1)
    ax.set_ylabel('Y')
    #ax.set_ylim(-y_dim/2-1, y_dim/2+1)
    ax.set_zlabel('Z')
    #ax.set_zlim(v_min, v_max)
    
    ax.set_xlim([0,(2*pimage_dim)])
    ax.set_ylim([0,(2*pimage_dim)])
    ax.set_zlim([v_min,v_max])
    elev = 18
    azim = 127
    plt.gca().view_init(elev, azim)

    text_kwargs = dict(ha='left', va='center', fontsize=12, color='black')
    fig.text(0.18, 0.21, ("pID: " + str(pID)), **text_kwargs)
    fig.text(0.18, 0.18, ("RMSE: " + str(rmse)[:7]), **text_kwargs)
    fig.text(0.18, 0.15, ("F(x, y): " + functype), **text_kwargs)
    
    plt.savefig(image_filename)
    #plt.close()
    plt.show()
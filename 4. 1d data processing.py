# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:36:58 2024

@author: user1
"""

#this lesson covers 2d data manipulation. we will need from data to work with
#so we will import some functions from the "generating_data" script.
# import os
# import sys
# folder, filename = os.path.split(__file__)  # get folder and filename of this script
# #modfolder = os.path.join(folder)            # I cant remember why this is here
# sys.path.insert(0, folder)               # add the current folder to the system path
# import generating_data as generating_data
# ''' note: this runs the script to a ton of plots will come up '''

# spectrum = generating_data.gendata_raman(num_peaks=5) #simulate a raman spectrum



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
spectrum = gendata_raman(num_peaks)


#%%

import pywt #this is pywavelets. it is used to perform DWT calculations
import numpy as np
import matplotlib.pyplot as plt

def normalize(spectrum):
    #spectra -= np.min(spectra)
    sumofsquares = np.sum(np.square(spectrum)) #compute sum of squares
    newspec = spectrum/np.sqrt(sumofsquares)  #normalize to sum of squares
    return newspec

def lowpass(spectrum, level): 
    #first decompose to spectra to the specified level.
    coeffs = pywt.wavedec(spectrum, 'sym5', level=level) #tranform to wavelets
    #remove all high frequency levels
    for i in range(1,(level+1)):
        coeffs[i] = np.zeros_like(coeffs[i])
        #print(i)
    #reconstruct the wavelets to spectra space
    newspec = pywt.waverec(coeffs, 'sym5')

    return newspec
    
def remove_baseline(spectrum, level, iterations):
    '''    Reference:
        C. M. Galloway, E. C. Le Ru, and P. G. Etchegoin,
        "An Iterative Algorithm for Background Removal in Spectroscopy
        by Wavelet Transforms," Appl. Spectrosc. 63, 1370-1376 (2009)
    '''
    #rawdata = spectra#["int2"]  #unpack spectra from the dataframe
    bg = spectrum#rawdata  #the background will be calculated from the spectra
    for i in range(iterations):
        #first decompose to spectra to the specified level.
        coeffs = pywt.wavedec(bg, 'sym5', level=level) #tranform to wavelets
        #remove all high frequency levels
        for i in range(2,(level+1)):
            coeffs[i] = np.zeros_like(coeffs[i])
            #print(i)
        #reconstruct the wavelets to spectra space
        low_freq_spec = pywt.waverec(coeffs, 'sym5')
        #take the mimimum of current spec background vs low freq reconstruction
        bg = np.minimum(bg, low_freq_spec)
    #subtract the background from the original spectra
    newspec = spectrum - bg#rawdata - bg
    return newspec

#generate a new raman spectrum
spectrum = gendata_raman(num_peaks=5, plot=False) #simulate a raman spectrum
plt.plot(spectrum)

#remove baseline, remove noise and normalize a spectrum
spec = spectrum
spec = lowpass(spec, 2)
spec = remove_baseline(spec, 9, 10)
spec = normalize(spec)
plt.plot(spec)
#plt.show()

#remove noise, take second derivative, mean center and normalize a spectrum
spec2 = spectrum
spec2 = lowpass(spec2, 3)
spec2 = np.gradient(spec2, edge_order=2)
spec2 = spec2 - np.mean(spec2)
spec2 = normalize(spec2)
plt.plot(spec2)
plt.title("raw raman, corrected raman, and gradient raman")
plt.show()
#%%
''' resampling a spectrum '''
        

# wavenums = np.linspace(400.2, 2000.4, len(spectrum))
# print(wavenums)
# plt.scatter(wavenums, spec, s=2)
# plt.title("raman data")
# plt.show()
#The raman signal is now mapped to a raman shift (cm-1) so different spectra
#can now be compared


# in order to compare spectra collected on different machines, we sometimes need
# to interpolate and resample the data so that all spectra are sampled with equal spacing


import pandas as pd
def upsample_spectrum(spec, newsize):
    #spectrum should be a pandas dataframe
    
    #resample the spectrum
    #create a new high resolution regular index
    spec.index = spec['cm-1']

    start   = np.min(spec['cm-1'])
    stop    = np.max(spec['cm-1'])
    num     = newsize
    reg_idx = np.linspace(start, (stop-1), num)
    
    #create a new dataframe for the newly indexed, blank data
    reg_idx_df = pd.DataFrame(data=[], index=reg_idx, columns=["int"])
    
    #add the new blank dataframe to the original dataframe, then sort it by index
    upsampled_spectrum = pd.concat([spec, reg_idx_df]).sort_index()
    
    #interpolate the data and fill it in relative to the index and drop duplicates
    interpolated_spectrum = upsampled_spectrum.interpolate(method='index').drop_duplicates()
    
    #redefine the index 
    resampled_spectrum = interpolated_spectrum.reindex(reg_idx)
    return resampled_spectrum

#generate a spectrum and perform preprocessing
spectrum = gendata_raman(num_peaks=5, plot=False) #simulate a raman spectrum
#remove baseline, remove noise and normalize a spectrum
spec = spectrum
spec = lowpass(spec, 2)
spec = remove_baseline(spec, 7, 10)
spec = normalize(spec)
print(len(spec))


# make it resembles a raman spectrum by including wavenumbers
# in another equal length array
start = 400.2
stop  = 2460.7
wavenums = np.linspace(start, stop, len(spec))
raman = pd.DataFrame({'cm-1':wavenums, 'int':spec}) #generate a pandas dataframe
hdspec = upsample_spectrum(raman, newsize=20000)
print(len(hdspec))


plt.scatter(hdspec['cm-1'], hdspec['int'], s=0.1)
plt.title("upsampled raman data")
plt.show()


#%%
''' match a raman spectrum to a library of raman spectra '''
#generate an experimental spectrum, 
# then generate a whole library of spectra
# then compare the experimental spectrum to the library and find the best match

def make_dataframe_from_spec():
    #this function is just a shortcut to automate making a random raman spectrum
    #make new spec data
    spectrum = gendata_raman(num_peaks=5, plot=False) #simulate a raman spectrum

    #remove baseline, remove noise and normalize a spectrum
    spec = spectrum
    spec = lowpass(spec, 2)
    spec = remove_baseline(spec, 7, 10)
    spec = normalize(spec)
    
    #resample
    start = 400.2
    stop  = 2460.7
    wavenums = np.linspace(start, stop, len(spec))
    raman = pd.DataFrame({'cm-1':wavenums, 'int':spec}) #generate a pandas dataframe
    hdspec = upsample_spectrum(raman, newsize=20000)
    
    return hdspec


def dotScore(v1, v2):
    len1 = np.sqrt(np.dot(v1, v1))
    len2 = np.sqrt(np.dot(v2, v2))
    ds = np.arccos(np.dot(v1, v2) / (len1 * len2))
    return ds

def dotProduct(v1, v2):
    len1 = np.sqrt(np.dot(v1, v1))
    len2 = np.sqrt(np.dot(v2, v2))
    dp = np.dot(v1, v2) / (len1 * len2)
    #print(dp)
    return np.cos(dp)


def matchSpectra(exp, lib_list):
    scores = []
    dotproducts = []
    for lib in lib_list:
        xdata_, ldata_ = make_equal_spec_length(exp, lib)
        scores.append(dotScore(xdata_,ldata_)) 
        dotproducts.append(dotProduct(xdata_, ldata_))
    best_match_idx = np.argmin(scores)       
    dotscore   = scores[best_match_idx]
    dotproduct = dotproducts[best_match_idx]
    matchraman  = lib_list[best_match_idx]        
    return matchraman, dotproduct, dotscore

def make_equal_spec_length(exp, lib):
    ''' this function takes in two spectra object of arbitrary length
    and find the overlapping portion and then makes truncated versions
    of that section. The function then returns the two equal length spectra
    as np.arrays '''
    
    emin, emax = np.min(exp["cm-1"]), np.max(exp["cm-1"])
    lmin, lmax = np.min(lib["cm-1"]), np.max(lib["cm-1"])
    dif1 = emin - lmin
    dif2 = emax - lmax
    
    expfront, libfront = 0, 0
    expback, libback   = -1, -1
    
    if dif1 > 0: libfront = int(dif1)
    if dif1 < 0: expfront = int(dif1 * -1) 
    
    if dif2 > 0: expback  = int(dif2 * -1) -1
    if dif2 < 0: libback  = int(dif2) -1
    
    xdata = np.array(exp['int'])
    ldata = np.array(lib['int'])
    xdata_ = xdata[expfront:expback]
    ldata_ = ldata[libfront:libback]
    
    return xdata_, ldata_



#make n different spectra and save in a list. this will be our spectral library
n = 100
library = []
for i in range(n)  :
    newspec = make_dataframe_from_spec()
    #print(newspec.head())
    plt.plot(newspec['cm-1'], newspec['int'])
    library.append(newspec)
plt.title("library spectra")
plt.show()



#make one experimental spectrum
expspec = make_dataframe_from_spec()
plt.plot(expspec['cm-1'], expspec['int'])
plt.title("experimental spectrum")
plt.show()


#match the experimental data to one of the library spectrums
matchraman, dotproduct, dotscore = matchSpectra(expspec, library)
print(dotproduct, dotscore)

plt.plot(expspec['cm-1'], expspec['int'])
plt.plot(matchraman['cm-1'], matchraman['int'])
plt.title("best matched spec")
plt.show()


#%%






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
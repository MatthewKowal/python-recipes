# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:36:58 2024

@author: user1
"""




#%%
def import_expdata(exp_path):
    
    #import experimental data
    exp_data = [] # a list to hold all the new spectra
    filelist = get_exp_filelist(exp_path) # a list of all the new spectra
    
    #import each new spectrum
    for a_file in filelist:
        #determine metadate
        filename = os.path.basename(a_file)
        category   = "unknown"                       #metadata[0] #e.g. pa, ce, pp, pe, etc.
        color      = "not specified"                 #metadata[1] #e.g. red, green, blue
        samplename = filename[:-4]                   #metadata[2] #e.g. Polyamide 1., zoo-23-177
        source     = "Grant Lab Experimental Sample" #metadata[3] #e.g. Grant Lab, SLoPP
        status     = "raw" 
        #read spectral data
        rawdata = pd.read_csv(a_file, names=['cm-1', 'int'])
        rawdata['int'] = rawdata['int'].astype(float)
        #make sure the spectra is of even length
        if len(rawdata)%2 == 1: rawdata = rawdata[:-1] #drop last row if odd
        #build a new instance of the spectra class with the spectrums name and data and add it to a list
        new_spectra = Spectra(filename, rawdata, category, color, samplename, source, status)
        exp_data.append(new_spectra)
        #exp_data is a list of Spectra that conotains all our experimental data
    
    #process spectra
    processed_expdata = processSpectra(exp_data)


    return processed_expdata#a_string

def get_exp_filelist(path): #experimental spetra are all in one folder
    print("\n\nLoading experiment files...")
    print("\t\t NOTE: all files should be in the following format: [sample name].csv ")
    print("\t\t       Be sure to use brackets!")
    filelist = []
    for filename in os.listdir(path):
        full_filepath = os.path.join(path, filename)
        filelist.append(full_filepath)
        print(filename)
    print("\n")
    return filelist
#!!!  
def import_raw_library(librarypath):
    '''reads a folder tree of raw plastic Raman standard spectra.
        Then, processed them and saves the data as a pickle file'''
    
    print("Importing Spectra...")
    
    #check for processed library pickle file
    picklefilename = "mp library processed.pickle"
    rootpath, filename = os.path.split(__file__)  # get folder and filename of this script
    pfilepath = os.path.join(rootpath, picklefilename)
    if os.path.exists(pfilepath):
        with open(pfilepath, "rb") as f:
            spectra_list = pickle.load(f)
        print("Loading processed library pickle file...")
        print("\t", len(spectra_list), " files loaded.\n")
        return spectra_list
    
    filelist = get_lib_filelist(librarypath)

    
    
    spectra_list = [] 
    xmin_list = []
    xmax_list = []
    endp_list = []
    
    for f in filelist:
        
        file_name = os.path.basename(f)       
    
        metadata = file_name.split('__')
        if len(metadata)!=5:
            print("ERROR: file (", file_name, ") is missing 1 or more properties")
            print("\t Length of ", len(metadata), " should be 5")
            print("\t", metadata)
        category   = metadata[0] #e.g. pa, ce, pp, pe, etc.
        color      = metadata[1] #e.g. red, green, blue
        samplename = metadata[2] #e.g. Polyamide 1., zoo-23-177
        source     = metadata[3] #e.g. Grant Lab, SLoPP
        status     = metadata[4][:-4] #e.g. raw, preprocessed
                      #metadata[4][:-4] #e.g. raw, preprocessed
        
        #read the csv file from the file list and store it in a pandas dataframe
        #print(a_file)
        rawdata = pd.read_csv(f, names=['cm-1', 'int'])
        rawdata['int'] = rawdata['int'].astype(float)
        
        #tabulate endpoint to later plot in a histogram
        xmin = np.min(rawdata['cm-1'])
        xmax = np.max(rawdata['cm-1'])
        #print(xmin, xmax)
        xmin_list.append(xmin)
        xmax_list.append(xmax)
        endp_list.append(xmin)
        endp_list.append(xmax)
        
        
      
        #make sure the spectra is of even length
        if len(rawdata)%2 == 1: rawdata = rawdata[:-1] #drop last row if odd
    
        #build a new instance of the spectra class with the spectrums name and data and add it to a list
        new_spectra = Spectra(file_name,
                          rawdata,
                          category, color, samplename, source, status)
        spectra_list.append(new_spectra)   
    print("\t", len(spectra_list), " files loaded.\n")
  
    #process spectra
    processed_spectra_list = processSpectra(spectra_list)
    
    #save processed spectra as a pickle file
    save_pickle = True
    if save_pickle == True:
        rootpath, filename = os.path.split(__file__)  # get folder and filename of this script
        #exportpath    = os.path.join(rootpath, "export")
        #if not os.path.exists(exportpath): os.makedirs(exportpath)
        picklepath    = os.path.join(rootpath, "mp library processed.pickle")
        print("saving pickle fils to: ", picklepath)
        pickle.dump(processed_spectra_list, open(picklepath, "wb"))
    
    return processed_spectra_list


def get_lib_filelist(path): #the library files are in subfolders with material names
    filelist = []
    #print(os.listdir(path))
    #print("teststst")
    print("\t class      # of spectra \n\t ------     ---------")
    for folder in os.listdir(path):
        working_path = os.path.join(path, folder)
        #print(working_path)
        #print(os.listdir(working_path))
        n = 0
        for sample in os.listdir(working_path):
            working_file = os.path.join(working_path, sample)
            #print(working_file)
            filelist.append(working_file)
            n+=1
        print("\t ", folder, " \t\t", n)
    return filelist



def export_excel(matched_speclist):
    print("\nExporting excel file")
    #load matched data
    #m_exp_data =  pickle.load(open(m_exp_data_pkl_path, 'rb'))
    m_exp_data = matched_speclist
    #create blank excel file
    output = pd.DataFrame(columns=[])
    
    # #fill in excel file
    # for s in m_exp_data:
    #     output = output.append({"sample name":s.samplename,
    #                             "prediction":s.match,
    #                             "match score":s.dotscore,
    #                             }, ignore_index=True)  
    for s in matched_speclist:
        new_row = {"sample name":s.samplename,
                   "prediction":s.match,
                   "match score":s.dotscore,
                   "dot product":s.dotproduct}
        #print(type(new_row))
        #print(new_row)
        new_row_df = pd.DataFrame(new_row, index=[0])

        old_rows = output.copy()
        output = pd.concat([old_rows, new_row_df], ignore_index=True)


    #generate filename
    rootpath, filename = os.path.split(__file__)  # get folder and filename of this script
    exportpath    = os.path.join(rootpath, "export")
    if not os.path.exists(exportpath): os.makedirs(exportpath)
    save_excel_path = os.path.join(exportpath, "_matched_spectra.xlsx")
    
    #save excel file
    output.to_excel(save_excel_path, index=False)
    
    
    # a_string  = "Saved Match Data to:\n"
    # a_string += "\t\t" + save_excel_path +"\n\n"
    # a_string += "Results:\n"
    # a_string += "\n"+ output.to_string() +"\n"
    #a_string = m_exp_data_pkl_path+"\n"+base_path+"\n"+save_filename+"\n"+save_excel_path
    #a_string += "\n"+ output.to_string()
    
    return #a_string


folder, filename = os.path.split(__file__)  # get folder and filename of this script
#modfolder = os.path.join(folder)            # I cant remember why this is here
sys.path.insert(0, folder)               # add the current folder to the system path
import mp_backend as mp

def export_images(matched_speclist): #export images
    print("saving images")
    col="contodo"
    # base_path = os.path.split(mpath)[0]
    # expdata_matches = pickle.load(open(mpath, 'rb'))
    
    # a_string = "Saving composite spectra images:\n"
    
    #generate filepath
    rootpath, filename = os.path.split(__file__)  # get folder and filename of this script
    exportpath    = os.path.join(rootpath, "export")
    if not os.path.exists(exportpath): os.makedirs(exportpath)

    
    for s in matched_speclist:
        img_filepath = os.path.join(exportpath, (s.samplename + "-spec.png"))
    
        # plt.figure(figsize=(8,8))
        # plt.suptitle(s.samplename)
    
        # plt.subplot(2,1,1)
        # #plot unprocessed experimental data
        # plt.plot(s.specdata["cm-1"], s.specdata["int"],
        #           color="#002145",
        #           label=("Experimental Spectra, Unprocessed"))
        # #plot unprocessed library match
        # plt.plot(s.matchspec["cm-1"], s.matchspec["int"],
        #            color="#00A7E1",
        #            label=("Library Spectra, Unprocessed"))
        # plt.legend()
    
        # plt.subplot(2,1,2)
        # # #plot baseline corrected match
        # plt.plot(s.specdata["cm-1"], s.specdata[col],
        #           color="#002145",
        #           label=("Experimental Spectra, Processed"))
        # # #plot baseline corrected sample
        # plt.plot(s.matchspec["cm-1"], s.matchspec[col],
        #           color="#00A7E1",
        #           label=("Library Spectra, Processed: " + s.match))
                  
        plt.figure(figsize=(12,8))
        plt.suptitle((s.samplename+" - dotproduct:", str(s.dotproduct)))
    
        plt.subplot(3,1,1)
        #plot unprocessed experimental data
        plt.plot(s.specdata["cm-1"], s.specdata["int"],
                  color="#002145",
                  label=("Experimental Spectra, Unprocessed"))
        #plot unprocessed library match
        plt.plot(s.matchspec["cm-1"], s.matchspec["int"],
                   color="#00A7E1",
                   label=("Library Spectra, Unprocessed"))
        plt.legend()
    
        plt.subplot(3,1,2)
        #plot unprocessed experimental data
        plt.plot(s.specdata["cm-1"], s.specdata["zeroed"],
                  color="#002145",
                  label=("Experimental Spectra, Baseline Corrected"))
        #plot unprocessed library match
        plt.plot(s.matchspec["cm-1"], s.matchspec["zeroed"],
                   color="#00A7E1",
                   label=("Library Spectra, Baseline Corrected"))
        plt.legend()
    
    
        plt.subplot(3,1,3)
        # #plot baseline corrected match
        plt.plot(s.specdata["cm-1"], s.specdata[col],
                  color="#002145",
                  label=("Experimental Spectra, Processed"))
        # #plot baseline corrected sample
        plt.plot(s.matchspec["cm-1"], s.matchspec[col],
                  color="#00A7E1",
                  label=("Library Spectra, Processed: " + s.match))
        
    
    
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(img_filepath)#, bbox_inches="tight")
        #plt.show()
        plt.close()
        #a_string += "\t\t" + img_filepath + "\n"
        
    #a_string += "\n"    
    #a_string = "test" + mpath + "\n" + base_path + "\n" + img_filepath
    
    return #a_string
#%%


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
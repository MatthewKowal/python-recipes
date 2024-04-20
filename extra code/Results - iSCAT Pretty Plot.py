# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:17:20 2023

@author: user1
"""


#add the current folder to the system directory to look for modules
import sys
import os
import time
import numpy as np
from PIL import Image

import matplotlib.colors as mcolors

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


folder, filename = os.path.split(__file__)  # get folder and filename of this script
#modfolder = os.path.join(folder)            # I cant remember why this is here
sys.path.insert(0, folder)               # add the current folder to the system path


import iscat_yolo_v1_0_4 as iscat


def i16_to_i8(i16):
    clipmin = np.min(i16)
    clipmax = np.max(i16)
    i8 = np.clip( ((i16 - clipmin) * 255 / (clipmax-clipmin)), 0, 255).astype(np.uint8)
    return i8


def save_pimages_from_pickle(pkl_path, tag):
    op = os.path.split(pkl_path)[0]
    print(op)
    
    pimage_folder = os.path.join(op, tag)
    if not os.path.exists(pimage_folder): os.makedirs(pimage_folder)
 
    for p in pl:
        #print(type(p.yoloimage_vec[0]))
             
        #save 30x30 pimage
        for c, i in enumerate(p.pimage_vec):
            #print(p.px_vec[c], p.py_vec[c], p.wx_vec[c], p.wy_vec[c])
            #print(os.path.join(constants["output path"], "pimage", ("pimage-" + str(c) +".png")), type(i), i.shape)
            im = Image.fromarray(i16_to_i8(i))
            #im.show()
            img_out_path = os.path.join(pimage_folder, ("pimage-pID " + str(p.pID) +"-"+ str(c) +".png"))
            #print(img_out_path)
            im.save(img_out_path)
     
        
     

def plot_gradient(cmap):
    # Plot a colorbar to visualize the colormap
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_title('Custom Colormap')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    plt.show()

def create_custom_colormap(colors, name='custom_colormap', n_bins=1000):
    """Create a custom colormap from a list of RGB colors."""
    cmap = mcolors.LinearSegmentedColormap.from_list(name, colors, N=n_bins)
    return cmap




# load 25 nm image
p, f = 10, 12
pkl_path = r"C:/Users/user1/Desktop/ITO electrode microscopy/_ACS Nano/Results 1 - Mass Photometry/Results - 1.1 iSCAT Contrast/pimages/25nm/2023-03-16_15-50-04_raw_12_256_200yolo-c45-trim10.pkl"
pl        = iscat.load_pickle_data(pkl_path)
img_data25 = pl[p-1].pimage_vec[f]
#plt.imshow(img_data25)
#plt.show()

# load 50 nm image
p, f = 24, 15
pkl_path = r"C:/Users/user1/Desktop/ITO electrode microscopy/_ACS Nano/Results 1 - Mass Photometry/Results - 1.1 iSCAT Contrast/pimages/50nm/2023-03-16_17-30-24_raw_12_256_200yolo-c45-trim10.pkl"
pl        = iscat.load_pickle_data(pkl_path)
img_data50 = pl[p-1].pimage_vec[f]
#plt.imshow(img_data50)
#plt.show()

# load 100 nm image
p, f = 8, 3  #2, 15   # 9, 21
pkl_path = r"C:/Users/user1/Desktop/ITO electrode microscopy/_ACS Nano/Results 1 - Mass Photometry/Results - 1.1 iSCAT Contrast/pimages/100nm/2023-03-17_16-54-19_raw_12_256_200yolo-c45-trim10.pkl"
#p, f = 589, 23   
#pkl_path = r"C:/Users/user1/Desktop/ITO electrode microscopy/_ACS Nano/Results 1 - Mass Photometry/Results - 1.1 iSCAT Contrast/pimages/100nm 2/2023-03-17_18-06-25_raw_12_256_200yolo-c45-trim10.pkl"
#p, f = 6, 17   # 4,21
#pkl_path = r"C:/Users/user1/Desktop/ITO electrode microscopy/_ACS Nano/Results 1 - Mass Photometry/Results - 1.1 iSCAT Contrast/pimages/100nm 3/2023-03-17_17-52-42_raw_12_256_200yolo-c45-trim10.pkl"
pl        = iscat.load_pickle_data(pkl_path)
img_data100 = pl[p-1].pimage_vec[f]
#save_pimages_from_pickle(pkl_path, "100nm pimage")
#plt.imshow(img_data100)
#plt.show()



# load 200 nm image
p, f = 11, 21
pkl_path = r"C:/Users/user1/Desktop/ITO electrode microscopy/_ACS Nano/Results 1 - Mass Photometry/Results - 1.1 iSCAT Contrast/pimages/200nm/2023-04-11_19-23-15_raw_12_256_200yolo-c45-trim10.pkl"
pl        = iscat.load_pickle_data(pkl_path)
img_data200 = pl[p-1].pimage_vec[f]
#save_pimages_from_pickle(pkl_path, "200nm pimage")

#plt.imshow(img_data200)
#plt.show()

import numpy as np

# Save the array to a CSV file
data = img_data25
np.savetxt(r'C:/Users/user1/Documents/GitHub/python-recipes/data/25nm.csv', data, delimiter=',', fmt='%0.5f')

data = img_data50
np.savetxt(r'C:/Users/user1/Documents/GitHub/python-recipes/data/50nm.csv', data, delimiter=',', fmt='%0.5f')

data = img_data100
np.savetxt(r'C:/Users/user1/Documents/GitHub/python-recipes/data/100nm.csv', data, delimiter=',', fmt='%0.5f')

data = img_data200
np.savetxt(r'C:/Users/user1/Documents/GitHub/python-recipes/data/200nm.csv', data, delimiter=',', fmt='%0.5f')



import pickle
data = img_data25
file_path = r'C:/Users/user1/Documents/GitHub/python-recipes/data/25nm.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(data, file)

data = img_data50
file_path = r'C:/Users/user1/Documents/GitHub/python-recipes/data/50nm.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(data, file)

data = img_data100
file_path = r'C:/Users/user1/Documents/GitHub/python-recipes/data/100nm.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(data, file)

data = img_data200
file_path = r'C:/Users/user1/Documents/GitHub/python-recipes/data/200nm.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(data, file)

#%%

print("25nm min/max:  ", np.min(img_data25),  np.max(img_data25))
print("50nm min/max:  ", np.min(img_data50),  np.max(img_data50))
print("100nm min/max: ", np.min(img_data100), np.max(img_data100))
print("200nm min/max: ", np.min(img_data200), np.max(img_data200))










colors        = [ "#FCDE63","#34b3d1", "#fdb39c", "#808080", "#aac8a7", "#8294c4"]    # retro optical diagram colors
colors_darker = ["#DDC458", "#3479D1", "#FD8E7C", "#666666", "#95ad92", "#6a7a9e"] # darker version of retro optical colors (for outlines)


''' Create a custom colormap with a gradient between the colors '''
# Define the RGB values of the three colors
r1, g1, b1 = 55, 121, 209    # pink
r2, g2, b2 = 252, 222, 99    # yellow
r3, g3, b3 = 253, 142, 124   # blue

color1 = (r1/255, g1/255, b1/255)  # 3479d1
color2 = (r2/255, g2/255, b2/255)  # fcde63
color3 = (r3/255, g3/255, b3/255)  # fd8e7c


#use 2 colors
custom_colors = [color1, color3]

#or use 3 colors
#custom_colors = [color1, color2, color3]

#generate new color gradient map
newcmp = create_custom_colormap(custom_colors)
#plot_gradient(newcmp)





# from PIL import Image
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
# import matplotlib.cm as cm







def pretty_plot(img_data, zmin, zmax, zticks):

    plt.rcParams['figure.figsize'] = [8, 8]
    #fig = plt.figure(figsize=plt.figaspect(0.1)*2.5)
    
    
    plt.figure(dpi=300)
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 24}    
    SMALL_SIZE = 12
    MEDIUM_SIZE=18
    BIGGER_SIZE=20
    #matplotlib.rc('font', **font)
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE) 

    ax = plt.figure().add_subplot(projection='3d')
    #X, Y, Z = axes3d.get_test_data(0.05)
    ax.set_box_aspect(aspect=(1, 1, 1))
    
    ax.tick_params(direction='out', length=6, width=2, colors='k', pad=10) #_color='r', grid_alpha=0.5)
    
    x, y = np.mgrid[0:30, 0:30]
    
    ''' PLOT THE 3D SURFACE ''' 
    #ax.plot_surface(x, y, img_data, cmap=newcmp, alpha=0.3)
    ax.plot_surface(x, y, img_data, cmap=newcmp, edgecolor=colors_darker[1], lw=0.1, rstride=2, cstride=2, alpha=0.3)
    
    ''' PLOT THE PARTICLE FLOOR '''
    ax.contourf(x, y, img_data, zdir='z', levels=100, offset=zmin, cmap='gray', antialiased=True)
    ax.contourf(x, y, img_data, zdir='z', levels=100, offset=zmin, cmap='gray')
    
    #ax.imshow(img_data200, zdir='z', offset=0.5)
    
    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    #ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    #ax.contourf(x, y, data, zdir='z', levels=256, offset=v_min, cmap=cm.gray)
    #ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    #ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
    
    ''' SET AXIS LIMITS AND TICKS '''
    ax.set(xlim=(0, 30), ylim=(0, 30), zlim=(zmin, zmax))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks(zticks)
    
    ''' SET VIEW ANGLE OF 3D LOT '''
    elev = 14
    azim = 127
    plt.gca().view_init(elev, azim)
    plt.tight_layout()
    plt.show()

pretty_plot(img_data25, 0.96, 1.04, [0.96, 1, 1.04])
pretty_plot(img_data50, 0.93, 1.07, [0.93, 1, 1.07])
pretty_plot(img_data100, 0.84, 1.16, [0.84, 1, 1.16])
pretty_plot(img_data200, 0.54, 1.46, [0.54, 1, 1.46])


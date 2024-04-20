# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:20:23 2024

@author: user1
"""

#working with filenames and filepaths

import os

a = r"c:\windows\windows.exe"
print("a:\t\t", a, type(a), len(a))

#split a path and return a tuple composed of the file location and the filename.
#the tuple is similar to a list, but it is immutable
b = os.path.split(a)
print("b:\t\t", b, type(b), len(b), "\nb[0]:\t", b[0], "\nb[1]\t", b[1], "\n")

#automatically read the tuple into two strings.
#the first one is the file location, the second is the filename
c, d = os.path.split(a)
print("c:\t\t", c, "\nd:\t\t", d, "\n")

#join a file location and a file name into a string
e = os.path.join(c, d)
print("e:\t\t", e)

# getting the filename and location of the current running script
scriptfolder, scriptfilename = os.path.split(__file__)  # get folder and filename of this script
print(scriptfolder)
print(scriptfilename)

#%%


import os
import sys
import pickle

def load_var_from_pickle(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def save_var_as_pickle(variable, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(variable, file)




image = load_var_from_pickle(os.path.join(scriptfolder, "data", "200nm.pkl"))

save_var_as_pickle(image, os.path.join(scriptfolder, "data", "test.file"))
#%%

# get a list of files in a specific folder (not recursive)
def get_filelist(path): #experimental spetra are all in one folder
    filelist = []
    for filename in os.listdir(path):
        full_filepath = os.path.join(path, filename)
        filelist.append(full_filepath)
    return filelist

fl = get_filelist(scriptfolder)
[print(f) for f in fl]



#%%
#import an excel file using pandas
import os
scriptfolder, scriptfilename = os.path.split(__file__)  # get folder and filename of this script
import pandas as pd
import numpy as np

#load a huge excel file into a pandas dataframe
pl_df = pd.read_excel(os.path.join(scriptfolder, "data", "particle list.xlsx"))

#load complex data from a csv using pandas
pl2_df = pd.read_csv(os.path.join(scriptfolder, "data", "particle list.csv"))

#load simple raman data from a csv using pandas
raman_df = pd.read_csv(os.path.join(scriptfolder, "data", "raman acrylic.csv"))


#load simple raman data from a csv using numpy
raman_arr = np.genfromtxt(os.path.join(scriptfolder, "data", "raman acrylic.csv"), delimiter=',')
#import a csv file


#%%







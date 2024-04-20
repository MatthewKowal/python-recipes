# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:36:58 2024

@author: user1
"""

#datatypes

#NUMBERS
#integers are whole numbers
integer = 1
print(type(integer), integer)

#a floating point has a decimal place
floatingpoint = 1.234
print(type(floatingpoint), floatingpoint)

#STRINGS
string = "this is a string"
print(string)
print(string+string)
print(string*3)

#LISTS
#lists are like arrays that can store any datatype. 
some_list = [1, 2, 3, "five", 6, 7.0]
print(some_list)

#you can loop through a list and read each elements value
for i in some_list:
    print(type(i), i)
    
#you can append new values to a list
some_list.append("ate")
print(some_list)

#you can also replace any element with a new piece of data
some_list[0] = "one"
some_list[-1] = 8.0
print(some_list)

#to find more info about lists and learn what you can do with them,
#google "python list methods"





#%%

import numpy as np
# numpy arrays
# numpy arrays are usually the datatype you want for storing 1d and 2d data like spectra and images
# many calculations and manipulations can be done on numpy arrays with numpys built in functions

arr1 = np.linspace(0,7,8)
arr2 = np.ones(5)
arr22 = np.ones([5,6])
arr3 = np.zeros(10)
arr4 = np.zeros(arr22.shape)


#%%

#generating random numbers

import numpy as np

#generate random numbers between 0 and 1. you can request the shape of the data
print("\n np.random.rand")
print( np.random.rand()     )    # returns 1 value. a decimal between 0 and 1
print( np.random.rand()*100 )    # returns a number from zero to 100
print( np.random.rand(5)    )   # returns a 1d array
print( np.random.rand(5,5)  ) # returns a 2d array

#generate random integers
print("\n np.random.randint")
print(np.random.randint(100)) # returns an integer between 0 and this number
print(np.random.randint(190,200)) #returns an integer between these two numbers


#%%
#clear all variables
import sys
sys.modules[__name__].__dict__.clear()

#%%

# loops and list itteration
# 
some_list = [1,2,5,8,2,5,8,3,6,8,9,4]
print("\n\n loop through the elements of a list")
for v in some_list:
    print(v)
    
    
print("\n\n loop through the indicies of a list")
for i in range(len(some_list)):
    print(i)


print("\n\n use the indicies to address the list at an index")    
for i in range(len(some_list)):
    print(some_list[i])


print("\n\n enumerating the list gives us the index and the element value")
for i, v in enumerate(some_list):
    print(i, v)

#%%
some_list = [1,2,5,8,2,5,8,3,6,8,9,4]


# use loops and list itterations to generate a new list from some list data
w = [v+10 for v in some_list]
print(w)

x = [i for i in range(len(some_list))]
print(x)

y = [some_list[i]+10 for i in range(len(some_list))]
print(y)

z = [i+v for i, v in enumerate(some_list)]
print(z)

#this code can be written in multiple lines like this
zz = []
for i, v in enumerate(some_list):
    zz.append(i+v)
print(zz)





#%%

# objects

class Human:
    def __init__(self, name, age, hobbies_list):
        self.name           = name                
        self.age            = age                
        self.hobbies_list   = hobbies_list
        
    def ageOneYear(self):
        self.age += 1
        print(self.name, " is now one year older")
        
    def changeName(self, new_name):
        print(self.name, "changed their name to ", new_name)
        self.name = new_name
        

dude = Human("cyrus", 10, ["eating", "sleeping"])
print("\nObject:\t", dude, "\nType:\t\t", type(dude))
print("\nName:\t", dude.name, "\nAge\t\t", dude.age, "\nHobbies:", dude.hobbies_list, "\n")

dude.changeName("bill")
dude.ageOneYear()
print("\nName:\t", dude.name, "\nAge\t\t", dude.age, "\nHobbies:", dude.hobbies_list)



#%%



#filenames

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
import os
import sys
folder, filename = os.path.split(__file__)  # get folder and filename of this script
print(folder)
print(filename)

#%%

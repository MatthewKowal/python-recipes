# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:20:23 2024

@author: user1
"""

def save_var_as_pickle():
    return 1

def load_var_from_pickle():
    return 1

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
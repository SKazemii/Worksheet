# -*- coding: utf-8 -*-
"""
  FilterH5DataByCardReads.py

@author: Patrick Connor (Stepscan Technologies Inc.)
"""

# Imports
import time, datetime
import numpy as np
#from pylab import figure, imshow
import h5py, csv
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MLPackage import step_utils as ut

# User input
inHDF5File = '/Users/saeedkazemi/Documents/Python/Worksheet/Datasets/Card plus footprint data/Frames_20210723152012.h5'
inCardReaderFile = '/Users/saeedkazemi/Documents/Python/Worksheet/Datasets/Card plus footprint data/CardLog_20210723152012.csv'
parent_dir= '/Users/saeedkazemi/Documents/Python/Worksheet/Datasets/Segmented'#to save the data


# Card Reader loading and parsing 
# Card Read format: Card Reader ID, year, month, day of the week, day of the month, 
# hour (24h), minute, second, millisecond, 5 more numbers associated with the card number
CardReads = []
with open(inCardReaderFile) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        num_row = []
        for col in row:
            num_row.append(int(col))
        CardReads.append(num_row)

# Determine millisecond timestamps for card reader data
card_timestamps = np.zeros(len(CardReads))
for i in range(len(CardReads)):
    card_timestamps[i] = 1000*time.mktime(datetime.datetime(CardReads[i][1], CardReads[i][2], CardReads[i][4], 
                        CardReads[i][5], CardReads[i][6], CardReads[i][7], 0).timetuple()) + CardReads[i][8]        
        
# H5 File loading and parsing
pressure_data_file = h5py.File(inHDF5File,"r+") # open the file
full_data = pressure_data_file.get("I"); # get the raw data including metadata
metadata = full_data[:3000,:22]

# Note, the presure data could be very large and might not fit in memory.  You 
#  may have to find the range of timestamps you want and then grab chunks of 
#  data directly from the dataset rather than create a full sized array and 
#  extracting from that.
#pressure_data = (full_data[:,22:]).reshape([full_data.shape[0], metadata[0][5], metadata[0][4]]); # extract the raw pressure data, and reshape into a 3-dimensional array (frame no., y, x)

# Determine millisecond timestamps from the Epoch
H5_timestamps = np.zeros(len(metadata))
for i in range(len(metadata)):
    H5_timestamps[i] = 1000*time.mktime(datetime.datetime(metadata[i][8], metadata[i][9], metadata[i][11], 
                        metadata[i][12], metadata[i][13], metadata[i][14], 0).timetuple()) + metadata[i][15]

## e.g., Display cumulative image over recording
#figure()
#imshow(np.sum(pressure_data,0))

################

#pradeep
#plot the grid of foot segments
def plot_segmented_foot(foot):
    n_foot=int(np.sqrt(len(foot)))+1
    c=0
    Position = range(1,len(foot) + 1)
    fig = plt.figure()
    for k in range(len(foot)):
      ax = fig.add_subplot(n_foot,n_foot,Position[k])
      ax.imshow(np.sum(foot[c],0))
      ax.set_xlabel('Footprint-'+str(Position[k]))
      c=c+1;
    plt.show()
#get the list of individuals from the card log
Users=np.unique(np.vstack(CardReads)[:,12])


data_dir = os.path.join(parent_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(data_dir)
CardReads1=np.vstack(CardReads);inn=1;out=1
for u in [Users[0]]:
    foot_print_in=[];foot_print_out=[]
    u_path = os.path.join(data_dir, str(u));
    if not os.path.exists(u_path):
        os.mkdir(u_path);os.mkdir(u_path+'/in');os.mkdir(u_path+'/out')
    #store the data per user      
    for i,x in enumerate(CardReads1):
        if x[0]==1 and x[9]==1 and x[12]==u:
            print('in user')
            tm_stmp=card_timestamps[i]
            for idx,t in enumerate(H5_timestamps):
                #print(idx)
                if t>tm_stmp:
                    tmp=full_data[idx-3000:idx,22:]
                    walk_in=(tmp).reshape([tmp.shape[0], metadata[0][5], metadata[0][4]]);
                    d1=ut.extractFootPrints(walk_in)
                    in_foot_list=[]
                    for x in d1[0]:
                        tmp=np.vstack(x)
                        in_foot_list.append(tmp.reshape(80,60,200))
                        plt.figure()
                        print(np.sum(tmp,0))
                        print(np.sum(tmp,0).shape)

                        plt.imshow(np.sum(tmp,0))
                        plt.show()
                    hf = h5py.File(u_path+'/in/'+'U'+str(u)+'_'+str(inn), 'w')
                    hf.create_dataset('dataset_1', data=in_foot_list)
                    hf.close()
                    inn=inn+1
                    break;
        if x[0]==1 and x[9]==10 and x[12]==u:
            print('out user')
            tm_stmp=card_timestamps[i]
            for idx,t in enumerate(H5_timestamps):
                #print('inner loop to break..'+str(idx))
                if t>tm_stmp:
                    tmp=full_data[idx:idx+3000,22:]
                    walk_out=(tmp).reshape([tmp.shape[0], metadata[0][5], metadata[0][4]]);
                    d2=ut.extractFootPrints(walk_out)
                    out_foot_list=[]
                    for x in d2[0]:
                        tmp=np.vstack(x)
                        out_foot_list.append(tmp.reshape(80,60,200))
                    hf = h5py.File(u_path+'/in/'+'U'+str(u)+'_'+str(inn), 'w')
                    hf.create_dataset('dataset_1', data=out_foot_list)
                    hf.close()
                    out=out+1
                 
                    break;


######
#Temporal data plots
##########                
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# def plot_foot_video:
fig, (ax1, ax2) = plt.subplots(1, 2)

im1 = ax1.imshow(walk_in[20])#,cmap='gist_gray_r', vmin=0, vmax=1)
xdata=[];ydata=[]
t=np.arange(3000)
im2, = ax2.plot(t[0],np.sum(walk_in[520]), lw=2)


def init():
    im1.set_data(np.zeros((720, 720)))
    ax2.set_ylim(0, 40000)
    ax2.set_xlim(0, 2999)
    del xdata[:]
    del ydata[:]
    im2.set_data(xdata, ydata)
    im2.set_data(xdata,ydata)
    
def updatefig(i):   
    im1.set_data(walk_in[i])
    xdata.append(t[i]);ydata.append(np.sum(walk_in[i]))
    im2.set_data(xdata,ydata)
   
   
    return im1,im2

ani = animation.FuncAnimation(fig, updatefig,init_func=init, frames=3000,interval=50)
plt.show()
###################  
    

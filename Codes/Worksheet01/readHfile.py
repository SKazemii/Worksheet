# Imports
import time, datetime
import numpy as np
from pylab import figure, imshow
import h5py, csv, sys, os
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numpy import sum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MLPackage import step_utils as ut

np.set_printoptions(edgeitems=127)


# if False:#
if True:
    inHDF5File = '/Users/saeedkazemi/Documents/Python/Worksheet/Datasets/StepScan Live/barefoot_1.h5'


    pressure_data_file = h5py.File(inHDF5File, 'r')


    full_data = pressure_data_file.get("I"); # get the raw data including metadata
    # print(full_data.shape)

    metadata = full_data[:,:22]

    # print(metadata)

    # Determine millisecond timestamps from the Epoch
    # H5_timestamps = np.zeros(len(metadata))
    # for i in range(len(metadata)):
    #     H5_timestamps[i] = 1000 * time.mktime(datetime.datetime(metadata[i][8], metadata[i][9], metadata[i][11], 
    #                         metadata[i][12], metadata[i][13], metadata[i][14], 0).timetuple()) + metadata[i][15]
    # print(H5_timestamps.shape)

    parent_dir='./Datasets/Segmented'
    # CardReads1=np.vstack(CardReads)
    # Users=np.unique(metadata[:,12])

    # inn=1
    # out=1
    # for u in Users:
    foot_print=[]
    # foot_print_out=[]


    u_path = os.path.join(parent_dir, str(0))
    if not os.path.exists(u_path):
        os.mkdir(u_path)

    tmp=full_data[700:1000,22:]
    walk_in=(tmp).reshape([tmp.shape[0], metadata[0][5], metadata[0][4]])
    hf = h5py.File(u_path+'/'+'U_'+str(0)+'.h5', 'w')
    hf.create_dataset('dataset_1', data=walk_in)
    hf.close()


    d1=ut.extractFootPrints(walk_in)

    in_foot_list=[]
    for x in d1[0]:
        tmp=np.vstack(x)
        in_foot_list.append(tmp.reshape(80,80,100))
        
    print((in_foot_list[0].shape))

    hf = h5py.File(u_path+'/'+'U_'+str(0)+'_segmented.h5', 'w')
    hf.create_dataset('dataset_1', data=in_foot_list)
    hf.close()
    


    foot_print.append(walk_in)#append the user ID and the associated data
    figure()
    # print(np.sum(walk_in,0))
    print(np.sum(walk_in,0).shape)

    plt.imshow(np.sum(walk_in,0))


    # foot_print.append(tmp)#append the user ID and the associated data
    figure()
    # print(np.sum(tmp,0))
    # print(np.sum(tmp,0).shape)

    plt.imshow(np.sum(in_foot_list[0],0))
    plt.show()
# sys.exit()

print("[INFO] done!!!")
sys.exit()

def isFootprintUp(foot):
    foot = np.array(foot)    
    fshape = np.shape(foot);    
    tc = int(np.sum(np.sum(foot,(0,1))/float(np.sum(foot))*np.arange(fshape[2]))  )   
    #tc = (find(np.sum(np.sum(foot,1),0) < 500).tolist() + [foot.shape[2]])[0]/2;
    footEarly = np.sum(foot[:,:,:tc],2)
    footLate = np.sum(foot[:,:,tc:],2)
    
    # print(np.sum(np.sum(footEarly*np.reshape(np.arange(fshape[0]),[fshape[0],1])*np.ones(fshape[1]),0)))

    ycEarly = np.sum(np.sum(footEarly*np.reshape(np.arange(fshape[0]),[fshape[0],1])*np.ones(fshape[1]),0))/(np.sum(footEarly)+1e-20)
    ycLate = np.sum(np.sum(footLate*np.reshape(np.arange(fshape[0]),[fshape[0],1])*np.ones(fshape[1]),0))/(np.sum(footLate)+1e-20)

    # print((footEarly*np.reshape(np.arange(fshape[0]),[fshape[0],1])).shape)
    # print(np.reshape(np.arange(fshape[0]),[fshape[0],1]))
    # sys.exit()
    print(ycEarly.shape)
    print(footEarly.shape)
    return ycEarly > ycLate

def maxCC1D(vector):  #finds the maximum connected area under the curve, where areas/bumps are separated by input vector values of 0.
    maxarea = 0; maxLength = 0;    
    tsum = 0; buff = 25; #10;
    zerocount = 0;    
    for iterJ in range(1,np.size(vector)): 
        if vector[iterJ] == 0:            
            if vector[iterJ-1] != 0:
                zerocount = 0;
            zerocount += 1;
            if tsum > maxarea and zerocount == buff:
                maxarea = tsum;
                maxLength = iterJ-buff;
            if zerocount > buff:
                tsum = 0;
        else:            
            tsum = tsum + vector[iterJ];
    if tsum > maxarea:  # In case the body finished on the last element of the vector
        maxarea = tsum;
        maxLength = iterJ;
    return [maxarea, maxLength];
 

def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res



print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

hf = h5py.File('/Users/saeedkazemi/Documents/Python/Worksheet/Datasets/Segmented/0/U_0.h5', 'r')
n1 = hf.get('dataset_1')
n1 = np.array(n1[:,:,:])
print(n1.shape)

# n2=n1[500:800,:,:]

# d=util.extractFootPrints(n2)



# determine central time of each footprint by finding the times in which they put most pressure to one side or the other
data = n1.transpose(1,2,0)[:,:,:].copy() # make a copy
threshold = 20 # threshold above which pressures are considered to be real
dProfile = np.sum(data, (0,1))   
noFPMask = dProfile < np.mean(dProfile); # mostly frames that do not contain footprints
background = (np.sum((data > 0)*noFPMask,2)/float(np.sum(noFPMask))) > 0.01;        


data = (data.transpose(2,0,1)*(1-background)).transpose(1,2,0);        
datathresh = 100; #np.percentile(np.sum(data,(0,1)),25) + 500;   
footmask = data[:,:,:] > threshold;    

# print(footmask*data )
# print(footmask.shape )
# sys.exit()
peaks = []; # find the peaks in the masksum curve to identify the central times in which feet are planted 

ys = 80; xs = 60; ts = 200; #the dimensions of a footprint "box" maximum size of footprint box. 
footprints = []; yclist = []; xclist = []; tclist = []
GRF = np.sum(np.sum(footmask*data,1),0);    
iterI = -1
while(find((GRF[:-1] < 50)*(GRF[1:] >= 50)).size > 0): # All it does take GRF shift back 1 timestamp find less 50 
    
    iterI += 1;        
    tc = find((GRF[:-1] < 50)*(GRF[1:] >= 50))[0]
    # plt.plot(range(0,1300),tc)
    # plt.show()
    # print(max(0,tc-ts/2))
    # print(min(np.size(footmask,2),tc+ts/2))
    # print(footmask.shape )
    tslice = footmask[:,:,int(max(0,tc-ts/2)):int(min(np.size(footmask,2),tc+ts/2))]        
    flat = np.sum(tslice,2) > 0   
    profile = np.sum(flat,1)        
    walkUp = isFootprintUp(tslice)
    # print(walkUp)
    print(walkUp.shape)

    if walkUp:      
        footend = maxCC1D(profile)[1]
        flatmask = flat*0; flatmask[max(footend-ys,0):footend] = 1; 
    else:
        footend = maxCC1D(profile[::-1])[1]
        flatmask = flat*0; flatmask[(tslice.shape[0]-footend):min(tslice.shape[0],tslice.shape[0]-footend+ys)] = 1; 
    flat = flat*flatmask; # mask out all footprints other than the one we're focusing on.               
    yc = int(np.sum(np.sum(flat*np.reshape(np.arange(0,np.size(footmask,0)),[np.size(footmask,0),1])*np.ones(np.size(footmask,1)),0))/np.sum(flat))
    xc = int(np.sum(np.sum(flat*np.reshape(np.ones(np.size(footmask,0)),[np.size(footmask,0),1])*np.arange(0,np.size(footmask,1)),0))/np.sum(flat))
    for iterR in range(3):             # Finds centre of pressure in y and x axis           
        tfp = np.zeros([ys, xs, ts])
        tfp[int(ys/2-(yc-max(0,(yc-ys/2)))):int(ys/2-(yc-min(np.size(data,0),yc+ys/2))),int(xs/2-(xc-max(0,(xc-xs/2)))):int(xs/2-(xc-min(np.size(data,1),xc+xs/2))),int(ts/2-(tc-max(0,(tc-ts/2)))):int(ts/2-(tc-min(np.size(data,2),tc+ts/2)))] = \
            data[int(max(0,(yc-ys/2))):int(min(np.size(data,0),yc+ys/2)),int(max(0,(xc-xs/2))):int(min(np.size(data,1),xc+xs/2)),int(max(0,(tc-ts/2))):int(min(np.size(data,2),tc+ts/2))];
        if sum(sum(sum(tfp,0),0) > datathresh) == 0:
            continue; #there is no footprint, it's just picking up noise.
        flat = np.amax(tfp,2) > threshold

        # move the volume to better capture the footprint
        footSlices = find(sum(sum(tfp,0),0) > datathresh).tolist(); footSlices.append(220); #footSlices.append(240)            
        tc += footSlices[0]; # fixing tc so that the heelstrike occurs at beginning
        yc += int(np.sum(np.sum(flat*np.reshape(np.arange(0,np.size(flat,0)),[np.size(flat,0),1])*np.ones(np.size(flat,1)),0))/np.sum(flat)) - ys/2;            
        xc += int(np.sum(np.sum(flat*np.reshape(np.ones(np.size(flat,0)),[np.size(flat,0),1])*np.arange(0,np.size(flat,1)),0))/np.sum(flat)) - xs/2;
        duration = footSlices[find(np.diff(footSlices) > 15)[0]] - footSlices[0] + 1;#find(np.diff(footSlices) == 1)[-1]+1 - find(np.diff(footSlices) == 1)[0]
        #duration = find(np.diff(footSlices) > 10)[0] + 1;# find the end of the footprint        
        #tslice = footmask[max(0,(yc-ys/2)):min(np.size(data,0),yc+ys/2),max(0,(xc-xs/2)):min(np.size(data,1),xc+xs/2),max(0,tc-ts/2):min(np.size(footmask,2),tc-ts/2+duration)];        
        
    if duration == 200:
        print('*************** Time slice is not big enough to contain the full foot *************** START')
        np.sum(np.sum(tfp,1),0)
            
    #grab footprint and append to list of footprints
    footprint = np.zeros([ys, xs, ts])
    footprint[(ys/2-(yc-max(0,(yc-ys/2)))):(ys/2-(yc-min(np.size(data,0),yc+ys/2))),(xs/2-(xc-max(0,(xc-xs/2)))):(xs/2-(xc-min(np.size(data,1),xc+xs/2))),(ts/2-(tc-max(0,(tc-ts/2)))):(ts/2-(tc-min(np.size(data,2),tc-ts/2+duration)))] = \
        data[max(0,(yc-ys/2)):min(np.size(data,0),yc+ys/2),max(0,(xc-xs/2)):min(np.size(data,1),xc+xs/2),max(0,(tc-ts/2)):min(np.size(data,2),max(0,(tc-ts/2))+duration)];           
    #erase footprint from footmask so that it does not influence future centroid calcuations
    data[max(0,(yc-ys/2)):min(np.size(data,0),yc+ys/2),max(0,(xc-xs/2)):min(np.size(data,1),xc+xs/2),max(0,(tc-ts/2)):min(np.size(data,2),tc-ts/2+duration)] -= \
        footprint[(ys/2-(yc-max(0,(yc-ys/2)))):(ys/2-(yc-min(np.size(data,0),yc+ys/2))),(xs/2-(xc-max(0,(xc-xs/2)))):(xs/2-(xc-min(np.size(data,1),xc+xs/2))),(ts/2-(tc-max(0,(tc-ts/2)))):(ts/2-(tc-min(np.size(data,2),tc-ts/2+duration)))];
    footmask[max(0,(yc-ys/2)):min(np.size(data,0),yc+ys/2),max(0,(xc-xs/2)):min(np.size(data,1),xc+xs/2),max(0,(tc-ts/2)):min(np.size(data,2),tc-ts/2+duration)] -= \
        footprint[(ys/2-(yc-max(0,(yc-ys/2)))):(ys/2-(yc-min(np.size(data,0),yc+ys/2))),(xs/2-(xc-max(0,(xc-xs/2)))):(xs/2-(xc-min(np.size(data,1),xc+xs/2))),(ts/2-(tc-max(0,(tc-ts/2)))):(ts/2-(tc-min(np.size(data,2),tc-ts/2+duration)))] > thresh;
    GRF = np.sum(np.sum(footmask,1),0);    

    if yc < ys/3.0 or yc > np.size(data,0)-ys/3.0 or duration < 5: #or xc < xs/3.0 or xc > np.size(data,1)-xs/3.0: # footprint too close to edge?
        print("!")
        continue; # do not include footprints that are only partially on the tile.            
    else: # keep footprint
        footprints.append(footprint); yclist.append(yc); xclist.append(xc); tclist.append(tc)
        print(".")

sidx = np.argsort(tclist)
print( (np.array(footprints)[sidx].tolist(), np.array(yclist)[sidx], np.array(xclist)[sidx], np.array(tclist)[sidx])   )




hf = h5py.File('/home/pk/Desktop/pradeep/Stepscan/Petrick/data_segment/46/out/46_1', 'r')
n1 = hf.get('dataset_1')
n1 = np.array(n1)
n2=n1[500:800,:,:]


foot_list=[]
for x in d[0]:
    tmp=np.vstack(x)
    foot_list.append(tmp.reshape(80,60,200))
  
#foot orientation
align_foot=[]
for x in foot_list:
    otn=footPrint2DOrientationEllipse(x)#[-xoffset, -yoffset, -rotoffset, scaleoffset, fitness]
    align_foot.append(alignFootPrint(x,-otn[0],-otn[1],-otn[2]))#alignFootPrint(footprint, xoffset=0, yoffset=0, rotoffset=0, ):  

plot_segmented_foot(align_foot)
plot_segmented_foot(foot_list)
######################
#plt a grid of the segmented foot images
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
    #figure()
    #imshow(np.sum(foot[0],0))
################
#Extract Features
    #1. basic foot features ([toeout, footlength, footwidth, footarea, duration])
    #2. Center of pressure for every footprint
    #3. extract the "M"-shaped vertical ground reaction force curve
    #4. extract basic GRF statistical and keypoint features [meanGRF, stdGRF, peaks[0], smoothGRF[peaks[0]], peaks[1], smoothGRF[peaks[1]], trough[0], smoothGRF[trough[0]]]
    #5. computes 2D image features of various kinds- mean, max, duration, PTI, etc.
    #6. [cadence, velocity, stepWidth, stepLengthL, stepLengthR, strideLengthL, strideLengthR]
##############
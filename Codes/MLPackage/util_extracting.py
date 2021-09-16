from matplotlib.mlab import find
import h5py
from scipy import linalg
from PIL import Image;
import numpy as np
import pylab as pl
# from numpy.fft.fftpack import fft2, ifft2
# from logpolar import logpolar
from numpy import sum
from scipy.signal import convolve2d

def isFootprintUp(foot):
    foot = np.array(foot)    
    fshape = np.shape(foot);    
    tc = np.sum(np.sum(foot,(0,1))/float(np.sum(foot))*np.arange(fshape[2]));   
    #tc = (find(np.sum(np.sum(foot,1),0) < 500).tolist() + [foot.shape[2]])[0]/2;
    footEarly = np.sum(foot[:,:,:tc],2);
    footLate = np.sum(foot[:,:,tc:],2);
    ycEarly = sum(sum(footEarly*np.reshape(np.arange(fshape[0]),[fshape[0],1])*np.ones(fshape[1]),0))/sum(footEarly+1e-20)
    ycLate = sum(sum(footLate*np.reshape(np.arange(fshape[0]),[fshape[0],1])*np.ones(fshape[1]),0))/sum(footLate+1e-20)
    return ycEarly > ycLate;

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
            
def extractFootPrints(data):     
    # determine central time of each footprint by finding the times in which they put most pressure to one side or the other
    data[:,:,:] = 1*data[:,:,:] # make a copy
    thresh = 20; # threshold above which pressures are considered to be real
    dProfile = np.sum(data, (0,1));    
    noFPMask = dProfile < np.mean(dProfile); # mostly frames that do not contain footprints
    background = (np.sum((data > 0)*noFPMask,2)/float(np.sum(noFPMask))) > 0.01;        
    data = (data.transpose(2,0,1)*(1-background)).transpose(1,2,0);        
    datathresh = 100; #np.percentile(np.sum(data,(0,1)),25) + 500;   
    footmask = data[:,:,:] > thresh;    

    peaks = []; # find the peaks in the masksum curve to identify the central times in which feet are planted 

    ys = 80; xs = 60; ts = 200; #the dimensions of a footprint "box" maximum size of footprint box. 
    footprints = []; yclist = []; xclist = []; tclist = [];
    GRF = np.sum(np.sum(footmask,1),0);    
    iterI = -1;    
    while(find((GRF[:-1] < 50)*(GRF[1:] >= 50)).size > 0): # All it does take GRF shift back 1 timestamp find less 50 
        
        iterI += 1;        
        tc = find((GRF[:-1] < 50)*(GRF[1:] >= 50))[0];
        tslice = footmask[:,:,max(0,tc-ts/2):min(np.size(footmask,2),tc+ts/2)];        
        flat = np.sum(tslice,2) > 0   
        profile = np.sum(flat,1)        
        walkUp = isFootprintUp(tslice)
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
            tfp[(ys/2-(yc-max(0,(yc-ys/2)))):(ys/2-(yc-min(np.size(data,0),yc+ys/2))),(xs/2-(xc-max(0,(xc-xs/2)))):(xs/2-(xc-min(np.size(data,1),xc+xs/2))),(ts/2-(tc-max(0,(tc-ts/2)))):(ts/2-(tc-min(np.size(data,2),tc+ts/2)))] = \
                data[max(0,(yc-ys/2)):min(np.size(data,0),yc+ys/2),max(0,(xc-xs/2)):min(np.size(data,1),xc+xs/2),max(0,(tc-ts/2)):min(np.size(data,2),tc+ts/2)];
            if sum(sum(sum(tfp,0),0) > datathresh) == 0:
                continue; #there is no footprint, it's just picking up noise.
            flat = np.amax(tfp,2) > thresh
    
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
    return (np.array(footprints)[sidx].tolist(), np.array(yclist)[sidx], np.array(xclist)[sidx], np.array(tclist)[sidx]);        

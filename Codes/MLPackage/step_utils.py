# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:16:12 2015

@author: Patrick
"""
#from matplotlib.mlab import find
import h5py, sys
from scipy import linalg
from PIL import Image;
import numpy as np
import pylab as pl
from numpy.fft import fft2, ifft2
#from logpolar import logpolar
from numpy import sum
from scipy.signal import convolve2d

def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

def createDatabase(ifilename, sizeParams, dtype, compression=None):
    f = h5py.File(ifilename,'w');
    f.create_dataset('barefoot/data', [5]+sizeParams[1:], fillvalue=0, compression=compression, maxshape=tuple([None]+sizeParams[1:]), dtype=dtype)
    f.create_dataset('shod_other/data', [5]+sizeParams[1:], fillvalue=0, compression=compression, maxshape=tuple([None]+sizeParams[1:]), dtype=dtype)
    f.create_dataset('shod_common/natural/data', [5]+sizeParams[1:], fillvalue=0, compression=compression, maxshape=tuple([None]+sizeParams[1:]), dtype=dtype)
    f.create_dataset('shod_common/fast/data', [5]+sizeParams[1:], fillvalue=0, compression=compression, maxshape=tuple([None]+sizeParams[1:]), dtype=dtype)
    f.create_dataset('shod_common/slow/data', [5]+sizeParams[1:], fillvalue=0, compression=compression, maxshape=tuple([None]+sizeParams[1:]), dtype=dtype)
    
    # timestamps
    f.create_dataset('barefoot/timestamps', [5, sizeParams[1]], fillvalue=0, maxshape=tuple([None, sizeParams[1]]), dtype='uint64')
    f.create_dataset('shod_other/timestamps', [5, sizeParams[1]], fillvalue=0, maxshape=tuple([None, sizeParams[1]]), dtype='uint64')
    f.create_dataset('shod_common/natural/timestamps', [5, sizeParams[1]], fillvalue=0, maxshape=tuple([None, sizeParams[1]]), dtype='uint64')
    f.create_dataset('shod_common/fast/timestamps', [5, sizeParams[1]], fillvalue=0, maxshape=tuple([None, sizeParams[1]]), dtype='uint64')
    f.create_dataset('shod_common/slow/timestamps', [5, sizeParams[1]], fillvalue=0, maxshape=tuple([None, sizeParams[1]]), dtype='uint64')

    # metadata
    f.create_dataset('barefoot/metadata', [5, sizeParams[0]], fillvalue=-1, maxshape=tuple([None, sizeParams[0]]))
    f.create_dataset('shod_other/metadata', [5, sizeParams[0]], fillvalue=-1, maxshape=tuple([None, sizeParams[0]]))
    f.create_dataset('shod_common/natural/metadata', [5, sizeParams[0]], fillvalue=-1, maxshape=tuple([None, sizeParams[0]]))
    f.create_dataset('shod_common/fast/metadata', [5, sizeParams[0]], fillvalue=-1, maxshape=tuple([None, sizeParams[0]]))
    f.create_dataset('shod_common/slow/metadata', [5, sizeParams[0]], fillvalue=-1, maxshape=tuple([None, sizeParams[0]]))

    # survey metatdata
    f.create_dataset('surveydata', [5, 7], fillvalue=0, chunks=tuple([5, 7]), maxshape=tuple([None, 7]))   
    f.close();

#def createDatabaseTest(ifilename, sizeParams, dtype):
#    f = h5py.File(ifilename,'w');
#    f.create_dataset('barefoot/data', [5]+sizeParams[1:], fillvalue=0, maxshape=tuple([None]+sizeParams[1:]), dtype=dtype)
#    f.create_dataset('shod_other/data', [5]+sizeParams[1:], fillvalue=0, maxshape=tuple([None]+sizeParams[1:]), dtype=dtype)
#    f.create_dataset('shod_common/natural/data', [5]+sizeParams[1:], fillvalue=0, maxshape=tuple([None]+sizeParams[1:]), dtype=dtype)
#    f.create_dataset('shod_common/fast/data', [5]+sizeParams[1:], fillvalue=0, maxshape=tuple([None]+sizeParams[1:]), dtype=dtype)
#    f.create_dataset('shod_common/slow/data', [5]+sizeParams[1:], fillvalue=0, maxshape=tuple([None]+sizeParams[1:]), dtype=dtype)
#    
#    # timestamps
#    f.create_dataset('barefoot/timestamps', [5, sizeParams[-1]], fillvalue=0, maxshape=tuple([None, sizeParams[-1]]), dtype='uint64')
#    f.create_dataset('shod_other/timestamps', [5, sizeParams[-1]], fillvalue=0, maxshape=tuple([None, sizeParams[-1]]), dtype='uint64')
#    f.create_dataset('shod_common/natural/timestamps', [5, sizeParams[-1]], fillvalue=0, maxshape=tuple([None, sizeParams[-1]]), dtype='uint64')
#    f.create_dataset('shod_common/fast/timestamps', [5, sizeParams[-1]], fillvalue=0, maxshape=tuple([None, sizeParams[-1]]), dtype='uint64')
#    f.create_dataset('shod_common/slow/timestamps', [5, sizeParams[-1]], fillvalue=0, maxshape=tuple([None, sizeParams[-1]]), dtype='uint64')
#
#    # metadata
#    f.create_dataset('barefoot/metadata', [5, sizeParams[0]], fillvalue=-1, maxshape=tuple([None, sizeParams[0]]))
#    f.create_dataset('shod_other/metadata', [5, sizeParams[0]], fillvalue=-1, maxshape=tuple([None, sizeParams[0]]))
#    f.create_dataset('shod_common/natural/metadata', [5, sizeParams[0]], fillvalue=-1, maxshape=tuple([None, sizeParams[0]]))
#    f.create_dataset('shod_common/fast/metadata', [5, sizeParams[0]], fillvalue=-1, maxshape=tuple([None, sizeParams[0]]))
#    f.create_dataset('shod_common/slow/metadata', [5, sizeParams[0]], fillvalue=-1, maxshape=tuple([None, sizeParams[0]]))
#
#    # survey metatdata
#    f.create_dataset('surveydata', [5, 7], fillvalue=0, chunks=tuple([5, 7]), maxshape=tuple([None, 7]))   
#    f.close();


def getNextIndexAndResizeAsNecessary(metadata, data, timestamps):
    ids = find(metadata[:,0]==-1);
    if len(ids)==0:
        idx = metadata.shape[0];         
        dsize = list(data.shape); dsize[0] += 5; data.resize(dsize);
        msize = list(metadata.shape); msize[0] += 5; metadata.resize(msize);
        tsize = list(timestamps.shape); tsize[0] += 5; timestamps.resize(tsize);
        return idx
    else:
        return ids[0];

# merge the separate footwear condition databases into one database (and add a metadata flag at position 3 to say which it belonged to) 
def convertToMergedDataset(mergeFilename, dbfilename):
    db = h5py.File(dbfilename,"r+")        
    f = h5py.File(mergeFilename,"w")    
    metashape = list(db.get('barefoot/metadata').shape); metashape[1] += 1;
    data = f.create_dataset('data', [5]+list(db.get('barefoot/data').shape)[1:], maxshape=tuple([None]+list(db.get('barefoot/data').shape)[1:]), dtype=db.get('barefoot/data').dtype)
    metadata = f.create_dataset('metadata', [5]+metashape[1:], maxshape=tuple([None]+metashape[1:]),dtype='float64')
    timestamps = f.create_dataset('timestamps', [5]+list(db.get('barefoot/timestamps').shape)[1:], maxshape=tuple([None]+list(db.get('barefoot/timestamps').shape)[1:]),dtype='uint64')
    nRecs = 0;    
    for iterD in range(5):
        h5datadir = ["barefoot/", "shod_other/", "shod_common/natural/", "shod_common/fast/", "shod_common/slow/"][iterD]
        ndatadirRecs = sum(db.get(h5datadir+'metadata')[:,0] > -1);        
        dsize = list(data.shape); dsize[0] = nRecs+ndatadirRecs; data.resize(dsize); data[nRecs:] = db.get(h5datadir+'data')[:ndatadirRecs]
        dsize = list(metadata.shape); dsize[0] = nRecs+ndatadirRecs; metadata.resize(dsize); 
        metadata[nRecs:,0:3] = db.get(h5datadir+'metadata')[:ndatadirRecs,0:3];
        metadata[nRecs:,3] = iterD;
        if metashape[1] > 4:
            metadata[nRecs:,4:] = db.get(h5datadir+'metadata')[:ndatadirRecs,3:]
        dsize = list(timestamps.shape); dsize[0] = nRecs+ndatadirRecs; timestamps.resize(dsize); timestamps[nRecs:] = db.get(h5datadir+'timestamps')[:ndatadirRecs]
        nRecs+= ndatadirRecs;
        print( ".");
    db.close();
    f.close();                            

def centerFFT(inpfft):
    yn = np.size(inpfft,0); xn = np.size(inpfft,1);
    inpfft_centfft = 0*inpfft; 
    inpfft_centfft[0:(yn/2),0:(xn/2)] = inpfft[(yn/2):(yn),(xn/2):(xn)];
    inpfft_centfft[0:(yn/2),(xn/2):(xn)] = inpfft[(yn/2):(yn),0:(xn/2)];
    inpfft_centfft[(yn/2):(yn),0:(xn/2)] = inpfft[0:(yn/2),(xn/2):(xn)];
    inpfft_centfft[(yn/2):(yn),(xn/2):(xn)] = inpfft[0:(yn/2),0:(xn/2)]; 
    return inpfft_centfft;

def getCOP(footprint):            
    #footprint = oFootprint/mean(sum(sum(oFootprint,0),0))    
    yc = np.zeros(1); xc = np.zeros(1); duration = 1;
    if np.size(np.shape(footprint)) > 2:
        yc = np.zeros(np.size(footprint,2));
        xc = np.zeros(np.size(footprint,2));
        duration = np.size(footprint,2)
    for iterI in range(duration):
        if duration == 1:            
            footframe = footprint;                
        else:            
            footframe = footprint[:,:,iterI];                    
        yc[iterI] = sum(sum(footframe*np.reshape(np.arange(0,np.size(footprint,0)),[np.size(footprint,0),1])*np.ones(np.size(footprint,1)),0))/sum(footframe);
        xc[iterI] = sum(sum(footframe*np.reshape(np.ones(np.size(footprint,0)),[np.size(footprint,0),1])*np.arange(0,np.size(footprint,1)),0))/sum(footframe);

    yc[np.isnan(yc)] = np.size(footprint,0)/2;
    xc[np.isnan(xc)] = np.size(footprint,1)/2;        
    return np.array(yc.tolist() + xc.tolist());

def getAngle(vector): # takes in a vector and outputs an angle (360 degrees)
    vector = vector/np.sqrt(sum(vector**2));
    quad = [vector[0]>=0, vector[1]>=0, vector[0]*vector[1]>=0];
    if quad[0] and quad[1] and quad[2]:
        angle = np.arcsin(vector[0])*180/np.pi;        
    elif quad[0]:
        angle = 180 - np.arcsin(vector[0])*180/np.pi;
    elif quad[1]:
        angle = 360 - np.arccos(vector[1])*180/np.pi;
    elif quad[2]:
        angle = 180 + np.arctan(vector[0]/vector[1])*180/np.pi
    else:
        print( "******************* Angle Not Set *************************")
        print (vector)
        angle = 0;
        #import pdb; pdb.set_trace()        
    return angle;
    
def footPrint2DOrientation(footprint, templates, tempmeta, gsize = 0):
    #flatfoot = np.mean(footprint,2); # should probably make this np.amax() because the template is based on peak pressures
    flatfoot = np.amax(footprint,2); 

    if gsize > 0:
        from scipy.signal import convolve2d
        flatfoot = convolve2d(flatfoot, gauss2d([gsize+1,gsize+1], gsize/2, [gsize/2,gsize/2]), mode='same') # blur the input foot to improve alignment for wierd feet.
    
    tempflat = 1.0*flatfoot;
    adjust = np.array([0,0,0]);
    for iterI in range(10):
        scores = np.sum(templates*tempflat, (1,2))
        if np.sum(abs(np.median(tempmeta[np.argsort(scores)[-5:]],0))) == 0:
            return adjust
        #print np.median(tempmeta[np.argsort(scores)[-5:]],0), 
        adjust += np.median(tempmeta[np.argsort(scores)[-5:]],0)        
        tempflat = alignFootPrint(flatfoot.reshape([80,60,1]), xoffset=-adjust[1], yoffset=-adjust[0], rotoffset=-adjust[2], ).reshape([80,60])    
    return adjust
    
def footPrint2DOrientationEllipse(footprint):
    maxfoot = np.amax(footprint,2);
    footarea = np.sum(footprint > 0.2*np.percentile(maxfoot, 80),2);
    [yoffset, xoffset] = getCOP(footarea);
    yoffset = int(yoffset + 0.5 - footarea.shape[0]/2);    
    xoffset = int(xoffset + 0.5 - footarea.shape[1]/2);    
    positions = np.zeros([np.sum(footarea>0),2]);    
    positions[:,0] = (np.reshape(np.arange(0,np.size(footarea,0)),[np.size(footarea,0),1])*np.ones(np.size(footarea,1)))[footarea>0];
    positions[:,1] = (np.reshape(np.ones(np.size(footarea,0)),[np.size(footarea,0),1])*np.arange(0,np.size(footarea,1)))[footarea>0];
    D, V = linalg.eig(np.cov(positions.T))    
    rotoffset = int(90 - getAngle(V[np.argmax(D)])+0.5)
    scaleoffset = 0;  fitness = 0; # keep these for compatibility with previous method
    return [-xoffset, -yoffset, -rotoffset, scaleoffset, fitness];

def footPrint2DOrientationPatakyEtAl(footprint, footTemplateIdx, bRot180=False):
    # load the appropriate template to compare    
    #ftemp = np.load('bareshodtemp.npy')
    ftemp = np.load('MUN104LBlur.npy')

#    if footTemplateIdx == 0 or footTemplateIdx == 1: # left barefoot
#        ftemp = np.load('MUN104FootTemplates.npy')[footTemplateIdx];        
#    elif footTemplateIdx == 2 or footTemplateIdx == 3: # left or right shod foot
#        print "s",        
#        imftemp = Image.open('shodAlignmentTemplate.png')
#        ftemp = np.array(imftemp.getdata())[:,2].reshape(imftemp.size[1], imftemp.size[0]);

    # compute the FFT, Centered FFT and Log-polar Centered FFT of the foot template
    logsize = [180, 180];
    center = [np.size(footprint,0)/2, np.size(footprint,1)/2];
    ftemp_fft = fft2(ftemp);# Take the FFT
    ftemp_centPS = centerFFT(ftemp_fft*ftemp_fft.conjugate()); # compute the centered frequency power spectrum
    ftemp_LPT_centPS = logpolar(ftemp_centPS, center[0], center[1], logsize[0], logsize[1]); 
    
    # compute the FFT, Centered FFT and Log-polar Centered FFT of the footprint to be aligned
    if len(footprint.shape) == 2:
        amaxfoot = footprint;
    else:
        amaxfoot = np.amax(footprint,2); 
    maxfootprint = np.rot90(amaxfoot,2*bRot180); 
    maxfootprint = maxfootprint*255.0/np.amax(np.amax(maxfootprint));    
    footprint_fft = fft2(maxfootprint); 
    footprint_centPS = centerFFT(footprint_fft*footprint_fft.conjugate());
    footprint_LPT_centPS = logpolar(footprint_centPS, center[0], center[1], logsize[0], logsize[1]);     
    
    ## determine rotation and scale offsets and transform the original image accordingly
    ftemp_LPT_centPS_fft = fft2(ftemp_LPT_centPS);
    footprint_LPT_centPS_fft = fft2(footprint_LPT_centPS);    
    conv = centerFFT(ifft2(ftemp_LPT_centPS_fft.conjugate()*footprint_LPT_centPS_fft));
    nrows = np.size(footprint_LPT_centPS_fft,0); 
    ncols = np.size(footprint_LPT_centPS_fft,1);
    conv[0:nrows,(ncols/4-int(ncols/5.5)):(ncols/4+int(ncols/5.5))] = 0; #prevent extreme rotations, because nothing should be rotated extremely.
    conv[0:nrows,(3*ncols/4-int(ncols/5.5)):(3*ncols/4+int(ncols/5.5))] = 0;
    idx = np.argmax(np.transpose(conv)); 
    scaleoffset = np.exp((nrows/2 - (idx % nrows))*np.real(np.log((nrows ** 2 + ncols ** 2) ** 0.5)/nrows));
    rottemp = (int(idx/nrows)) % (ncols/2)
    rotoffset = ((rottemp > (ncols/4))*(ncols/2-rottemp) + (rottemp <= (ncols/4))*(-rottemp))* 360/ncols;
    imfoot = Image.fromarray(np.uint8(maxfootprint))
    imfoot = imfoot.rotate(-rotoffset,Image.BICUBIC).resize((int(np.size(footprint,1)/scaleoffset),int(np.size(footprint,0)/scaleoffset)),Image.BICUBIC) #rotate and resize image
    imfoot = imfoot.crop(((imfoot.size[0]-np.size(footprint,1))/2, (imfoot.size[1]-np.size(footprint,0))/2,(imfoot.size[0]+np.size(footprint,1))/2, (imfoot.size[1]+np.size(footprint,0))/2));
    footprint_rescaled_rotated = np.array(imfoot.getdata()).reshape(imfoot.size[1], imfoot.size[0])
    
    #compute the translation offset and adjust the image accordingly
    ftemp_fft = fft2(ftemp); 
    footprint_fft = fft2(footprint_rescaled_rotated);
    conv = centerFFT(ifft2(ftemp_fft.conjugate()*footprint_fft));
    idx = np.argmax(np.transpose(conv)); 
    nrows = np.size(ftemp_fft,0); # extract features

    ncols = np.size(ftemp_fft,1);
    yoffset = (idx % nrows) - nrows/2; 
    xoffset = (int(idx/nrows)) - ncols/2;
    imfoot = imfoot.offset(-xoffset, -yoffset);
    footprint_rescaled_rotated_translated = np.array(imfoot.getdata()).reshape(imfoot.size[1], imfoot.size[0]);
    
    #compute match score (cosine of angle between template and adjusted input)
    fitness = 0.5*sum(ftemp*footprint_rescaled_rotated_translated)/np.sqrt(sum(ftemp**2))/np.sqrt(sum(footprint_rescaled_rotated_translated**2)) + \
              0.5*sum((ftemp>0)*(footprint_rescaled_rotated_translated>0))/np.sqrt(sum((ftemp>0)**2))/np.sqrt(sum((footprint_rescaled_rotated_translated>0)**2)); # use binarized images rather than real values to capture overlap or shape better
    
    return [xoffset, yoffset, rotoffset, scaleoffset, fitness];

# expects a closed polygon (i.e., a triangle will have four points, the first and last the same)
# points are given in (x, y) format - an (y, x format will give the opposite result)
def isClockwise(points):  
    esum = 0;    
    for iterI in range(len(points)-1):
        esum += (points[iterI+1][0] - points[iterI][0])*(points[iterI+1][1] + points[iterI][1])
    return esum > 0;
    
def rotate2DPoints(posX, posY, center, theta):
    """Rotates the given polygon which consists of corners represented as (x,y),
    around the ORIGIN, clock-wise, theta degrees"""
    theta = np.math.radians(theta)   
    
    pX = posX - center[0]; pY = posY - center[1];    
    rpX = 0.0*pX; rpY = 0.0*pY;
    for iterP in range(0, np.size(pX)):
        rpX[iterP] = pX[iterP]*np.math.cos(theta) - pY[iterP]*np.math.sin(theta)
        rpY[iterP] = pX[iterP]*np.math.sin(theta) + pY[iterP]*np.math.cos(theta)
        
    rpX +=center[0]; rpY +=center[1];

    return [rpX, rpY]; 

def get2DPoints(nx, ny, cover): #cover = [y,x]
    gridx = (np.array(np.arange(0, nx*ny)) % nx)/np.double(nx-1)*cover[1];
    gridy = np.floor(np.arange(0, nx*ny) / np.double(nx))/np.double(ny-1)*cover[0];    
    return [gridx, gridy];

def alignFootPrint(footprint, xoffset=0, yoffset=0, rotoffset=0, ):    
    #find initial heel strike and make this the first frame
    footprint = np.array(footprint);    
    frameswithdata = find(np.sum(np.sum(footprint,0),0)>0);    

    # Rotation (alignment) of footprint
    alignedFootPrint = 0*np.array(footprint);
    gridx = np.array(range(0, np.size(footprint,0)*np.size(footprint,1))) % np.size(footprint,1);
    gridy = np.floor(np.arange(0, np.size(footprint,0)*np.size(footprint,1)) / np.size(footprint,1)).astype('int');
    [rgridx, rgridy] = rotate2DPoints(gridx, gridy, [np.size(footprint,1)/2, np.size(footprint,0)/2], rotoffset)                
    for frame in range(frameswithdata[0], frameswithdata[-1]+1):
        # Nearest Neighbours #rotate:2126
        rgridz = np.zeros(np.size(footprint,0)*np.size(footprint,1));        
        rdist = -1.0 + np.zeros(np.size(footprint,0)*np.size(footprint,1));        
        for iterP in range(0,np.size(footprint,0)*np.size(footprint,1)):        
            if np.floor(rgridx[iterP]) >= 0 and np.ceil(rgridx[iterP]) < np.size(footprint,1) and np.floor(rgridy[iterP]) >= 0 and np.ceil(rgridy[iterP]) < np.size(footprint,0):
                yidx = int(np.ceil(rgridy[iterP])); xidx = int(np.floor(rgridx[iterP]))               
                if rdist[iterP] == -1.0 or rdist[iterP] > np.sqrt((rgridy[iterP]-yidx)**2 + (rgridx[iterP]-xidx)**2):
                    rgridz[iterP] = footprint[yidx ,xidx, frame];
                yidx = int(np.ceil(rgridy[iterP])); xidx = int(np.ceil(rgridx[iterP]))               
                if rdist[iterP] == -1.0 or rdist[iterP] > np.sqrt((rgridy[iterP]-yidx)**2 + (rgridx[iterP]-xidx)**2):
                    rgridz[iterP] = footprint[yidx ,xidx, frame];
                yidx = int(np.floor(rgridy[iterP])); xidx = int(np.floor(rgridx[iterP]))               
                if rdist[iterP] == -1.0 or rdist[iterP] > np.sqrt((rgridy[iterP]-yidx)**2 + (rgridx[iterP]-xidx)**2):
                    rgridz[iterP] = footprint[yidx ,xidx, frame];
                yidx = int(np.floor(rgridy[iterP])); xidx = int(np.ceil(rgridx[iterP]))               
                if rdist[iterP] == -1.0 or rdist[iterP] > np.sqrt((rgridy[iterP]-yidx)**2 + (rgridx[iterP]-xidx)**2):
                    rgridz[iterP] = footprint[yidx ,xidx, frame];
            else:
                rgridz[iterP] = 0;
        alignedFootPrint[:,:,frame] = rgridz.reshape(np.size(footprint,0), np.size(footprint,1));
    alignedFootPrint = np.roll(alignedFootPrint, xoffset, axis=1);
    alignedFootPrint = np.roll(alignedFootPrint, yoffset, axis=0);
   
    return alignedFootPrint;

def extractFootPrints(data1):     
    # determine central time of each footprint by finding the times in which they put most pressure to one side or the other
    data = 1*data1[:,:,:].transpose(0,2,1) # make a copy
    thresh = 20; # threshold above which pressures are considered to be real
    dProfile = np.sum(data, (0,1));  
    

    noFPMask = dProfile < np.mean(dProfile); # mostly frames that do not contain footprints
    background = (np.sum((data > 0)*noFPMask,2)/float(np.sum(noFPMask))) > 0.01;   
    data = (data.transpose(2,0,1)*(1-background)).transpose(1,2,0);  

    datathresh = 100; #np.percentile(np.sum(data,(0,1)),25) + 500;   
    footmask = data[:,:,:] > thresh;

    ys = 80; xs = 60; ts = 200; #the dimensions of a footprint "box" 
    footprints = []; yclist = []; xclist = []; tclist = [];
    GRF = np.sum(np.sum(footmask,1),0);    
    iterI = -1; 


    # print(np.transpose(np.nonzero(data < 0)))
    # print(GRF) 
    # print(GRF.shape) 
    # print((((GRF[:-1] < 50)))) 
    # print((((GRF[1:] >= 50)))) 
    # print((((GRF[:-1] < 50)*(GRF[1:] >= 50)))) 
    # print((find((GRF[:-1] < 50)*(GRF[1:] >= 50)))) 
    # print((find((GRF[:-1] < 50)*(GRF[1:] >= 50))).shape) 
    # print((find((GRF[:-1] < 50)*(GRF[1:] >= 50))).size) 
    # # print(np.sum((data > 0)*noFPMask,2)) 
    # # print(((data > 0)*noFPMask).shape) 
    # # print(np.mean(dProfile)) 
    # plt.plot(range(0,GRF.shape[0]), GRF)
    # plt.figure()
    # plt.plot(range(0,dProfile.shape[0]), noFPMask)

    # plt.show()
    # sys.exit() 
        

    peaks = []; # find the peaks in the masksum curve to identify the central times in which feet are planted 

       
    while(find((GRF[:-1] < 50)*(GRF[1:] >= 50)).size > 0):
        
        iterI += 1;        
        tc = find((GRF[:-1] < 50)*(GRF[1:] >= 50))[0]
        
        tslice = footmask[:,:,int(max(0,tc-ts/2)):int(min(np.size(footmask,2),tc+ts/2))];        
        flat = np.sum(tslice,2) > 0   
        
        profile = np.sum(flat,1)  
        # print(flat.shape)
        # print(profile.shape)
        # sys.exit()      
        walkUp = isFootprintUp(tslice);
        if walkUp:      
            footend = maxCC1D(profile)[1];
            flatmask = flat*0; flatmask[max(footend-ys,0):footend] = 1; 
        else:
            footend = maxCC1D(profile[::-1])[1];
            flatmask = flat*0; flatmask[(tslice.shape[0]-footend):min(tslice.shape[0],tslice.shape[0]-footend+ys)] = 1; 
        flat = flat*flatmask; # mask out all footprints other than the one we're focusing on.               
        yc = int(np.sum(np.sum(flat*np.reshape(np.arange(0,np.size(footmask,0)),[np.size(footmask,0),1])*np.ones(np.size(footmask,1)),0))/np.sum(flat))
        xc = int(np.sum(np.sum(flat*np.reshape(np.ones(np.size(footmask,0)),[np.size(footmask,0),1])*np.arange(0,np.size(footmask,1)),0))/np.sum(flat))
        for iterR in range(3):                        
            tfp = np.zeros([ys, xs, ts]);
            tfp[int(ys/2-(yc-max(0,(yc-ys/2)))):int(ys/2-(yc-min(np.size(data,0),yc+ys/2))),int(xs/2-(xc-max(0,(xc-xs/2)))):int(xs/2-(xc-min(np.size(data,1),xc+xs/2))),int(ts/2-(tc-max(0,(tc-ts/2)))):int(ts/2-(tc-min(np.size(data,2),tc+ts/2)))] = \
                data[int(max(0,(yc-ys/2))):int(min(np.size(data,0),yc+ys/2)),int(max(0,(xc-xs/2))):int(min(np.size(data,1),xc+xs/2)),int(max(0,(tc-ts/2))):int(min(np.size(data,2),tc+ts/2))];
            if sum(sum(sum(tfp,0),0) > datathresh) == 0:
                continue; #there is no footprint, it's just picking up noise.
            flat = np.amax(tfp,2) > thresh;
    
            # move the volume to better capture the footprint
            footSlices = find(sum(sum(tfp,0),0) > datathresh).tolist(); footSlices.append(220); #footSlices.append(240)            
            tc += footSlices[0]; # fixing tc so that the heelstrike occurs at beginning
            yc += int(np.sum(np.sum(flat*np.reshape(np.arange(0,np.size(flat,0)),[np.size(flat,0),1])*np.ones(np.size(flat,1)),0))/np.sum(flat)) - ys/2;            
            xc += int(np.sum(np.sum(flat*np.reshape(np.ones(np.size(flat,0)),[np.size(flat,0),1])*np.arange(0,np.size(flat,1)),0))/np.sum(flat)) - xs/2;
            duration = footSlices[find(np.diff(footSlices) > 15)[0]] - footSlices[0] + 1;#find(np.diff(footSlices) == 1)[-1]+1 - find(np.diff(footSlices) == 1)[0]
            #duration = find(np.diff(footSlices) > 10)[0] + 1;# find the end of the footprint        
            #tslice = footmask[max(0,(yc-ys/2)):min(np.size(data,0),yc+ys/2),max(0,(xc-xs/2)):min(np.size(data,1),xc+xs/2),max(0,tc-ts/2):min(np.size(footmask,2),tc-ts/2+duration)];        
            
        if duration == 200:
            print( '*************** Time slice is not big enough to contain the full foot *************** START');
            np.sum(np.sum(tfp,1),0)
                
        #grab footprint and append to list of footprints
        footprint = np.zeros([ys, xs, ts],dtype=int)
        # print(int(ys/2-(yc-max(0,(yc-ys/2)))))
        # print(int(ys/2-(yc-min(np.size(data,0),yc+ys/2))))
        # print(int(xs/2-(xc-max(0,(xc-xs/2)))))
        # print(int(xs/2-(xc-min(np.size(data,1),xc+xs/2))))
        # print(int(ts/2-(tc-max(0,(tc-ts/2)))))
        # print(int(ts/2-(tc-min(np.size(data,2),tc-ts/2+duration))))
        footprint[int(ys/2-(yc-max(0,(yc-ys/2)))):int(ys/2-(yc-min(np.size(data,0),yc+ys/2))),int(xs/2-(xc-max(0,(xc-xs/2)))):int(xs/2-(xc-min(np.size(data,1),xc+xs/2))),int(ts/2-(tc-max(0,(tc-ts/2)))):int(ts/2-(tc-min(np.size(data,2),tc-ts/2+duration)))] = \
            data[int(max(0,(yc-ys/2))):int(min(np.size(data,0),yc+ys/2)),int(max(0,(xc-xs/2))):int(min(np.size(data,1),xc+xs/2)),int(max(0,(tc-ts/2))):int(min(np.size(data,2),max(0,(tc-ts/2))+duration))];           
        #erase footprint from footmask so that it does not influence future centroid calcuations
        #tmp=data[max(0,(yc-ys/2)):min(np.size(data,0),yc+ys/2),max(0,(xc-xs/2)):min(np.size(data,1),xc+xs/2),max(0,(tc-ts/2)):min(np.size(data,2),tc-ts/2+duration)]
        data[int(max(0,(yc-ys/2))):int(min(np.size(data,0),yc+ys/2)),int(max(0,(xc-xs/2))):int(min(np.size(data,1),xc+xs/2)),int(max(0,(tc-ts/2))):int(min(np.size(data,2),tc-ts/2+duration))] -= \
            footprint[int(ys/2-(yc-max(0,(yc-ys/2)))):int(ys/2-(yc-min(np.size(data,0),yc+ys/2))),int(xs/2-(xc-max(0,(xc-xs/2)))):int(xs/2-(xc-min(np.size(data,1),xc+xs/2))),int(ts/2-(tc-max(0,(tc-ts/2)))):int(ts/2-(tc-min(np.size(data,2),tc-ts/2+duration)))];
        footmask[int(max(0,(yc-ys/2))):int(min(np.size(data,0),yc+ys/2)),int(max(0,(xc-xs/2))):int(min(np.size(data,1),xc+xs/2)),int(max(0,(tc-ts/2))):int(min(np.size(data,2),tc-ts/2+duration))] =(footmask[int(max(0,(yc-ys/2))):int(min(np.size(data,0),yc+ys/2)),int(max(0,(xc-xs/2))):int(min(np.size(data,1),xc+xs/2)),int(max(0,(tc-ts/2))):int(min(np.size(data,2),tc-ts/2+duration))].astype(np.float32) - \
            (footprint[int(ys/2-(yc-max(0,(yc-ys/2)))):int(ys/2-(yc-min(np.size(data,0),yc+ys/2))),int(xs/2-(xc-max(0,(xc-xs/2)))):int(xs/2-(xc-min(np.size(data,1),xc+xs/2))),int(ts/2-(tc-max(0,(tc-ts/2)))):int(ts/2-(tc-min(np.size(data,2),tc-ts/2+duration)))] > thresh).astype(np.float32)).astype(np.bool)
        GRF = np.sum(np.sum(footmask,1),0);    

        if yc < ys/3.0 or yc > np.size(data,0)-ys/3.0 or duration < 5: #or xc < xs/3.0 or xc > np.size(data,1)-xs/3.0: # footprint too close to edge?
            print ("!") 
            continue; # do not include footprints that are only partially on the tile.            
        else: # keep footprint
            footprints.append(footprint); yclist.append(yc); xclist.append(xc); tclist.append(tc);
            print( ".")

        

    sidx = np.argsort(tclist)
    return (np.array(footprints)[sidx].tolist(), np.array(yclist)[sidx], np.array(xclist)[sidx], np.array(tclist)[sidx]);        

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

def computeCOPx(volume):
    fshape = list(volume.shape);    
    nFrames = fshape[2]
    xc = np.zeros(nFrames);
    for iterF in range(nFrames): # for each frame in a single footprint
        footframe = volume[:,:,iterF];        
        xc[iterF] = np.sum(np.sum(footframe*np.reshape(np.ones(fshape[0]),[fshape[0],1])*np.arange(0,fshape[1]),0))/(np.sum(footframe)+1e-20); # compute the horizontal pressure center        
    return xc
    
def isFootprintUp(foot):
    foot = np.array(foot)    
    fshape = np.shape(foot);    
    tc = np.sum(np.sum(foot,(0,1))/float(np.sum(foot))*np.arange(fshape[2]));   
    #tc = (find(np.sum(np.sum(foot,1),0) < 500).tolist() + [foot.shape[2]])[0]/2;
    footEarly = np.sum(foot[:,:,:int(tc)],2);
    footLate = np.sum(foot[:,:,int(tc):],2);
    ycEarly = sum(sum(footEarly*np.reshape(np.arange(fshape[0]),[fshape[0],1])*np.ones(fshape[1]),0))/sum(footEarly+1e-20)
    ycLate = sum(sum(footLate*np.reshape(np.arange(fshape[0]),[fshape[0],1])*np.ones(fshape[1]),0))/sum(footLate+1e-20)
    return ycEarly > ycLate;
    
def gaussC(x, y, sigma, center): # Gaussian
    xc = center[0];
    yc = center[1];
    exponent = ((x-xc)**2 + (y-yc)**2)/(2.0*sigma);
    #exponent = (abs(x-xc) + abs(y-yc))/(2.0*sigma);
    return 4*(np.exp(-exponent));   

def gauss2d(gsize, sigma, center):
    R, C = np.meshgrid(np.arange(gsize[0]), np.arange(gsize[1]));
    return gaussC(R, C, sigma, center);
    
def antidispersion(shodimage, bareimage, nIters=10):
    shodimage = shodimage/np.sum(shodimage)
    bareimage = bareimage/np.sum(bareimage)   
    currdiff = shodimage-bareimage

    #distmask = np.exp(-1.0*np.arange(0.0,50.0,0.2)**2)
    gsize = 20; dcent = gsize/2;
    distmask = np.tile(gauss2d([gsize,gsize], dcent, [dcent,dcent]),[4,1,1]);
    distmask[0, dcent:]=0; distmask[1, :dcent]=0; distmask[2, :,dcent:]=0; distmask[3, :,:dcent]=0; #up, down, left, right
    
    nextdiff = 0*currdiff;
    moves = np.zeros([4, shodimage.shape[0], shodimage.shape[1]])
    pct = 1.0
    
    # perform the dispersion    
    for iterI in range(nIters-1):
        moves[0] = convolve2d(currdiff*(currdiff < 0), distmask[0], mode="same");
        moves[1] = convolve2d(currdiff*(currdiff < 0), distmask[1], mode="same");
        moves[2] = convolve2d(currdiff*(currdiff < 0), distmask[2], mode="same");
        moves[3] = convolve2d(currdiff*(currdiff < 0), distmask[3], mode="same");
        moves = moves*(currdiff>0)/(np.sum(moves,0)+1e-100); #normalize and use only moves for positive currdiff elements
        nextdiff = 0*currdiff;
        nextdiff[1:] += moves[0,:-1]*currdiff[:-1];
        nextdiff[:-1] += moves[1,1:]*currdiff[1:];
        nextdiff[:, 1:] += moves[2, :, :-1]*currdiff[:, :-1];
        nextdiff[:, :-1] += moves[3, :, 1:]*currdiff[:, 1:];
        nextdiff = nextdiff*pct + currdiff*(currdiff > 0)*(1-pct) + currdiff*(currdiff < 0);
        currdiff = nextdiff;
    return currdiff+bareimage;

#pradeep
import matplotlib.pyplot as plt
import matplotlib.animation as animation
   
def videoplot(walk):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    im1 = ax1.imshow(walk[20])#,cmap='gist_gray_r', vmin=0, vmax=1)
    xdata=[];ydata=[]
    t=np.arange(3000)
    im2, = ax2.plot(t[0],np.sum(walk[20]), lw=2)
    
    
    def init():
        im1.set_data(np.zeros((720, 720)))
        ax2.set_ylim(0, 40000)
        ax2.set_xlim(0, 2999)
        del xdata[:]
        del ydata[:]
        im2.set_data(xdata, ydata)
        im2.set_data(xdata,ydata)
        
    def updatefig(i):   
        im1.set_data(walk[i])
        xdata.append(t[i]);ydata.append(np.sum(walk[i]))
        im2.set_data(xdata,ydata)
       
       
        return im1,im2
    
    ani = animation.FuncAnimation(fig, updatefig,init_func=init, frames=3000,interval=50)
    plt.show()
    return;
    
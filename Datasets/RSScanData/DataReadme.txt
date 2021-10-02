README:

The data provided here is pressure mat data only from the CASIA Database D.  It comes from a RSScan Footscan device, which is approximately 0.5m by 2m? (255 by 64 pressure sensors). The data is provided in a Python readable format (.npz, .npy). 

This dataset may be used freely for academic research.  Anyone seeking to use it for commercial purposes must first contact Shuai (Kyle) Zheng (szhengcvpr@gmail.com).

If you use this dataset in a publication, please kindly cite the following papers:

Shuai Zheng, Kaiqi Huang, Tieniu Tan, Dacheng Tao. A cascade fusion scheme for gait and cumulative foot pressure image recognition. Pattern Recognition. pp. 3603-3610. 45(10). 2012.

Shuai Zheng, Kaiqi Huang, Tieniu Tan. Evaluation framework on translation-invariant representation for cumulative foot pressure image. International Conference on Image Processing. 2011.

Patrick Connor.  Combining underfoot pressure features for shod and unshod gait biometrics.  (Submitted)


There are two datasets: 
  - 1) barefoot only, having 99 subjects (96 with at least 10 recording trials of about 3 barefoot steps or one gait cycle each) and approximately 2900 footsteps 
  - 2) shod data with corresponding subject barefoot data (15 subjects with 10 barefoot trials, 15 subjects with one shoe type, 13 subjects with a second shoe type) for a total of about 1300 footsteps.  Note that shoe types vary from individual to individual (I think there's even an example of high heels).

There are three primary formats:
  - 1) Trial-wise: 255 (y) by 64 (x) by 249 (time) volumes showing the pressures on the full mat for a 2.49 s period. The metadata for each trial has the following format: [subject ID, weight, shoe size].  These datasets include both gait and balance trials (I tried separating them, but was not entirely successful, so I kept them together).  The only post processing I did was to zero column 64 (i.e., x=64), which is on the far right of the sensor, because it seems that these values were consistently high (perhaps a defect/feature of the pressure mat).
  - 2) Foot-wise: 60 (y) by 40 (x) by 100 (time) volumes showing pressures for a single foot, each extracted from the trial-wise data over a 1.00 s period.  The footsteps are roughly geometrically centered (x,y) and translated in time to begin at time 0 (i.e., data[data.files[i]][:,:,0]).  The metadata for each footprint has the following format:  [subject ID, left(0)/right(1) foot classification, foot index in gait cycle, partial, y center-offset, x center--offset, time center-offset].  Besides the subject's ID number, we provide a classification of the foot as either left or right, the index or order of the feet in the trial, a flag saying when the footstep is incomplete because it did not fall fully on the sensor space, and the center of the foot with respect to the trial-wise recording space (allows determining distance between consecutive footsteps).  With each foot-wise dataset comes a folder containing all of the cumulative (max) footprints in jpeg format for easy viewing.  Each jpeg has a filename with the following format:  (0-Left foot, 1-Right foot)_(SubjectID)_(total Trials Processed Thus Far)_(0-first,1-second, or 2-third foot in the trial sequence)_(i, the corresponding index in the per foot dataset).  Note that the data has been manually curated to identify partial footprints.  This allows one to keep or throw out these data points as appropriate.  The barefoot prints in the barefoot-only and shod datsets  has also been manually curated to ensure the left-right classification is correct.  The shod print data is much harder to curate for left-right classification and therefore was not done.  Instead, the left-right flag in the metadata is set to 2 for footwear type 1 or 3 for footwear type 2.
  - 3) Aligned Foot-wise:  same as foot-wise, except that the feet are aligned (translation and rotation) to a specific foot template.  For the barefoot dataset, we use left and right "healthy foot templates" (average of 104 healthy subjects from the Munster dataset, http://fiber.shinshu-u.ac.jp/tpataky/datasets.html).  Because the shod data is not easily curated, we do not have strong confidence in the left-right classification and thus a specialized shoe template is used to align these data, which is also provided.  The metadata for the unaligned footwise data is relevant to these datasets as well.  However, we also provide alignment metadata.  The first three columns consist of the x (in pixels), y (in pixels), and rotational (in degrees) offsets used.  Rotational correction is restricted to between -24 and 24 degrees and has a resolution of 2 degrees.  Rotation uses an accurate implementation of nearest neighbours interpolation (was found to distort less than bicubic and bilinear).  The last two columns of the alignment metadata are: the relative scale of the foot to the template (1.0 being same size) and the fitness score (1.0 being a perfect match). The aligned data is also stretched in time to fit the full 100 time steps (also using nearest neighbour interpolation).  There is also a folder of cumulative (max) footprint images for each, with filename format:  (left(0), right(1), first footwear (2), second footwear (3))_(data index, i).  Please note that this aligned data is NOT normalized for apparent mass.

The data itself is large when unzipped (up to 5GB).  The pressure values are between 0 and 255.  Thus, they do not reflect actual pressure values, but rather some multiple thereof (I believe that the ranges are the same across trials and subjects, but I am not certain).  There are several zip files of data, each containing a dataset and its metadata:
perTrialDataBarefoot.zip 
perTrialDataShod.zip
perFootDataBarefoot.zip
perFootDataShod.zip
alignedPerFootDataBarefoot.zip
alignedPerFootDataShod.zip

Once unzipped, you can load each data file (.npz) in python:
    data = load(‘rawFootScanDataBarefoot.npz’);

To get a specific trial (i), use:
    i = 0;
    triali = data[data.files[i]];

Also, you will find the metadata in appropriately named .npy files.  These can be loaded similarly:
    Metadata = load(‘FootMetaDataBarefoot.npy’);

And accessed:  
   Metadata[i];

For questions about the recording or experimental procedure, please contact Shuai (Kyle) Zheng (szhengcvpr@gmail.com).
For questions about the post processing of the data, as described above, feel free to contact Patrick Connor (patrick.c.connor@gmail.com)




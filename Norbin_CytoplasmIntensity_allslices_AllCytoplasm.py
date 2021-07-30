
from skimage import filters, morphology
import os
import matplotlib.pyplot as pyplot
import glob
import numpy
import scipy
from aicsimageio.writers import OmeTiffWriter
from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects

import tifffile
import matplotlib
import csv
matplotlib.rcParams['pdf.fonttype'] = 42

#####################################
# Declaration of input folder location
#####################################

##channels 0 0 1 2 
#path=os.path.join("D:/Chantelle/201105_RCN-DIV16_shFus-shNorb_75uL virus/Norbin/*/*.lsm")
#path=os.path.expanduser("D:/Chantelle/191213_RCN-DIV16_shFus-Norbin/IFA/*/*.lsm")

##channels 0 0 2 3
#path=os.path.join("D:/Chantelle/190313_RCN_virus_shNorbin_shFus/IFA/*/*.lsm")
#path=os.path.expanduser("D:/Chantelle/Other replicates no MAP2/190403_RCN_virus_shNorbin-shFus_75uL-100uL/488 emission range changed/*/*.lsm")
path=os.path.join("D:/Chantelle/190416_RCN/*/*.lsm")
date='190416'


images = glob.glob(path)
print(len(images))

#####################
structure_channel = 0 #Channel number of Norbin staining
nucleus_channel= 2 #Channel number of DAPI staining
#####################

output_dir=os.path.join(os.getcwd(),"masks_"+date+"_NCDN/")
os.makedirs(output_dir,exist_ok=True)

Fusinthist=[]
Fus5inthist=[]
Norbininthist=[]
Norbin1inthist=[]
PLKOinthist=[]

FusinthistMedian=[]
Fus5inthistMedian=[]
NorbininthistMedian=[]
Norbin1inthistMedian=[]
PLKOinthistMedian=[]

Fusint=[]
Fus5int=[]
Norbinint=[]
PLKOint=[]

namesfus=[]
namesfus5=[]
namesPLKO=[]

numfus=[]
numfus5=[]
numPLKO=[]
numNorbin=[]
numNorbin1=[]

FusFuschintMedian=[]
Fus5FuschintMedian=[]
PLKOFuschintMedian=[]
NorbinFuschintMedian=[]

NorbinFuschinthist = []
PLKOFuschinthist = []
FusFuschinthist = []
Fus5Fuschinthist = []

plko=0
fus=0
fus5=0
norb=0
figcum,axcum=pyplot.subplots(figsize=(12,8))
for i,imagei in enumerate(images):
    outpath = output_dir + os.path.basename(imagei)
    MakeHist=False
    print(os.path.basename(imagei))

    image1=tifffile.imread(imagei).astype(numpy.float32)
    IMG=image1


    X=IMG.shape[3]
    Y=IMG.shape[4]
    N_CHANNELS = IMG.shape[2]
    N_FRAMES = IMG.shape[1]
    
    print('X,Y', X,Y)
    print('n_channels, n_frames', N_CHANNELS,N_FRAMES)

    
    project=numpy.max(IMG[0,:,:,:,:],axis=0)

    project = numpy.moveaxis(project, 0, 2)
    IMG= numpy.moveaxis(IMG[0,:,:,:,:],1,3)

    figtemp=pyplot.figure(figsize=(12, 8))
    pyplot.imshow(project[:,:,0])
    pyplot.show()
   
    foreground_mask=numpy.zeros((N_FRAMES,X,Y))
    ############
    #Nucleus mask
    ############
    dapithresh = filters.threshold_triangle(project[:,:,nucleus_channel])
    dapimask= project[:,:,nucleus_channel] > dapithresh
    dapimask = remove_small_objects(dapimask, 20)
    dapimask = scipy.ndimage.morphology.binary_fill_holes(dapimask)
    dapimask = morphology.binary_dilation(dapimask, selem=morphology.square(3), out=None)
    ############
    #3D Foreground mask
    ############
    for frame in range(N_FRAMES):
        image=IMG[frame,:,:,:]
        
        struct_img0 =image[:,:,structure_channel].copy()

        gaussian_smoothing_sigma = 0.1

        fore_img_smooth = image_smoothing_gaussian_3d( struct_img0, sigma=gaussian_smoothing_sigma)
        foreground_thresh= filters.threshold_triangle(fore_img_smooth.astype('uint16'))

        foreground_struct_img0= struct_img0 .astype('uint16') > foreground_thresh
        foreground_struct_img0= remove_small_objects(foreground_struct_img0, 30)
        foreground_mask[frame,:,:]= foreground_struct_img0
    


    ############
    #Apply foreground and nucleus masks
    ############
    intensity=IMG[:,:, :, structure_channel]*foreground_mask*(1-dapimask)
    inthist = intensity.flatten()

    ############
    # Display foreground mask and save images
    ############
    figtemp=pyplot.figure(figsize=(12, 8))
    pyplot.imshow(numpy.max(foreground_mask,axis=0))
    pyplot.show()
    threeddapimask=dapimask*numpy.ones((N_FRAMES,X,Y))
    channelcombine=numpy.stack((IMG[:,:,:,structure_channel],IMG[:,:,:,nucleus_channel],threeddapimask,foreground_mask,intensity))
    #print(channelcombine.shape)

    writer = OmeTiffWriter(outpath + 'Norbin_Nucleus.tiff',overwrite_file=True)
    writer.save(channelcombine.astype('uint16').reshape((5,N_FRAMES,1024,1024)))

    ############
    # Group intensity be knockdown condition
    ############

    if 'shNorbin02' in os.path.basename(os.path.dirname(imagei)):
        print('Norbin2')

        Norbininthist.extend(inthist[numpy.nonzero(inthist)])
        NorbininthistMedian.append(numpy.median(inthist[numpy.nonzero(inthist)]))


    if 'shNorbin01' in os.path.basename(os.path.dirname(imagei)):
        print('Norbin1')

        Norbin1inthist.extend(inthist[numpy.nonzero(inthist)])
        Norbin1inthistMedian.append(numpy.median(inthist[numpy.nonzero(inthist)]))


        norb+=1

    elif 'shFus318'in os.path.basename(os.path.dirname(imagei)):
        print('Fus8')
        namesfus.append(os.path.basename(imagei))
        Fusinthist.extend(inthist[numpy.nonzero(inthist)])
        FusinthistMedian.append(numpy.median(inthist[numpy.nonzero(inthist)]))



    elif 'shFus315'in os.path.basename(os.path.dirname(imagei)):
        print('Fus5')
        namesfus5.append(os.path.basename(imagei))

        Fus5inthist.extend(inthist[numpy.nonzero(inthist)])
        Fus5inthistMedian.append(numpy.median(inthist[numpy.nonzero(inthist)]))




    elif  'PLKO'  in os.path.basename(os.path.dirname(imagei)):
        print('PLKO')
        namesPLKO.append(os.path.basename(imagei))

        PLKOinthist.extend(inthist[numpy.nonzero(inthist)])
        PLKOinthistMedian.append(numpy.median(inthist[numpy.nonzero(inthist)]))



norm=numpy.median(numpy.array(PLKOinthistMedian))
#####################################################
# Write normalized median intensity to .csv file
###################################################### 
with open(os.path.join(os.getcwd(),"Norbincumcurve","Median", os.path.basename(imagei)+'CytoplasmMedianintensity_normalized.csv'), mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    results_writer.writerow(['shFus318',FusinthistMedian/norm])
    results_writer.writerow(['shFus315',Fus5inthistMedian/norm])
    results_writer.writerow(['shNorbin02',NorbininthistMedian/norm])
    results_writer.writerow(['shNorbin01',Norbin1inthistMedian/norm])
    results_writer.writerow(['PLKO',PLKOinthistMedian/norm])

#####################################################
# Histogram of masked pixels for all images of current folder. Display as cumulative frequency curve
###################################################### 
fig,ax=pyplot.subplots(figsize=(12,8))
countsP,binsP,p=ax.hist(numpy.divide(PLKOinthist,norm),bins=500,density=True,histtype='step',cumulative=True,stacked=True,range=(0,30),label='PLKO',color='blue')
countsF5,binsF5,p=ax.hist(numpy.divide(Fus5inthist,norm),bins=500,density=True,histtype='step',cumulative=True,stacked=True,range=(0,30),label='shFus315',color='green')
countsN,binsN,p=ax.hist(numpy.divide(Norbininthist,norm),bins=500,density=True,histtype='step',cumulative=True,stacked=True,range=(0,30),label='shNorbin02',color='red')
countsN1,binsN1,p=ax.hist(numpy.divide(Norbin1inthist,norm),bins=500,density=True,histtype='step',cumulative=True,stacked=True,range=(0,30),label='shNorbin01',color='brown')
countsF,binsF,p=ax.hist(numpy.divide(Fusinthist,norm),bins=500,density=True,histtype='step',cumulative=True,stacked=True,range=(0,30),label='shFus318',color='orange')
pyplot.plot(numpy.sort(numpy.divide(PLKOinthist,norm)), numpy.linspace(0, 1, len(numpy.divide(PLKOinthist,norm)), endpoint=True),color='blue',label='PLKO')
pyplot.plot(numpy.sort(numpy.divide(Fusinthist,norm)), numpy.linspace(0, 1, len(numpy.divide(Fusinthist,norm)), endpoint=True),color='orange',label='shFus318')
pyplot.plot(numpy.sort(numpy.divide(Fus5inthist,norm)), numpy.linspace(0, 1, len(numpy.divide(Fus5inthist,norm)), endpoint=True),color='green',label='shFus315')

pyplot.plot(numpy.sort(numpy.divide(Norbin1inthist,norm)), numpy.linspace(0, 1, len(numpy.divide(Norbin1inthist,norm)), endpoint=True),color='brown',label='shNorbin01')
pyplot.plot(numpy.sort(numpy.divide(Norbininthist,norm)), numpy.linspace(0, 1, len(numpy.divide(Norbininthist,norm)), endpoint=True),color='red',label='shNorbin02')
ax.legend(loc="upper right")
fig.suptitle('Cytoplasmic intensity')
#ax.set_ylim([0,0.8])
pyplot.savefig(outpath + 'NCDN Cytoplasm intensity_normalized_median.pdf',transparent=True)

#####################################################
# Save Histogram to .npy file
###################################################### 
Dictnorm={'Bins':binsP,'PLKO':countsP,'shNorbin01':countsN1,'shNorbin02':countsN,'shFus318':countsF,'shFus315':countsF5}
numpy.save(output_dir +"Dict"+date+"_normmedian.npy", Dictnorm)

pyplot.show()



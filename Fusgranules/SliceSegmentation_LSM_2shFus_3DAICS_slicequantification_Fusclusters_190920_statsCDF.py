
from skimage import filters,morphology
from skimage.measure import  regionprops, label
import os
import matplotlib.pyplot as pyplot
import glob
import numpy as np
import numpy
from skimage.color import label2rgb

from aicsimageio.writers import OmeTiffWriter
# function for core algorithm
from aicssegmentation.core.seg_dot import  dot_3d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice
from skimage.morphology import remove_small_objects

from scipy.stats import norm

import tifffile


def get_foreground(img,thresh):
    """Gets the foreground of the image using a gaussian blur of
    sigma = 20 and the otsu threshold.

    :param img: A numpy array

    :returns : A binary 2D numpy array of the foreground
    """
    blurred = filters.gaussian(img, sigma=thresh)
    blurred /= blurred.max()
    val = filters.threshold_otsu(blurred)
    return (blurred > val).astype('uint8')


threshint=30000
path2=os.path.expanduser("D:/Chantelle/190920/Slices/*.lsm")
date="190920"
images2=glob.glob(path2)
print(len(images2))

output_dir=os.path.join(os.getcwd(),"masks_"+date+"_FusGranules/")
os.makedirs(output_dir,exist_ok=True)
file='Image1'

inthistFus=[]
inthistNorbin=[]
inthistFus5=[]
inthistPLKO=[]

Norbinint=[]
Fusint=[]
Fusint5=[]
PLKOint=[]

namesfus=[]
namesfus5=[]
namesPLKO=[]
namesnorbin=[]

plko=0
fus=0
fus5=0
norb=0


namesplko=[]
namesfus=[]
namesfus5=[]
namesnorb=[]

numclusterPLKO=[]
numclusterNorbin=[]

for i,imagei in enumerate(images2):
    print(os.path.basename(imagei))
    outpath = output_dir + os.path.basename(imagei)
    #reader = AICSImage(imagei)
    #IMG = reader.data.astype(np.float32)
    image1=tifffile.imread(imagei).astype(np.float32)
    IMG=image1
    
    #################################
    #Remove broken frames from certain stacks
    ##############################


    if '190920_RCN-DIV16_expA_shFus318_05_488-Fus-r_594-MAP2_z0-5.lsm' in imagei:
        print('IMG', IMG.shape)
        IMG= np.delete(IMG,[1,5], axis=1)
        print('IMG', IMG.shape)

    if '190920_RCN-DIV16_expA_shFus318_01_488-Fus-r_594-MAP2_z0-5.lsm' in imagei:
        print('IMG', IMG.shape)
        IMG= np.delete(IMG,[1,2,3], axis=1)
        print('IMG', IMG.shape)

    if '190920_RCN - DIV16_expA_PLKO_04_488-Fus-r_594-MAP2_z0-5.lsm' in imagei:
        print('IMG', IMG.shape)
        IMG= np.delete(IMG,[1], axis=1)
        print('IMG', IMG.shape)

    print(np.min(IMG),np.max(IMG))
    print('IMG', IMG.shape)
    project=numpy.max(IMG[0,:,:,:,:],axis=0)
    print('project', project.shape)

    N_CHANNELS = IMG.shape[2]
    print(' N_CHANNELS ', N_CHANNELS )


    project = numpy.moveaxis(project, 0, 2)



    #####################
    structure_channel = 0
    #####################

    struct_img0 = IMG[0,:,structure_channel, :, :].copy()
    print('struct_img0',struct_img0.shape)


    writer = OmeTiffWriter(outpath + 'OriginalImage.tiff',overwrite_file=True)
    writer.save(struct_img0.astype('uint16'))


    ############
    # Nucleus mask
    ############

    dapithresh = filters.threshold_triangle(project[:,:,1])
    dapimask= project[:,:,1] > dapithresh

    #dapimask=get_foreground(image1[:,:,2].astype('uint8'), 1)
    dapimask=morphology.remove_small_objects(dapimask,min_size=30)
    dapimask = morphology.binary_dilation(dapimask, selem=morphology.square(12), out=None)
    pyplot.imsave(outpath + 'DetectedNucleusMask.tiff', dapimask.astype('uint8'))


    nonucleus=struct_img0*(1-dapimask)
    m, s = norm.fit(nonucleus.flat)
    pmin = nonucleus.min()
    pmax = nonucleus.max()
    p99 = np.percentile(nonucleus, 99.99)

    up_ratio = 0
    for up_i in np.arange(0.5, 1000, 0.5):
        if m + s * up_i > p99:
            if m + s * up_i > pmax:
                up_ratio = up_i - 0.5
            else:
                up_ratio = up_i
            break

    low_ratio = 0
    for low_i in np.arange(0.5, 1000, 0.5):
        if m - s * low_i < pmin:
            low_ratio = low_i - 0.5
            break
#    print(low_ratio, up_ratio)

    ################################
    ## PARAMETERS for this step ##
    # intensity_scaling_param = [1, 40]
    intensity_scaling_param = [low_ratio, up_ratio]
    gaussian_smoothing_sigma = 1
    ################################


    # intensity normalization
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)

    # smoothing with 2d gaussian filter slice by slice
    structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)

    ################################
    ## PARAMETERS for this step ##

    #s3_param = [[1, 0.025]] #good for 190920 with nuclei in normalizationn
    s3_param = [[0.75, 0.045]]#okayish for 190920 without nuclei in normalizationn

    ################################



    bw = dot_3d_wrapper(structure_img_smooth, s3_param)

    ################################
    ## PARAMETERS for this step ##
    minArea = 5
    ################################

    seg = remove_small_objects(bw > 0, min_size=minArea, connectivity=1, in_place=False)



    seg = seg > 0
    out = seg.astype(np.uint8)
    out[out > 0] = 255
    ##########################
    #Apply Nucleus mask
    ###########################
    combine=out*(1-dapimask)

    print('combine',combine.shape)


   # writer = OmeTiffWriter(outpath + 'SpotsSegmentationAICS.tiff',overwrite_file=True)
    #writer.save(combine.astype('uint16'))


    thresholdedspotstot = numpy.empty((1024,1024,0), dtype=np.uint16)
    boxes = []
    for SliceIT in range(combine.shape[0]):
        labels,num=label(combine[SliceIT,:,:],return_num=True)
        image_label_overlay =label2rgb(labels)
        regions=regionprops(labels, intensity_image=struct_img0[SliceIT,:,:])

        #print('num',num)
        areafill=[]
        int=[]
        dist=[]
        crops=[]
        axis=[]
        slices=[]
        ecc=[]
        num=0
        numtest=0
        initimage = numpy.zeros(combine[SliceIT,:,:].shape)

        for region in regions:
            if np.all([region.area>5,region.area<300,region.max_intensity<threshint]):
                try:
                    axis.append((region.major_axis_length, region.minor_axis_length))
                except:
                    continue
                crop=region.intensity_image
                crop = crop / numpy.max(crop)
                crops.append(crop)
                areafill.append(region.area)

                int.append(region.max_intensity)
                dist.append(region.centroid)
                initimage[region.slice]=1
                numtest+=1

            num+=1
        if numtest<2:
            print('No bright clusters')
            continue

        thresholdedspots = combine[SliceIT, :, :] * initimage
        thresholdedspotstot=numpy.dstack((thresholdedspotstot,thresholdedspots))



        gamma=0.3
        if 'shNorbin02' in imagei:
            print('norb')

            if len(int)>10:
                inthistNorbin.extend(int)
                if len(namesnorbin)==0:
                    print('empty list')
                    numclusterNorbin.append(len(int))
                elif os.path.basename(imagei) in namesnorbin:
                    numclusterNorbin[-1]+=len(int) 
                else:
                    numclusterNorbin.append(len(int))    
                namesnorbin.append(os.path.basename(imagei))

            norb+=1


        elif 'shFus318'in imagei:
            print('fus')
            namesfus.append(os.path.basename(imagei))
            inthistFus.extend(int)
            fus+=1

        elif 'shFus315' in imagei:
            print('fus5')
            namesfus5.append(os.path.basename(imagei))
            inthistFus5.extend(int)
            fus5+=1


        elif 'plko_0' in os.path.basename(imagei).lower():
            print('PLKO')
            
            
            if len(int)>10:
                inthistPLKO.extend(int)
                if len(namesPLKO)==0:
                    numclusterPLKO.append(len(int))
                elif os.path.basename(imagei) in namesPLKO:
                    numclusterPLKO[-1]+=len(int) 
                else:
                    numclusterPLKO.append(len(int))        
                namesPLKO.append(os.path.basename(imagei))

            plko+=1


    thresholdedspotstot = numpy.moveaxis(thresholdedspotstot, 2, 0)
    writer = OmeTiffWriter(outpath + 'ThresholdSpotsMask.tiff',overwrite_file=True)
    writer.save(thresholdedspotstot.astype('uint16'))


fig, ax = pyplot.subplots(figsize=(12, 8))
ax.hist(inthistPLKO, bins=300, density=True, alpha=0.2, label='PLKO')
ax.hist(inthistFus, bins=300, density=True, alpha=0.2, label='shFus315')
ax.hist(inthistFus5,bins=300,density=True,alpha=0.2,label='shFus315')
ax.hist(inthistNorbin, bins=300, density=True, alpha=0.2, label='shNorbin')
ax.legend(loc="upper right")
fig.suptitle('Spot Max intensity')
pyplot.savefig(outpath + 'Spot Max intensity' + str(threshint) + '.pdf', transparent=True)

pyplot.figure()
pyplot.plot(numpy.sort(inthistPLKO), numpy.linspace(0, 1, len(inthistPLKO), endpoint=False), label='PLKO')
pyplot.plot(numpy.sort(inthistFus5), numpy.linspace(0, 1, len(inthistFus5), endpoint=False), label='shFus315')
pyplot.plot(numpy.sort(inthistFus), numpy.linspace(0, 1, len(inthistFus), endpoint=False), label='shFus318')
pyplot.plot(numpy.sort(inthistNorbin), numpy.linspace(0, 1, len(inthistNorbin), endpoint=False), label='shNorbin')
pyplot.legend(loc="upper right")
pyplot.title('Spot Max intensity')
pyplot.savefig(outpath + 'Spot Max intensity' + str(threshint) + '_Cumulativecurve.pdf', transparent=True)



################################
# Creation of output dictionary
##################################
Dictnorm={'PLKO':inthistPLKO,'shNorbin02':inthistNorbin,'numPLKO':numclusterPLKO,'numshNorbin02':numclusterNorbin}
numpy.save(os.path.join(os.getcwd(),"Fusgranules","noMAP2","Dict"+date+"_normskewnorm.npy"), Dictnorm)



pyplot.show()

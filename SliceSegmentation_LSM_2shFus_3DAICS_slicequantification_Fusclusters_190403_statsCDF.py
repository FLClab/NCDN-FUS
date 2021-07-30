
from skimage import io, filters, morphology
from skimage.measure import  regionprops, label
import os
import matplotlib.pyplot as pyplot
import glob
import numpy as np
import numpy
from skimage.color import label2rgb
import scipy
import math
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
# function for core algorithm
from aicssegmentation.core.seg_dot import dot_3d, dot_3d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice,image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects, watershed, dilation, erosion, ball
# function for post-processing (size filter)
from skimage.feature import peak_local_max
from skan import Skeleton, summarize, skeleton_to_csgraph, draw
from matplotlib.patches import Rectangle
from scipy.stats import norm
from scipy.signal import find_peaks

import tifffile
from scipy.spatial import distance




def sample_coords(coords, dist=128):
    """
    Samples a list of coords

    :param coords: A 2D `numpy.ndarray` of coordinates
    :param dist: The minimal distance between sampled points

    :returns : A 2D `numpy.ndarray` of sampled coordinates
    """
    sampled_coords = []
    while len(coords) > 0:
        sampled_coords.append(coords[0][numpy.newaxis])
        distances = distance.cdist(sampled_coords[-1], coords).ravel()
        coords = coords[distances >= dist]
    return numpy.concatenate(sampled_coords, axis=0)

def compute_adj_matrix(skeleton):
    """
    Computes the adjency matrix from the skeleton between each nodes

    :param skeleton: A `skan.Skeleton` object

    :returns : A `numpy.ndarray` of each nodes in the skeleton
               A graph computed from the adjency matrix
    """
    nodes = []
    for i in range(skeleton.n_paths):
        coords = skeleton.path_coordinates(i)
        nodes.extend([coords[0], coords[-1]])
    nodes = numpy.unique(nodes, axis=0)

    adj = numpy.zeros((len(nodes), len(nodes)))
    path_lengths = skeleton.path_lengths()
    for i in range(skeleton.n_paths):
        coords = skeleton.path_coordinates(i)
        m = numpy.argwhere(numpy.all(nodes == coords[0], axis=1)).ravel()
        n = numpy.argwhere(numpy.all(nodes == coords[-1], axis=1)).ravel()
        adj[m, n] = path_lengths[i]
        adj[n, m] = path_lengths[i]
    return nodes, networkx.from_numpy_array(adj)


def find_longest_path(graph):
    """
    Calculates the longest path from a shortest path algorithm

    :param graph: A `networkx.graph`

    :returns : A source node
               A end node
               The longest path length
               Each nodes in the path
    """
    paths_lengths = dict(networkx.shortest_path_length(graph, weight='weight'))
    paths = dict(networkx.shortest_path(graph, weight='weight'))
    dist_max = 0
    max_source, max_destination = None, None
    for (source, destination) in paths_lengths.items():
        dist_max_key = list(sorted(destination.items(), key=lambda item: item[1]))[-1][0]
        if destination[dist_max_key] > dist_max:
            max_source, max_destination = source, dist_max_key
            dist_max = destination[dist_max_key]
    return max_source, \
           max_destination, \
           paths_lengths[max_source][max_destination], \
           paths[max_source][max_destination]


def analyse(binary_image, show_results=False):
    """
    Analyses the branch of a binary image

    :param binary_image: A 2D binary `numpy.ndarray`

    :returns : A `dict` with maximal and total length of branches and coords
    """
    #binary_image = binary_image[256:512, 256:512]
    binary_image_skeleton = morphology.skeletonize(binary_image)
    skeleton = skan.Skeleton(binary_image_skeleton)
    nodes, graph = compute_adj_matrix(skeleton)

    source, dest, length, path = find_longest_path(graph)

    coords_array = []
    for i in range(len(path) - 1):
        node_start, node_end = path[i], path[i + 1]
        for j in range(skeleton.n_paths):
            coords = skeleton.path_coordinates(j)
            if (numpy.all(nodes[node_start] == coords[0]) & numpy.all(nodes[node_end] == coords[-1])) or \
                    (numpy.all(nodes[node_end] == coords[0]) & numpy.all(nodes[node_start] == coords[-1])):
                coords_array.append(coords)
                continue
    coords_array = numpy.concatenate(coords_array)
    if show_results:
        fig, ax = pyplot.subplots(figsize=(7, 7))
        ax.imshow(binary_image_skeleton, cmap="gray")
        ax.plot(coords_array[:, 1], coords_array[:, 0], color="tab:blue")
        pyplot.show()

    path_lengths = {
        "coords": coords_array,
        "max_length": length,
        "total_length": sum(skeleton.path_lengths())
    }
    return path_lengths




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
path2=os.path.join("D:/Chantelle/190403_Bassoon/*.lsm")

images2=glob.glob(path2)

print(len(images2))

output_dir=os.path.expanduser("~/Desktop/Chantelle/masks_"+str(threshint)+"_1BassoonDataser_3DAICS_SuggNorm_newparam_4aroundmid_threshes_slicequant_DAPINEW/")
os.makedirs(output_dir,exist_ok=True)


file='Image1'

inthistFus=[]
inthistNorbin=[]
inthistFus5=[]
inthistPLKO=[]

areahistNorbin=[]
areahistFus=[]
areahistFus5=[]
areahistPLKO=[]

Norbinratio=[]
Fusratio=[]
Fusratio5=[]
PLKOratio=[]
Norbinlindensity=[]
Fuslindensity=[]
Fus5lindensity=[]
PLKOlindensity=[]

Norbinrationum=[]
Fusrationum=[]
Fusrationum5=[]
PLKOrationum=[]
Norbinarea=[]
Fusarea=[]
Fusarea5=[]
PLKOarea=[]
Norbindist=[]
Fusdist=[]
Fusdist5=[]
PLKOdist=[]
Norbinint=[]
Fusint=[]
Fusint5=[]
PLKOint=[]
PLKOSizeCompare=[]
NorbinSizeCompare=[]
FusSizeCompare=[]
FusSizeCompare5=[]
namesfus=[]
namesfus5=[]
namesPLKO=[]
namesnorbin=[]
Norbinecc=[]
Fusecc=[]

PLKOecc=[]

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
    print(imagei)
    outpath = output_dir + os.path.basename(imagei)
    #reader = AICSImage(imagei)
    #IMG = reader.data.astype(np.float32)
    image1=tifffile.imread(imagei).astype(np.float32)
    IMG=image1


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


    mid = np.int(np.around(0.5 * IMG.shape[1]))


    print(mid)
    if '-new_' in imagei:
        print('I''M A NEW IMAGE')
        IMG=IMG[:,mid-3:mid+1,:,:,:]
        print('IMG', IMG.shape)

    elif IMG.shape[2]>4:
        IMG=IMG[:,mid-1:mid+3,:,:,:]
        print('IMG', IMG.shape)

    print(np.min(IMG),np.max(IMG))
    print('IMG', IMG.shape)
    project=numpy.max(IMG[0,:,:,:,:],axis=0)
    print('project', project.shape)

    N_CHANNELS = IMG.shape[2]
    print(' N_CHANNELS ', N_CHANNELS )


    #image1 = image1[0, mid - 1, :, :, :]

    #image1=numpy.moveaxis(image1,0,2)
    project = numpy.moveaxis(project, 0, 2)
#
#    pyplot.figure(figsize=(12, 8))
#    pyplot.imshow(project[:,:,0])
#    pyplot.figure(figsize=(12, 8))
#    pyplot.imshow(project[:,:,1])
#    pyplot.figure(figsize=(12, 8))


    #####################
    structure_channel = 0
    #####################

    struct_img0 = IMG[0,:,structure_channel, :, :].copy()
    print('struct_img0',struct_img0.shape)


    writer = OmeTiffWriter(outpath + 'OriginalImage.tiff',overwrite_file=True)
    writer.save(struct_img0.astype('uint16'))


    ############
    #Dendrite and Nucleus masks
    ############
    
    
    gaussian_smoothing_sigma = 5
    structure_img_smooth = image_smoothing_gaussian_3d(project[:,:,2], sigma=gaussian_smoothing_sigma)
    thresh=numpy.percentile(structure_img_smooth,95)
    print(thresh)
    DapiMask = (structure_img_smooth-500) > thresh
    dapimask = morphology.binary_dilation(DapiMask, selem=morphology.square(12), out=None)
    pyplot.imsave(outpath + 'DetectedNucleusMask_2.tiff', dapimask.astype('uint8'))

#    dapithresh = filters.threshold_triangle(project[:,:,1])
#    dapimask= project[:,:,1] > dapithresh
#
#    #dapimask=get_foreground(image1[:,:,2].astype('uint8'), 1)
#    dapimask=morphology.remove_small_objects(dapimask,min_size=30)
#    dapimask = morphology.binary_dilation(dapimask, selem=morphology.square(12), out=None)
#    pyplot.imsave(outpath + 'DetectedNucleusMask.tiff', dapimask.astype('uint8'))


    # gaussian_smoothing_sigma = 5
    # structure_img_smooth = image_smoothing_gaussian_3d(project[:,:,1], sigma=gaussian_smoothing_sigma)
    # thresh=numpy.percentile(structure_img_smooth,95)
    # DapiMask = structure_img_smooth > thresh
    # dapimask = morphology.binary_dilation(DapiMask, selem=morphology.square(12), out=None)
    # pyplot.imsave(outpath + 'DetectedNucleusMask_2.tiff', dapimask.astype('uint8'))
    #





    m, s = norm.fit(struct_img0.flat)
    pmin = struct_img0.min()
    pmax = struct_img0.max()
    p99 = np.percentile(struct_img0, 99.99)

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

    ################################
    ## PARAMETERS for this step ##
    # intensity_scaling_param = [1, 40]
    intensity_scaling_param = [low_ratio, up_ratio]
    gaussian_smoothing_sigma = 1
    ################################
    #from aicssegmentation.core.pre_processing_utils import suggest_normalization_param
    #suggest_normalization_param(struct_img0)


    # intensity normalization
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)

    # smoothing with 2d gaussian filter slice by slice
    structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)

    ################################
    ## PARAMETERS for this step ##
    #s3_param = [[1, 0.05]]
    #s3_param=[[0.5, 0.05]]
    s3_param = [[1, 0.02]]


    #s3_param_IFB=[[1,0.05]]
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
    #Apply MAP2 and nucleus masks
    combine=out*(1-dapimask)
    print('combine',combine.shape)

    #pyplot.imsave(outpath + 'DetectedDendriteMask.tiff', map2mask2.astype('uint8'))




    writer = OmeTiffWriter(outpath + 'SpotsSegmentationAICS.tiff',overwrite_file=True)
    writer.save(combine.astype('uint16'))


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
            if np.all([region.area>5,region.area<200,region.max_intensity<threshint]):
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


        dists = numpy.array(dist)


        countt=[]
        locallindensity=[]
        arearatio=[]
        rationum = []



        #print('count',np.max(countt))
        #print('count', np.mean(countt))

        #print(np.min(arearatio),np.max(arearatio))
        #print('arearatio',len(arearatio))
        #print('arearatio', arearatio[0])
        LinDensity=locallindensity
        ratio=arearatio
        #print(len(LinDensity))

        #print(min(arearatio),max(arearatio))
        #print(min(rationum),max(rationum))


        areas=numpy.array(areafill)
        dists = numpy.array(dist)
        print('dists',dists.shape)
        intensities=numpy.array(int)
        minint=numpy.min(intensities)


        axislengths=numpy.array(axis)
        #print('Axis',numpy.max(axislengths))


        meanintensities=numpy.mean(intensities)
        area=numpy.mean(areas)

        distances = scipy.spatial.distance.cdist(dist, dist, 'euclidean')

        notself=numpy.sort(distances)
        try:
            mindistances=notself[:,1]
            meanmindistances = numpy.mean(mindistances)
        except IndexError:
            mindistances=[]
            meanmindistances =[]

        #ratio=numpy.sum(combine)/numpy.sum(map2mask2*(1-dapimask))
        #rationum=num/numpy.sum(map2mask2*(1-dapimask))

        gamma=0.3
        if 'shnorbin02'  in os.path.basename(imagei).split('/')[-1].lower():
            print('norb')
            
            #Norbinratio.extend(ratio)
            #Norbinrationum.extend(rationum)
            #Norbindist.append(meanmindistances)
            #Norbinarea.append(area)
            #Norbinint.append(meanintensities)
            #Norbinlindensity.extend(LinDensity)
            if len(int)>10:
                inthistNorbin.extend(int)
                if len(namesnorbin)==0:
                    numclusterNorbin.append(len(int))
                elif os.path.basename(imagei) in namesnorbin:
                    numclusterNorbin[-1]+=len(int) 
                else:
                    numclusterNorbin.append(len(int))                

                areahistNorbin.extend(areafill)
                namesnorbin.append(os.path.basename(imagei))
            #Norbinecc.extend(ecc)
            norb+=1


        elif '318'  in os.path.basename(imagei).split('/')[-1].lower():
            print('fus')
            namesfus.append(os.path.basename(imagei))
            #Fusratio.extend(ratio)
            #Fusrationum.extend(rationum)
            #Fusarea.append(area)
            #Fusdist.append(meanmindistances)
            #Fusint.append(meanintensities)
            inthistFus.extend(int)
            areahistFus.extend(areafill)
            #Fuslindensity.extend(LinDensity)
            #Fusecc.extend(ecc)

            fus+=1

        elif '315'  in os.path.basename(imagei).split('/')[-1].lower():
            print('fus5')
            namesfus5.append(os.path.basename(imagei))
            Fusratio5.extend(ratio)
            Fusrationum5.extend(rationum)
            Fusarea5.append(area)
            Fusdist5.append(meanmindistances)
            Fusint5.append(meanintensities)
            inthistFus5.extend(int)
            areahistFus5.extend(areafill)
            Fus5lindensity.extend(LinDensity)

            fus5+=1



        elif 'plko' or 'shctl' in os.path.basename(imagei).split('/')[-1].lower():
            print('PLKO')
            
            
            #PLKOratio.extend(ratio)
            PLKOarea.append(area)
            #PLKOrationum.extend(rationum)
            #PLKOdist.append(meanmindistances)
            #PLKOint.append(meanintensities)
            if len(int)>10:
                inthistPLKO.extend(int)
                if len(namesPLKO)==0:
                    numclusterPLKO.append(len(int))
                elif os.path.basename(imagei) in namesPLKO:
                    numclusterPLKO[-1]+=len(int) 
                else:
                    numclusterPLKO.append(len(int))
                areahistPLKO.extend(areafill)
                namesPLKO.append(os.path.basename(imagei))
            #PLKOlindensity.extend(LinDensity)
            #PLKOecc.extend(ecc)
            plko+=1


    thresholdedspotstot = numpy.moveaxis(thresholdedspotstot, 2, 0)
    writer = OmeTiffWriter(outpath + 'ThresholdSpotsMask.tiff',overwrite_file=True)
    writer.save(thresholdedspotstot.astype('uint16'))

# fig, ax = pyplot.subplots(figsize=(12, 8))
# ax.hist(inthistPLKO, bins=200, density=True, alpha=0.2, label='PLKO')
# xt = pyplot.xticks()[0]
# xmin, xmax = min(xt), max(xt)
# lnspc = np.linspace(xmin, xmax, len(inthistPLKO))
# # lets try the normal distribution first
# m, s = scipy.stats.norm.fit(inthistPLKO)  # get mean and standard deviation
# pdf_g = scipy.stats.norm.pdf(lnspc, m, s)  # now get theoretical values in our interval
# ax.plot(lnspc, pdf_g, label="Norm")  # plot it
# ax.hist(inthistFus, bins=200, density=True, alpha=0.2, label='shFus318')
# xmin, xmax = min(inthistFus), max(inthistFus)
# lnspc = np.linspace(xmin, xmax, len(inthistFus))
# # lets try the normal distribution first
# m, s = scipy.stats.norm.fit(inthistFus)  # get mean and standard deviation
# pdf_g = scipy.stats.norm.pdf(lnspc, m, s)  # now get theoretical values in our interval
# ax.plot(lnspc, pdf_g, label="Norm")  # plot it
# ax.hist(inthistNorbin, bins=200, density=True, alpha=0.2, label='shNorbin')
#
# xmin, xmax = min(inthistNorbin), max(inthistNorbin)
# lnspc = np.linspace(xmin, xmax, len(inthistNorbin))
# # lets try the normal distribution first
# m, s = scipy.stats.norm.fit(inthistNorbin)  # get mean and standard deviation
# pdf_g = scipy.stats.norm.pdf(lnspc, m, s)  # now get theoretical values in our interval
# ax.plot(lnspc, pdf_g, label="Norm")  # plot it
# ax.legend(loc="upper right")
# fig.suptitle('Spot Max intensity')
# #pyplot.savefig(outpath + 'Spot Max intensity' + str(threshint) + '.pdf', transparent=True)
shapiroPLKO=scipy.stats.shapiro(inthistPLKO)
PLKOAnderson=scipy.stats.anderson(inthistPLKO,'norm')

shapiroNorbin=scipy.stats.shapiro(inthistNorbin)
NorbinAnderson=scipy.stats.anderson(inthistNorbin,'norm')

print('SHAPIRO ')
print('Norbin',shapiroNorbin)
print('PLKO',shapiroPLKO)
print('ANDERSON')
print('Norbin',NorbinAnderson)
print('PLKO',PLKOAnderson)
KSNorbin=scipy.stats.ks_2samp(inthistNorbin,inthistPLKO)
print('Norbin',KSNorbin)
fig, ax = pyplot.subplots(figsize=(12, 8))
ax.hist(inthistPLKO, bins=300, density=True, alpha=0.2, label='PLKO')
ax.hist(inthistFus, bins=300, density=True, alpha=0.2, label='shFus315')
ax.hist(inthistFus5,bins=300,density=True,alpha=0.2,label='shFus315')
ax.hist(inthistNorbin, bins=300, density=True, alpha=0.2, label='shNorbin')
ax.legend(loc="upper right")
fig.suptitle('Spot Max intensity')
pyplot.savefig(outpath + 'Spot Max intensity' + str(threshint) + '.pdf', transparent=True)

fig, ax = pyplot.subplots(figsize=(12, 8))
countsP,binsP,p=ax.hist(inthistPLKO, bins=300, density=True, alpha=0.2, label='PLKO')
#peaks, _ = find_peaks(countsP,distance=500)
#ax.plot(binsP[peaks],countsP[peaks], "x")
ax.vlines(numpy.mean(inthistPLKO), 0, 0.0005,color='blue', linestyle='dashed', label='meanPLKO')
ax.vlines(numpy.median(inthistPLKO), 0, 0.0005,color='blue' ,label='medianPLKO')
a,loc,scale  = scipy.stats.skewnorm.fit(inthistPLKO)
best_fit_line = scipy.stats.skewnorm.pdf(binsP,a,loc,scale)
peaks, _ = find_peaks(best_fit_line,distance=500)
ax.plot(binsP[peaks],best_fit_line[peaks], "x")
normfactor=binsP[peaks]
ax.plot(binsP, best_fit_line)
countsN,binsN,p=ax.hist(inthistNorbin, bins=300, density=True, alpha=0.2, label='shNorbin')
#peaks, _ = find_peaks(countsN,distance=500)
#ax.plot(binsN[peaks],countsN[peaks], "x")
a,loc,scale  = scipy.stats.skewnorm.fit(inthistNorbin)
best_fit_line = scipy.stats.skewnorm.pdf(binsN,a,loc,scale)
peaks, _ = find_peaks(best_fit_line,distance=500)
ax.plot(binsN[peaks],best_fit_line[peaks], "x")

ax.plot(binsN, best_fit_line)
ax.vlines(numpy.mean(inthistNorbin), 0, 0.0005, linestyle='dashed',color='green',label='meanNorb')
ax.vlines(numpy.median(inthistNorbin), 0, 0.0005, color='green',label='medianNorb')
ax.legend(loc="upper right")
fig.suptitle('Spot Max intensity')
pyplot.savefig(output_dir  + 'Spot Max intensity' + str(threshint) + '_peaks.pdf', transparent=True)




#fig, ax = pyplot.subplots(figsize=(12, 8))
# ax.hist(PLKOecc, bins=300, density=True, alpha=0.2, label='PLKO')
# ax.hist(Fusecc, bins=300, density=True, alpha=0.2, label='shFus318')
# ax.hist(Norbinecc, bins=300, density=True, alpha=0.2, label='shNorbin')
# ax.legend(loc="upper right")
# fig.suptitle('Eccentricity')
# pyplot.savefig(outpath + 'Eccentricity' + str(threshint) + '.pdf', transparent=True)
#
# pyplot.figure()
# pyplot.plot(numpy.sort(PLKOecc), numpy.linspace(0, 1, len(PLKOecc), endpoint=False), label='PLKO')
# pyplot.plot(numpy.sort(Fusecc), numpy.linspace(0, 1, len(Fusecc), endpoint=False), label='shFus318')
# pyplot.plot(numpy.sort(Norbinecc), numpy.linspace(0, 1, len(Norbinecc), endpoint=False), label='shNorbin')
# pyplot.legend(loc="upper right")
# pyplot.title('Eccentricity')
# pyplot.savefig(outpath + 'Eccentricity' + str(threshint) + '_CumulativeCurve.pdf', transparent=True)

pyplot.figure()
pyplot.plot(numpy.sort(inthistPLKO), numpy.linspace(0, 1, len(inthistPLKO), endpoint=False), label='PLKO')
pyplot.plot(numpy.sort(inthistFus5), numpy.linspace(0, 1, len(inthistFus5), endpoint=False), label='shFus315')
pyplot.plot(numpy.sort(inthistFus), numpy.linspace(0, 1, len(inthistFus), endpoint=False), label='shFus318')
pyplot.plot(numpy.sort(inthistNorbin), numpy.linspace(0, 1, len(inthistNorbin), endpoint=False), label='shNorbin')
pyplot.legend(loc="upper right")
pyplot.title('Spot Max intensity')
pyplot.savefig(outpath + 'Spot Max intensity' + str(threshint) + '_Cumulativecurve.pdf', transparent=True)

fig, ax = pyplot.subplots(figsize=(12, 8), sharex=True, sharey=True)
ax.hist(areahistPLKO, bins=100, density=True, alpha=0.2, label='PLKO')
ax.hist(areahistFus, bins=100, density=True, alpha=0.2, label='shFus318')
ax.hist(areahistFus5,bins=500,density=True,alpha=0.2,label='shFus315')
ax.hist(areahistNorbin, bins=100, density=True, alpha=0.2, label='shNorbin')
ax.legend(loc="upper right")

fig.suptitle('Spot Area')
pyplot.savefig(outpath + 'Spot Area' + str(threshint) + '_2.pdf', transparent=True)

pyplot.figure()
pyplot.plot(numpy.sort(areahistPLKO), numpy.linspace(0, 1, len(areahistPLKO), endpoint=False), label='PLKO')
pyplot.plot(numpy.sort(areahistFus5), numpy.linspace(0, 1, len(areahistFus5), endpoint=False), label='shFus315')
pyplot.plot(numpy.sort(areahistFus), numpy.linspace(0, 1, len(areahistFus), endpoint=False), label='shFus318')

pyplot.plot(numpy.sort(areahistNorbin), numpy.linspace(0, 1, len(areahistNorbin), endpoint=False), label='shNorbin')
pyplot.legend(loc="upper right")
pyplot.xlim(0, 140)
pyplot.title('Spot Area')
pyplot.savefig(outpath + 'Spot Area' + str(threshint) + '_cumulativecurve.pdf', transparent=True)

pyplot.figure()
pyplot.plot(numpy.sort(PLKOratio), numpy.linspace(0, 1, len(PLKOratio), endpoint=False), label='PLKO')
pyplot.plot(numpy.sort(Fusratio), numpy.linspace(0, 1, len(Fusratio), endpoint=False), label='shFus315')
pyplot.plot(numpy.sort(Fusratio5), numpy.linspace(0, 1, len(Fusratio5), endpoint=False), label='shFus315')
pyplot.plot(numpy.sort(Norbinratio), numpy.linspace(0, 1, len(Norbinratio), endpoint=False), label='shNorbin')
pyplot.legend(loc="upper right")
pyplot.xlim(0, 0.25)
pyplot.title('Spot Area/Dendrite Area ratio')
pyplot.savefig(outpath + 'Area per Dendrite Area' + str(threshint) + '_Cumulativecurve.pdf', transparent=True)

pyplot.figure()
pyplot.plot(numpy.sort(PLKOrationum), numpy.linspace(0, 1, len(PLKOrationum), endpoint=False), label='PLKO')
pyplot.plot(numpy.sort(Fusrationum), numpy.linspace(0, 1, len(Fusrationum), endpoint=False), label='shFus318')
pyplot.plot(numpy.sort(Fusrationum5), numpy.linspace(0, 1, len(Fusrationum5), endpoint=False), label='shFus315')
pyplot.plot(numpy.sort(Norbinrationum), numpy.linspace(0, 1, len(Norbinrationum), endpoint=False), label='shNorbin')
pyplot.legend(loc="upper right")
pyplot.title('Number of spots/Dendrite Area ratio')
pyplot.savefig(outpath + 'Number Dendrite Area' + str(threshint) + '_Cumulativecurve.pdf', transparent=True)

fig, ax = pyplot.subplots(figsize=(12, 8))
ax.hist(PLKOratio, bins=300, density=True, alpha=0.2, range=(0.001, 1), label='PLKO')
ax.hist(Fusratio, bins=300, density=True, alpha=0.2, range=(0.001, 1), label='shFus315')
ax.hist(Norbinratio, bins=300, density=True, alpha=0.2, range=(0.001, 1), label='shNorbin')
ax.legend(loc="upper right")
fig.suptitle('Spot Area/Dendrite Area ratio')
pyplot.savefig(outpath + 'Area per Dendrite Area' + str(threshint) + '.pdf', transparent=True)

fig, ax = pyplot.subplots(figsize=(12, 8))
ax.hist(PLKOrationum, bins=300, density=True, alpha=0.2, label='PLKO')
ax.hist(Fusrationum, bins=300, density=True, alpha=0.2, label='shFus318')
ax.hist(Fusrationum5, bins=300, density=True, alpha=0.2, label='shFus315')
ax.hist(Norbinrationum, bins=300, density=True, alpha=0.2, label='shNorbin')
ax.legend(loc="upper right")
pyplot.xlim(0, 0.015)
fig.suptitle('Number of spots/Dendrite Area ratio')
pyplot.savefig(outpath + 'Number per Dendrite volume' + str(threshint) + '.pdf', transparent=True)

pyplot.figure()
pyplot.plot(numpy.sort(PLKOlindensity), numpy.linspace(0, 1, len(PLKOlindensity), endpoint=False), label='PLKO')
pyplot.plot(numpy.sort(Fuslindensity), numpy.linspace(0, 1, len(Fuslindensity), endpoint=False), label='shFus318')
pyplot.plot(numpy.sort(Fus5lindensity), numpy.linspace(0, 1, len(Fus5lindensity), endpoint=False), label='shFus315')

pyplot.plot(numpy.sort(Norbinlindensity), numpy.linspace(0, 1, len(Norbinlindensity), endpoint=False),
            label='shNorbin')
pyplot.legend(loc="upper right")
pyplot.xlim(0, 0.015)
pyplot.title('Number of Spots/Dendrite length')
pyplot.savefig(outpath + 'Number per Dendrite length' + str(threshint) + '_Cumulativecurve.pdf', transparent=True)

fig, ax = pyplot.subplots(figsize=(12, 8))
ax.hist(PLKOlindensity, bins=300, density=True, alpha=0.2, label='PLKO')
ax.hist(Fuslindensity, bins=300, density=True, alpha=0.2, label='shFus315')
ax.hist(Norbinlindensity, bins=300, density=True, alpha=0.2, label='shNorbin')
ax.legend(loc="upper right")
fig.suptitle('Number of spots/Dendrite length ratio')
pyplot.savefig(outpath + 'Number per Dendrite length' + str(threshint) + '.pdf', transparent=True)

xrand1 = numpy.random.normal(1, scale=0.02, size=(len(Norbinarea), 1))
xrand2 = numpy.random.normal(1.5, scale=0.02, size=(len(Fusarea), 1))
xrand4 = numpy.random.normal(2, scale=0.02, size=(len(Fusarea5), 1))
xrand3 = numpy.random.normal(2.5, scale=0.02, size=(len(PLKOarea), 1))

fig, ax = pyplot.subplots(figsize=(12, 8))
ax.boxplot([Norbinarea, Fusarea,Fusarea5,PLKOarea], positions=[1, 1.5, 2,2.5],
           labels=['shNorbin02', 'shFus318', 'shFus315','PLKO'])
ax.scatter(xrand1, Norbinarea)
ax.scatter(xrand2, Fusarea)
ax.scatter(xrand3, PLKOarea)
ax.scatter(xrand4, Fusarea5)
ax.set_title('Mean spot area')
pyplot.savefig(output_dir + 'Spot Area' + str(threshint) + '_boxplot_frames.pdf', transparent=True)
# ax.set_ylim([10,19])

xrand1 = numpy.random.normal(1, scale=0.02, size=(len(Norbinint), 1))
xrand2 = numpy.random.normal(1.5, scale=0.02, size=(len(Fusint), 1))
xrand3 = numpy.random.normal(2, scale=0.02, size=(len(PLKOint), 1))
xrand4 = numpy.random.normal(2.5, scale=0.02, size=(len(Fusint5), 1))

fig, ax = pyplot.subplots(figsize=(12, 8))
ax.boxplot([Norbinint, Fusint, PLKOint,Fusint5], positions=[1, 1.5, 2,2.5],
           labels=['shNorbin02', 'shFus318', 'PLKO','shFus315'])
ax.scatter(xrand1, Norbinint)
ax.scatter(xrand2, Fusint)
ax.scatter(xrand3, PLKOint)
ax.scatter(xrand4, Fusint5)
#for l,name in enumerate(namesfus):
#    ax.text(xrand2[l], Fusint[l], name, fontsize=9)
#for l,name in enumerate(namesnorbin):
#    ax.text(xrand1[l], Norbinint[l], name, fontsize=9)

# ax.set_ylim([0,12000])
ax.set_title('Mean of Max spot intensities')

pyplot.savefig(output_dir + 'Mean spot intensity' + str(threshint) + '_boxplot_frames.pdf', transparent=True)
##############  normalization #####################################################
Dictnorm={'PLKO':inthistPLKO,'shNorbin02':inthistNorbin,'numPLKO':numclusterPLKO,'numshNorbin02':numclusterNorbin}

inthistPLKO=inthistPLKO/normfactor
inthistNorbin=inthistNorbin/normfactor
fig, ax = pyplot.subplots(figsize=(12, 8))
ax.hist(inthistPLKO, bins=300, density=True, alpha=0.2, label='PLKO')
ax.hist(inthistFus, bins=300, density=True, alpha=0.2, label='shFus315')
ax.hist(inthistFus5,bins=300,density=True,alpha=0.2,label='shFus315')
ax.hist(inthistNorbin, bins=300, density=True, alpha=0.2, label='shNorbin')
ax.legend(loc="upper right")
fig.suptitle('Spot Max intensity')
pyplot.savefig(outpath + 'Spot Max intensity' + str(threshint) + '.pdf', transparent=True)

fig, ax = pyplot.subplots(figsize=(12, 8))
countsP,binsP,p=ax.hist(inthistPLKO, bins=300, density=True, alpha=0.2, label='PLKO')


ax.vlines(numpy.mean(inthistPLKO), 0, 1, label='meanPLKO')
ax.vlines(numpy.median(inthistPLKO), 0, 1, label='medianPLKO')

a,loc,scale  = scipy.stats.skewnorm.fit(inthistPLKO)
best_fit_line = scipy.stats.skewnorm.pdf(binsP,a,loc,scale)
peaks, _ = find_peaks(best_fit_line,distance=500)
ax.plot(binsP[peaks],best_fit_line[peaks], "x")

ax.plot(binsP, best_fit_line)
#countsF,binsF,p=ax.hist(inthistFus, bins=300, density=True, alpha=0.2, label='shFus315')
#peaks, _ = find_peaks(countsF,distance=500)
#ax.plot(binsF[peaks],countsF[peaks], "x")
#countsF5,binsF5,p=ax.hist(inthistFus5,bins=300,density=True,alpha=0.2,label='shFus315')
#peaks, _ = find_peaks(countsF5,distance=500)
#ax.plot(binsF5[peaks],countsF5[peaks], "x")
countsN,binsN,p=ax.hist(inthistNorbin, bins=300, density=True, alpha=0.2, label='shNorbin')

#peaks, _ = find_peaks(countsN,distance=500)
#ax.plot(binsN[peaks],countsN[peaks], "x")
ax.vlines(numpy.mean(inthistNorbin), 0, 1, label='meanNorbin')
ax.vlines(numpy.median(inthistNorbin), 0, 1, label='medianNorbin')
a,loc,scale  = scipy.stats.skewnorm.fit(inthistNorbin)
best_fit_line = scipy.stats.skewnorm.pdf(binsN,a,loc,scale)
peaks, _ = find_peaks(best_fit_line,distance=500)
ax.plot(binsN[peaks],best_fit_line[peaks], "x")
ax.plot(binsN, best_fit_line)
ax.legend(loc="upper right")
fig.suptitle('Spot Max intensity')
pyplot.savefig(output_dir  + 'Spot Max intensity' + str(threshint) + '_peaks_normalizedtoskew.pdf', transparent=True)
numpy.save(output_dir +"Dict190403_normskewnorm.npy", Dictnorm)


pyplot.show()


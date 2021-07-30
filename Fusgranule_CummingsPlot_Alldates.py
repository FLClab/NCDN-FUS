# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:35:45 2021

@author: andde
"""
import os
import glob
import numpy
import matplotlib.pyplot as pyplot
from scipy.signal import find_peaks
import scipy
import statsutils
input_dir=os.path.join(os.getcwd(),"Fusgranules","noMAP2","*.npy")
os.makedirs(os.path.join(os.getcwd(),"Fusgranules","noMAP2","graphs"),exist_ok=True)

npylist=glob.glob(input_dir)

data={}
boxplotdata={}
boxplotdatastd={}
boxplotdatapeaks={}
boxplotdatadate={}

conditions=['PLKO','shNorbin01','shNorbin02']
labels=['PLKO','shNorbin01','shNorbin02']
labelsnew=['CTL','NCDN-KD1','NCDN-KD2']

imnum=['numclustersPLKO','numclustersshNorbin01','numclustersshNorbin02']
look={'shNorbin01':'lightgrey','shNorbin02':'dimgrey','PLKO':'blue'}
style={'190403':'solid','190416':'dashed','190920':'dashdot','2011067':'dotted','201107':'dotted','210412':'solid',}
position={'shNorbin01':3,'shNorbin02':2,'PLKO':1}



for cond in conditions:
    boxplotdata[cond]=[]
    boxplotdatadate[cond]=[]
    boxplotdatapeaks[cond]=[]
    boxplotdatastd[cond]=[]
colors=['lightgrey','dimgrey','blue','green','magenta','cyan','black','lime']
condcount=0

for file in npylist:
    datevaluesx=[]
    datevaluesy=[]
    
    a=numpy.load(file,allow_pickle=True)
    fig, ax = pyplot.subplots(figsize=(12, 8))
    fig3, ax3 = pyplot.subplots(figsize=(12, 8))

    ax3.set_ylabel('histogram')  # we already handled the x-label with ax1
    ax3.tick_params(axis='y')
    date=os.path.basename(file).split('_')[0].replace('Dict','')
    print(date)
    for condcount,key in enumerate(conditions): 
        print(key)
        counts=a.item().get(key) #returns None if key not in dict
        imnums=a.item().get('num'+key)
        boxplotdatastd[key]=[]
        boxplotdata[key]=[]
        listcond=[]
        if counts == None:
            print('i am not here')
            continue
        else:
            boxplotdata[key]=counts
            ctr=0
            

            for j,im in enumerate(imnums):
                print('im',im)

                countshist,bins,p=ax3.hist(numpy.sort(counts[ctr:ctr+im]), bins=300, density=True, alpha=0.2, label=str(j)+key+date,color=look[key])
                aa,loc,scale  = scipy.stats.skewnorm.fit(numpy.sort(counts[ctr:ctr+im]))
                best_fit_line = scipy.stats.skewnorm.pdf(bins,aa,loc,scale)
                ax3.plot(bins, best_fit_line,color=look[key])
                peaks, _ = find_peaks(best_fit_line,distance=500)
                if bins[peaks].shape==(0,):
                    print('OOPS no peak')
                    continue

                
                ax3.plot(bins[peaks],best_fit_line[peaks], "x",color=look[key])
                boxplotdatastd[key].append(bins[peaks][0])
                listcond.append(bins[peaks][0])
                ctr+=im
                ax3.legend()


            countshist,bins,p=ax.hist(counts, bins=300, density=True, alpha=0.2, label=key,color=look[key])
            aa,loc,scale  = scipy.stats.skewnorm.fit(counts)
            best_fit_line = scipy.stats.skewnorm.pdf(bins,aa,loc,scale)
            peaks, _ = find_peaks(best_fit_line,distance=500)
            if condcount==0:
                normfactor=bins[peaks][0]
                print('I AM THE NORM',normfactor)


            ax.plot(bins, best_fit_line,color=look[key])
            ax.plot(bins[peaks],best_fit_line[peaks], "x",color=look[key])
            ax.set_title('Fus spot intensity '+date,fontsize=16)
            print(position[key],bins[peaks][0]/normfactor)
            print('std',[numpy.min(boxplotdatastd[key]),numpy.max(boxplotdatastd[key])])
            minv, maxv = -0.3,0.3
            spread=(maxv - minv)*numpy.random.rand() + minv

            datevaluesy.append(position[key]+spread)
            datevaluesx.append(bins[peaks][0]/normfactor)

            boxplotdatapeaks[key].extend(val/normfactor for val in listcond)
    idx   = numpy.argsort(datevaluesy)
    datevaluesx=numpy.array(datevaluesx)[idx]
    datevaluesy=numpy.array(datevaluesy)[idx]

    fig.savefig(os.path.join(os.getcwd(),"Fusgranules","noMAP2","graphs",date+'fittedhistogram.pdf'),transparent=True)
    fig3.savefig(os.path.join(os.getcwd(),"Fusgranules","noMAP2","graphs",date+'fittedhistogram_perimage.pdf'),transparent=True)



looks=['blue','lightgrey','dimgrey']

#Removal of outliers (2 standard deviations)
boxplotdatapeaksfiltered={}
for key in conditions:
    print(key)
    test= boxplotdatapeaks[key]
    print(len(test))

    maxval=numpy.mean(test)+2*numpy.std(test)

    minval=numpy.mean(test)-2*numpy.std(test)

    test1=[x for x in test if minval<x]
    print(len(test1))
    test2=[x for x in test1 if x<maxval]
    print(len(test2))

    boxplotdatapeaksfiltered[key] = test2

#Bootstrapping of differences between peak positions
samples = [boxplotdatapeaksfiltered[label] for label in labels]
p_value, F_p_value = statsutils.resampling_stats(samples, labels, show_ci=False)
print('\n p_values',p_value)
table = statsutils.create_latex_table(p_value, samples, labels)
print(table)

#Construction of Cummings Violin plot
ctrl = boxplotdatapeaksfiltered['PLKO']
samplesi = [numpy.array(boxplotdatapeaksfiltered[labels[i]]) for i in range(1,3)]
fig, ax, parts= statsutils.cumming_plot(numpy.array(ctrl), samplesi,labels=labelsnew[1:3],looklist=looks[1:3])
ax.set_title('FUS normalized spot intensity',fontsize=16)
fig.savefig(os.path.join(os.getcwd(),"Fusgranules","noMAP2","graphs",'violinplot.pdf'),transparent=True)
pyplot.show()

#Construction of standard boxplot            
fig,ax=pyplot.subplots(figsize=(12,8))
for key in conditions:
    ax.scatter(numpy.random.normal(position[key],scale=0.02,size=(len(boxplotdatapeaksfiltered[key]),1)),boxplotdatapeaksfiltered[key],color=look[key],label=key)
ax.boxplot(boxplotdatapeaks.values())
ax.set_xticklabels(boxplotdatapeaks.keys())
fig.savefig(os.path.join(os.getcwd(),"Fusgranules","noMAP2","graphs",'boxplot.pdf'),transparent=True)

pyplot.show()
            

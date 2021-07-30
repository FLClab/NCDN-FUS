# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:35:45 2021

@author: andde
"""
import os
import glob
import csv
import numpy
import matplotlib.pyplot as pyplot
from collections import OrderedDict
from scipy.stats import norm
from scipy.signal import find_peaks
import scipy
import statsutils
input_dir=os.path.expanduser("~/Desktop/Fusgranules/noMAP2/*.npy")
npylist=glob.glob(input_dir)

data={}
boxplotdata={}
boxplotdatastd={}
boxplotdatapeaks={}
boxplotdatadate={}

conditions=['PLKO','shNorbin01','shNorbin02']
labels=['PLKO','shNorbin01','shNorbin02']
labelsnew=['CTL','NCDN-KD1','NCDN-KD2']
#imnum=['numclustersPLKO','numclustersshNorbin01','numclustersshNorbin02']
imnum=['numclustersPLKO','numclustersshNorbin01','numclustersshNorbin02']
look={'shNorbin01':'lightgrey','shNorbin02':'dimgrey','PLKO':'blue'}
style={'190403':'solid','190416':'dashed','190920':'dashdot','2011067':'dotted','201107':'dotted','210412':'solid',}
position={'shNorbin01':3,'shNorbin02':2,'PLKO':1}
#position={'shNorbin01':2,'shNorbin02':3,'PLKO':1}

fig1, ax1 = pyplot.subplots(figsize=(12, 8))
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
    fig2, ax2 = pyplot.subplots(figsize=(12, 8))
    ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax3.set_ylabel('histogram', color=color)  # we already handled the x-label with ax1
    ax3.tick_params(axis='y', labelcolor=color)
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

                ax2.plot(numpy.sort(counts[ctr:ctr+im]), numpy.linspace(0, 1, im, endpoint=False), label=str(j)+key+date,color=look[key])
                countshist,bins,p=ax3.hist(numpy.sort(counts[ctr:ctr+im]), bins=300, density=True, alpha=0.2, label=str(j)+key+date,color=look[key])
                aa,loc,scale  = scipy.stats.skewnorm.fit(numpy.sort(counts[ctr:ctr+im]))
                best_fit_line = scipy.stats.skewnorm.pdf(bins,aa,loc,scale)
                ax3.plot(bins, best_fit_line,color=look[key])
                fig2.tight_layout()  # otherwise the right y-label is slightly clipped
                
                peaks, _ = find_peaks(best_fit_line,distance=500)
                if bins[peaks].shape==(0,):
                    print('OOPS no peak')
                    continue

                
                ax3.plot(bins[peaks],best_fit_line[peaks], "x",color=look[key])
                boxplotdatastd[key].append(bins[peaks][0])
                listcond.append(bins[peaks][0])
                #ax.plot(numpy.linspace(0, 1, im, endpoint=False)[peaks],best_fit_line[peaks], "x",color=look[key])
                ctr+=im
                ax2.legend()
                ax3.legend()
            #fig2.show()
                
                
            
                
            
            
            countshist,bins,p=ax.hist(counts, bins=300, density=True, alpha=0.2, label=key,color=look[key])
            aa,loc,scale  = scipy.stats.skewnorm.fit(counts)
            best_fit_line = scipy.stats.skewnorm.pdf(bins,aa,loc,scale)
            peaks, _ = find_peaks(best_fit_line,distance=500)
            if condcount==0:
                normfactor=bins[peaks][0]
                #normfactor=numpy.array(1.0)
                #normfactor.astype(float)
                print('I AM THE NORM',normfactor)


            ax.plot(bins, best_fit_line,color=look[key])
            ax.plot(bins[peaks],best_fit_line[peaks], "x",color=look[key])
            ax.set_title('Fus spot intensity '+date,fontsize=16)
            print(position[key],bins[peaks][0]/normfactor)
            print('std',[numpy.min(boxplotdatastd[key]),numpy.max(boxplotdatastd[key])])
            minv, maxv = -0.3,0.3
            spread=(maxv - minv)*numpy.random.rand() + minv
            ax1.scatter(position[key]+spread*numpy.ones(len(boxplotdatastd[key])),boxplotdatastd[key]/normfactor,color=look[key],s=10)
            ax1.errorbar(position[key]+spread,bins[peaks][0]/normfactor,numpy.std(boxplotdatastd[key]/normfactor),marker='o',ms=10,alpha=0.3, mfc=look[key],mec='k',ecolor='k',elinewidth=1, linestyle='None')
            ax.legend()
            datevaluesy.append(position[key]+spread)
            datevaluesx.append(bins[peaks][0]/normfactor)
            
            test=[val/normfactor for val in listcond]
            print(len(test))
            maxval=numpy.mean(test)+2*numpy.std(test)
            print(maxval)
            print(max(test))
            
            minval=numpy.mean(test)-2*numpy.std(test)
            print(minval)
            print(min(test))
            test1=[x for x in test if minval<x]
            print(len(test1))
            test2=[x for x in test1 if x<maxval]
            print(len(test2))
            #condcount+=1
            boxplotdatapeaks[key].extend(test2)
    idx   = numpy.argsort(datevaluesy)
    datevaluesx=numpy.array(datevaluesx)[idx]
    datevaluesy=numpy.array(datevaluesy)[idx]
    ax1.plot(datevaluesy,datevaluesx,'k',linestyle=style[date],label=date)
    ax1.legend()
    ax1.set_xticks([1,2,3])
    ax1.set_xticklabels(conditions)
    fig.savefig(os.path.expanduser("~/Desktop/Fusgranules/noMAP2/")+date+'fittedhistogram.pdf',transparent=True)
    fig2.savefig(os.path.expanduser("~/Desktop/Fusgranules/noMAP2/")+date+'cumulativecurve_perimage.pdf',transparent=True)
    

    try:
        shapiroPLKO=scipy.stats.shapiro(boxplotdata['PLKO'])
        PLKOAnderson=scipy.stats.anderson(boxplotdata['PLKO'],'norm')
    
    
        shapiroNorbin2=scipy.stats.shapiro(boxplotdata['shNorbin02'])
        Norbin2Anderson=scipy.stats.anderson(boxplotdata['shNorbin02'],'norm')
        print(date)
        print('SHAPIRO ')
    
        print('Norbin2',shapiroNorbin2)
        print('PLKO',shapiroPLKO)
        print('ANDERSON')
    
        print('Norbin2',Norbin2Anderson)
        print('PLKO',PLKOAnderson)
        
        KSNorbin2=scipy.stats.ks_2samp(boxplotdata['shNorbin02'],boxplotdata['PLKO'])
        print('Norbin2',KSNorbin2)
        print(len(boxplotdata['shNorbin01']), 'NORBIN1')
        shapiroNorbin1=scipy.stats.shapiro(boxplotdata['shNorbin01'])
        Norbin1Anderson=scipy.stats.anderson(boxplotdata['shNorbin01'],'norm')
        KSNorbin1=scipy.stats.ks_2samp(boxplotdata['shNorbin01'],boxplotdata['PLKO'])
        print('SHAPIRO ')
        print('Norbin1',shapiroNorbin1)
        print('ANDERSON')
        print('Norbin1',Norbin1Anderson)
        print('KS_2SAMP')
        print('Norbin1',KSNorbin1)
    except (KeyError,ValueError):
        continue
looks=['blue','lightgrey','dimgrey']
#looks2=['gold','gold','green','green','red','red','brown','brown','blue','blue']
boxplotdatapeaksfiltered={}
for key in conditions:
    print(key)
    test= boxplotdatapeaks[key]
    print(len(test))
    maxval=numpy.mean(test)+2*numpy.std(test)
    print(maxval)
    print(max(test))
    
    minval=numpy.mean(test)-2*numpy.std(test)
    print(minval)
    print(min(test))
    test1=[x for x in test if minval<x]
    print(len(test1))
    test2=[x for x in test1 if x<maxval]
    print(len(test2))
    boxplotdatapeaksfiltered[key] = test2







    

samples = [boxplotdatapeaksfiltered[label] for label in labels]
p_value, F_p_value = statsutils.resampling_stats(samples, labels, show_ci=False)
print('\n p_values',p_value)
table = statsutils.create_latex_table(p_value, samples, labels)
print(table)

    # Example using cumming plot
ctrl = boxplotdatapeaksfiltered['PLKO']



#fus8mean=numpy.mean(test)
#fig, ax = pyplot.subplots()
samplesi = [numpy.array(boxplotdatapeaksfiltered[labels[i]]) for i in range(1,3)]
fig, ax, parts= statsutils.cumming_plot(numpy.array(ctrl), samplesi,labels=labelsnew[1:3],looklist=looks[1:3])
ax.set_title('FUS normalized spot intensity',fontsize=16)
fig.savefig(os.path.expanduser("~/Desktop/Fusgranules/noMAP2/")+'violinplot.pdf',transparent=True)
pyplot.show()
            


fig,ax=pyplot.subplots(figsize=(12,8))
for key in conditions:
    ax.scatter(numpy.random.normal(position[key],scale=0.02,size=(len(boxplotdatapeaksfiltered[key]),1)),boxplotdatapeaksfiltered[key],color=look[key],label=key)

ax.boxplot(boxplotdatapeaks.values())
ax.set_xticklabels(boxplotdatapeaks.keys())
fig.savefig(os.path.expanduser("~/Desktop/Fusgranules/noMAP2/")+'boxplot.pdf',transparent=True)
fig1.savefig(os.path.expanduser("~/Desktop/Fusgranules/noMAP2/")+'linefig_.pdf',transparent=True)
pyplot.show()
            

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

input_dir=os.path.join(os.getcwd(),"Norbincumcurve",'Median',"*.npy")
npylist=glob.glob(input_dir)

data={}
boxplotdata={}
boxplotdatastd={}
boxplotdatapeaks={}

conditions=['PLKO','shFus315','shFus318']
#conditions=['PLKO','shNorbin02','shNorbin01','shFus315','shFus318']

look={'shNorbin01':'tan','shNorbin02':'dimgrey','PLKO':'brown','shFus315':'dimgrey','shFus318':'tan'}
style={'Dict190403':'solid','Dict190416':'dashed','Dict191213':'dashdot','Dict190313':'dotted','Dict201105':'solid'}
position={'shNorbin01':3,'shNorbin02':2,'PLKO':1,'shFus315':4,'shFus318':5}
#position={'shNorbin01':2,'shNorbin02':3,'PLKO':1}
bins= numpy.linspace(0, 30,500, endpoint=True)
fig, ax = pyplot.subplots(figsize=(12, 8))
for cond in conditions:
    boxplotdata[cond]=[]
    boxplotdatapeaks[cond]=[]
    boxplotdatastd[cond]=[]
colors=['red','gold','blue','green','magenta','cyan','black','lime']
condcount=0

for file in npylist:
    datevaluesx=[]
    datevaluesy=[]
    
    a=numpy.load(file,allow_pickle=True)
    date=os.path.basename(file).split('_')[0]
    print(date)
    for condcount,key in enumerate(conditions): 
        print(key)
        counts=a.item().get(key) #returns None if key not in dict
     
        #boxplotdatastd[key]=[]
        #boxplotdata[key]=[]
        listcond=[]
        if numpy.isnan(counts).any():
            print('Empty')
            continue
        if counts[0] == None :
            print('i am not here')
            continue
        else:
            
            if boxplotdata[key]==[]:
                boxplotdata[key]=counts
            else:
                boxplotdata[key]=numpy.dstack((boxplotdata[key],counts))
for cond in conditions:
    mean=numpy.mean(boxplotdata[cond][0,:,:],axis=1)
    standard_dev=numpy.std(boxplotdata[cond][0,:,:],axis=1)
    ax.plot(bins,mean,color=look[cond],label=cond)
            
#pl.plot(mean)
    ax.fill_between(bins, mean-standard_dev, mean+standard_dev,color=look[cond],alpha=0.2)
          
                
                
            
                
            
            

ax.legend()
ax.set_xlim([0,5])
ax.set_ylim([0,1.01])
    #ax1.set_xticks([1,2,3])
    #ax1.set_xticklabels(conditions)
fig.savefig(os.path.join(os.getcwd(),"Norbincumcurve","Median",'Cumulativecurve_shaded_4dates_median_noshNorb.pdf'),transparent=True)
    #fig2.savefig(os.path.expanduser("~/Desktop/Fusgranules/")+date+'cumulativecurve_perimage.pdf',transparent=True)

pyplot.show()
            

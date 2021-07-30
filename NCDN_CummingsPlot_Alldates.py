# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:35:45 2021

@author: andde
"""
import os
import glob
import csv
import numpy
import random
from scipy import stats
import matplotlib.pyplot as pyplot
from collections import OrderedDict
import statsutils


input_dir=os.path.join(os.getcwd(),"Norbincumcurve","Median","*.csv")
csvlist=glob.glob(input_dir)

data={}
boxplotdata={}
fig,ax=pyplot.subplots(figsize=(12,8))
#conditions=['shFus318','shFus315','shNorbin01','shNorbin02','PLKO']
#ticklabels=['shFus318','shFus315','shNorbin01','shNorbin02','PLKO']
conditions=['shFus315','shFus318','PLKO']
ticklabels=['shFus315','shFus318','PLKO']
labelsnew=['FUS-KD1','FUS-KD2','CTL']

#looks=['gold','green','red','brown','blue']
#looks2=['gold','gold','green','green','red','red','brown','brown','blue','blue']
looks=['lightgrey','dimgrey','blue']
looks2=['lightgrey','lightgrey','dimgrey','dimgrey','green','green','blue','blue']
for cond in conditions:
    boxplotdata[cond]=[]
colors=['tan','grey','black','lime','cyan']
j=0
for file in csvlist:
    with open(file) as csv_file:
        
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        
        for row in csv_reader:
            
            if row != []:
                values=[]
                name=row[0]
                means=row[1].split(' ')
                for mean in means:
                    string = mean.replace("[","") 
                    string=string.replace("]","") 
                    if string!="":
                        values.append(eval(string))
                data[name]=values
        
        i=1 
        boxplotlist=[]
        names=[]
        ilist=[]                   
        for  cond in conditions:
            #names.append(key)
            xrand=numpy.random.normal(i,scale=0.05,size=(len(data[cond]),1))
            boxplotdata[cond].extend(data[cond])
            ax.scatter(xrand,data[cond],label=os.path.basename(file).split(".")[0].split("_")[0],c=colors[j],s=50)
            ilist.append(i)
            i+=1
    
    
        
        #ax.legend()
        j+=1
handles, labels = pyplot.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
pyplot.legend(by_label.values(), by_label.keys(),fontsize=18)
labels, data = [*zip(*boxplotdata.items())]
box=ax.boxplot(data)
for patch, color in zip(box['boxes'], looks):
    patch.set_color(color)
    patch.set_linewidth(2)
for partname in ('caps','whiskers'):
     for patch, color in zip(box[partname], looks2):
         patch.set_color(color)
         patch.set_linewidth(2)
#ax.set_ylim([0,1.75])
pyplot.xticks(ilist ,ticklabels,fontsize=18)
pyplot.yticks(fontsize=18)
ax.set_title('Norbin median cytoplasmic intensity fold change',fontsize=18)
pyplot.savefig(os.path.join(os.getcwd(),"Norbincumcurve","Median",'Norbin median cytoplasmic intensity fold change_5dates_2std_noshNorb.pdf'),transparent=True)
boxplotdatafiltered={}
for key in labels:
    print(key)
    test= boxplotdata[key]
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
    boxplotdatafiltered[key] = test2

labels = ticklabels
samples = [boxplotdatafiltered[label] for label in labels]
p_value, F_p_value = statsutils.resampling_stats(samples, labels, show_ci=False)
print('\n p_values',p_value)
table = statsutils.create_latex_table(p_value, samples, labels)
print(table)

    # Example using cumming plot
ctrl = boxplotdata['PLKO']
#test= boxplotdata['shFus318']
#fus8std=numpy.std(test)
#fus8mean=numpy.mean(test)
#fig, ax = pyplot.subplots()
samplesi = [numpy.array(boxplotdatafiltered[labels[i]]) for i in range(2)]
fig, ax, parts= statsutils.cumming_plot(numpy.array(ctrl), samplesi,labels=labelsnew[0:2],looklist=looks[0:2])
ax.set_title('NCDN median cytoplasmic intensity',fontsize=16)
fig.savefig(os.path.join(os.getcwd(),"Norbincumcurve","Median",'violinplot_medians_5dates_2std.pdf'),transparent=True)
pyplot.show()
            

import pandas as pd 
import numpy as np 
import mip
import os
import matplotlib
import matplotlib.pyplot as plt
import csv
import math
import random
from scipy.stats.mstats import gmean
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)


#import planar
#from planar import BoundingBox as Bbox


# # # # # # # # # # # # # # PARAMETERS # # # # # # # # # # # # # # # # #

# number of panels desired in the lottery
M = 1000

# instances you want to run plots for
instances = ['sf_a_35', 'sf_b_20', 'sf_c_44', 'sf_d_40', 'sf_e_110', 'cca_75', 'hd_30', 'mass_24','nexus_170','obf_30','newd_40']
instance_names_dict = {'sf_a_35':'sf(a)', 'sf_b_20': 'sf(b)', 'sf_c_44':'sf(c)', 'sf_d_40':'sf(d)', 'sf_e_110':'sf(e)', 'cca_75':'cca', 'hd_30':'hd', 'mass_24':'mass','nexus_170':'nexus','obf_30':'obf','newd_40':'ndem'}

# objectives (can only run one at a time)
LEXIMIN = 0
MAXIMIN = 1
NASH = 0

# which rounding algorithms to analyze
ILP = 0
ILP_MINIMAX_CHANGE = 1               
BECK_FIALA = 1
RANDOMIZED = 1
RANDOMIZED_REPLICATES = 1000
THEORY = 1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


instance_names = []
for instance in instances:
    instance_names.append(instance[0:instance.rfind('_')])


def compute_theoretical_bounds_indloss(k,M,n,C):
    """ Computes two potentially best theoretical upper bounds on change in any marginal in terms of instance parameters
    """
    bounds = {}
    bounds['bf'] = k/M
    bounds['panelLP'] = math.sqrt((1 + math.log(2)/math.log(C))/2)*math.sqrt(C * math.log(C))/M + 1/M
    return bounds


def add_dset_to_plot(x,y,c,level):
    for xcoord in x:
        leftend = xcoord+0.05+0.9/3*level
        rightend = leftend + 0.9/3
        plt.plot([leftend,rightend],[y[xcoord],y[xcoord]],c)




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# plot comparing fairness (Figure 1 & corresponding figure in Appendix)
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


######### Build Data ###########
opt_plot_data = []
ilp_plot_data = []
bf_plot_data = []
rand_plot_data = []
rand_std_dev = []
theory_plot_data = []

for instance in instances:

    # construct all data
    categories_df = pd.read_csv('../input-data/'+instance+'/categories.csv')
    respondents_df = pd.read_csv('../input-data/'+instance+'/respondents.csv')
    n = len(respondents_df)
    k = int(instance[instance.rfind('_')+1:]) # get number of people wanted on panel

    # compute number of unique realized feature-vectors in pool
    categories = list(categories_df['category'].unique())
    C = len(respondents_df.groupby(categories))


    if MAXIMIN == 1:
        stub = '../intermediate_data/'+instance+'_m'+str(M)+'_maximin_'

    elif NASH == 1:
        stub = '../intermediate_data/'+instance+'_m'+str(M)+'_nash_'

    else:
        break
        

    OPT_probabilities_df = pd.read_csv(stub+'opt_probabilities.csv')
    OPT_marginals_df = pd.read_csv(stub+'opt_marginals.csv')

    committees = [[int(OPT_probabilities_df['committees'].values[i][11:-2].split(',')[j]) for j in range(len(OPT_probabilities_df['committees'].values[i][11:-2].split(',')))] for i in range(len(list(OPT_probabilities_df['committees'].values)))]


    OPT_marginals = OPT_marginals_df['marginals']

    if MAXIMIN==1:
        opt_plot_data.append(min(OPT_marginals))
    elif NASH==1:
        opt_plot_data.append(gmean(OPT_marginals))


    if ILP==1:
        ILP_marginals = pd.read_csv(stub+'ILProunded_marginals.csv')['marginals']

        if MAXIMIN==1:
            ilp_plot_data.append(min(ILP_marginals))
        elif NASH==1:
            ilp_plot_data.append(gmean(ILP_marginals))

    if BECK_FIALA==1:
        BF_marginals = pd.read_csv(stub+'BFrounded_marginals.csv')['marginals']  

        if MAXIMIN==1:
            bf_plot_data.append(min(BF_marginals))
        elif NASH==1:
            bf_plot_data.append(gmean(BF_marginals))

    if RANDOMIZED==1:
        rand_data = []
        for rep in range(RANDOMIZED_REPLICATES):
            RAND_marginals = pd.read_csv(stub+'RANDrounded_marginals.csv')['marginals']  

            if MAXIMIN==1:
                rand_data.append(min(RAND_marginals))
            elif NASH==1:
                rand_data.append(gmean(RAND_marginals))

        rand_plot_data.append(np.mean(rand_data))
        rand_std_dev.append(np.std(rand_data))

    if THEORY==1:
        bounds = compute_theoretical_bounds_indloss(k,M,n,C)

        if MAXIMIN==1:
            indloss = min(bounds['bf'],bounds['panelLP'])
            theory_plot_data.append(min(OPT_marginals) - indloss)

        elif NASH==1:
            indloss = k*min(bounds['bf'],bounds['panelLP'])
            theory_plot_data.append(gmean(OPT_marginals) - indloss)


###### Build plot ##########
if NASH == 1 or MAXIMIN ==1:

    x = list(range(len(instances)))
    plt.figure(figsize=(8,4))

    if THEORY==1:
        # shade the region between optimal fairness and lower bound
        for xcoord in x:
            y1 = opt_plot_data[xcoord]
            y2 = theory_plot_data[xcoord]
            plt.fill_between([xcoord+0.05,xcoord+0.95],[y1,y1],[y2,y2],color='k', alpha = 0.15)

            # add lines at bottom for bound
            plt.plot([xcoord+0.049,xcoord+0.951],[y2,y2],color='k',linewidth=1)

    if ILP==1:
        add_dset_to_plot(x,ilp_plot_data,'b',0)

    if RANDOMIZED==1:
        print("maximum standard deviation (over randomized rounding replicates) in objective across instances: "+str(max(rand_std_dev)))
        add_dset_to_plot(x,rand_plot_data,'g',1)

    if BECK_FIALA==1:
        add_dset_to_plot(x,bf_plot_data,'orange',2)


    xticks = list(np.arange(0.5,len(instances)+0.5,1))
    instance_names = [instance_names_dict[instance] for instance in instances]
    plt.xticks(xticks,labels=instance_names)
    plt.xlim(-0.1,len(instances)+0.1)

    if MAXIMIN==1:
        objname = 'IP-Maximin'
        ymax = 0.4
    elif NASH==1:
        objname = 'IP-NW'
        ymax = 0.6
    plt.ylim(0,ymax)
    plt.text(max(x)+1,ymax*0.925,'m = '+str(M), ha='right')
    

    legend_elements = [Line2D([0], [0], color='b', lw=1, label=objname),
                       Line2D([0], [0], color='g', lw=1, label='Pipage'),
                       Line2D([0], [0], color='orange', lw=1, label='Beck-Fiala'),
                       Line2D([0], [0], color='k', lw=1, label='Theoretical Bound')]

    plt.legend(handles=legend_elements,fontsize=8,loc='upper left')

    # write in text loss
    yshift1 = 0.0525*ymax/0.4
    yshift2 = 0.0375*ymax/0.4 
    yshift3 = 0.0225*ymax/0.4 
    yshift4 = 0.0075*ymax/0.4 
    for i in range(len(instances)):
        if ILP==1:
            ilp_loss = str(abs(round((opt_plot_data[i] - ilp_plot_data[i])*M,1)))
            plt.text(xticks[i],opt_plot_data[i]+yshift1,'-'+ilp_loss+'/m', horizontalalignment='center',fontsize=7.25,color='b')
        if RANDOMIZED==1:
            rand_loss = str(abs(round((opt_plot_data[i] - rand_plot_data[i])*M,1)))
            plt.text(xticks[i],opt_plot_data[i]+yshift2,'-'+rand_loss+'/m', horizontalalignment='center',fontsize=7.25,color='green')
        if BECK_FIALA == 1:
            bf_loss = str(abs(round((opt_plot_data[i] - bf_plot_data[i])*M,1)))
            plt.text(xticks[i],opt_plot_data[i]+yshift3,'-'+bf_loss+'/m', horizontalalignment='center',fontsize=7.25,color='orange')
        if THEORY == 1:
            theory_loss = str(abs(round((opt_plot_data[i] - theory_plot_data[i])*M,2)))
            plt.text(xticks[i],opt_plot_data[i]+yshift4,'-'+theory_loss+'/m', horizontalalignment='center',fontsize=7.25,color='k')
        

    # put objective-specific labels 
    if MAXIMIN==1:
        plt.ylabel('Maximin objective value')
        plt.savefig('../m'+str(M)+'maximin_algos_comparison.pdf',bbox_inches='tight')
        
    elif NASH==1:
        plt.ylabel('Nash Welfare objective value')
        plt.savefig('../m'+str(M)+'nash_algos_comparison.pdf',bbox_inches='tight')




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# plot comparing leximin distributions (Figure 2 & corresponding figures in appendix)
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


if LEXIMIN==1:

    # plot features: custom positions for each instance
         # each entry is a list composed of plot features, in the following order:
            #[0:3]= inset position: lower x position, lower y position, width, height (all as fractions of plot size)
            #[4] = fraction of support to cover with inset
    pf = {'sf_a_35':[0.3,0.4,0.5,0.4,0.2],
            'sf_b_20':[0.3,0.4,0.5,0.4,0.2],
            'sf_c_44':[0.3,0.5,0.4,0.4,0.25], 
            'sf_d_40':[0.3,0.4,0.5,0.4,0.2], 
            'sf_e_110':[0.3,0.4,0.5,0.4,0.05],
            'cca_75':[0.3,0.4,0.5,0.4,0.125],
            'hd_30':[0.3,0.4,0.5,0.4,0.25],
            'mass_24':[0.3,0.55,0.3,0.4,0.285],
            'nexus_170':[0.3,0.55,0.4,0.4,0.15],
            'obf_30':[0.3,0.4,0.5,0.4,0.2],
            'newd_40':[0.3,0.4,0.5,0.4,0.2]}


    for instance in instances:

        # get instance parameters
        pfi = pf[instance]
        respondents_df = pd.read_csv('../data_panelot/'+instance+'/respondents.csv')
        n = len(respondents_df)
        k = int(instance[instance.rfind('_')+1:]) # get number of people wanted on panel

        categories_df = pd.read_csv('../data_panelot/'+instance+'/categories.csv')
        categories = list(categories_df['category'].unique())
        C = len(respondents_df.groupby(categories))

        stub = '../intermediate_data/'+instance+'_m'+str(M)+'_leximin_'

        OPT_probabilities_df = pd.read_csv(stub+'opt_probabilities.csv')
        committees = [[int(OPT_probabilities_df['committees'].values[i][11:-2].split(',')[j]) for j in range(len(OPT_probabilities_df['committees'].values[i][11:-2].split(',')))] for i in range(len(list(OPT_probabilities_df['committees'].values)))]
        probabilities = OPT_probabilities_df['probabilities'].values

        # read in data
        marginals = list(pd.read_csv(stub + 'opt_marginals.csv')['marginals'].values)
        marginals_ILP_rounded = list(pd.read_csv(stub + 'ILP_MMC_rounded_marginals.csv')['marginals'].values)
        marginals_BF_rounded = list(pd.read_csv(stub + 'BFrounded_marginals.csv')['marginals'].values)
        rand_data = []
        for rep in range(RANDOMIZED_REPLICATES):
            marginals_RAND_rounded = (pd.read_csv(stub+'RANDrounded_marginals.csv')['marginals'].values)
            rand_data.append(marginals_RAND_rounded)
        marginals_RAND_rounded = np.mean(rand_data,axis=0)
        marginals_RAND_rounded_std = np.std(rand_data,axis=0)
        
        # reports maximum standard deviation, since it's so small that it's not being plotted
        print("in instance "+instance+", maximum standard deviation (over replicates of randomized rounding runs) of any marginal is:" + str(max(marginals_RAND_rounded_std)))

        # sort everything by optimal marginals
        marginals_ILP_rounded_sorted = [x for y, x in sorted(zip(marginals, marginals_ILP_rounded))]
        marginals_RAND_rounded_sorted = [x for y, x in sorted(zip(marginals, marginals_RAND_rounded))]
        marginals_BF_rounded_sorted = [x for y, x in sorted(zip(marginals, marginals_BF_rounded))]
        marginals_sorted = sorted(marginals)


        # make plot
        fig, ax1 = plt.subplots(figsize=(8,2))

        x = list(range(n+1))

        # shade in regions showing tightest bounds
        bounds = compute_theoretical_bounds_indloss(k,M,n,C)
        indloss = min(bounds['bf'],bounds['panelLP'])
        top= [m+indloss for m in marginals_sorted]
        bottom=[m-indloss for m in marginals_sorted]
        plt.fill_between(x,top+[top[-1]],bottom+[bottom[-1]],color='k', alpha = 0.15,step='post')

        # specify main plot
        ax1.plot(x,marginals_sorted + [marginals_sorted[-1]],'k',alpha=0.5,linewidth=1,drawstyle='steps-post')
        ax1.plot(x,marginals_ILP_rounded_sorted  + [marginals_ILP_rounded_sorted[-1]],'cornflowerblue',linestyle='--',linewidth=1,alpha=0.5,drawstyle='steps-post')
        ax1.plot(x,marginals_RAND_rounded_sorted  + [marginals_RAND_rounded_sorted[-1]],'g:',linewidth=1,alpha=0.5,drawstyle='steps-post')
        ax1.plot(x,marginals_BF_rounded_sorted  + [marginals_RAND_rounded_sorted[-1]],'orange',linestyle='-.',linewidth=1,alpha=0.5,drawstyle='steps-post')

        ax1.set_xlim(0,n+0.5)
        ax1.set_xticks([])
        ax1.set_ylabel('marginal probability')
        ax1.set_xlabel('agents sorted by marginal given by $p^*$')
        ax1.set_ylim([-0.01,1.01])

        # legend
        custom_lines = [Line2D([0], [0], color='black', lw=1),
                        Line2D([0], [0], color='cornflowerblue', lw=1, linestyle = '--'),
                        Line2D([0], [0], color='green', lw=1, linestyle= ':'),
                        Line2D([0], [0], color='orange', lw=1, linestyle='-.')]
        ax1.legend(custom_lines,['$p^*$','IP-Marginals','Pipage','Beck-Fiala'],loc='upper left',fontsize=8)

        # specify inset plot
        ax2 = plt.axes([0,0,1,1])
        ip = InsetPosition(ax1, pfi[0:4])
        ax2.set_axes_locator(ip)

        ax2.plot(x[:-1],marginals_sorted ,'k',linewidth=0.9,alpha=0.75,drawstyle='steps-post')
        ax2.plot(x[:-1],marginals_ILP_rounded_sorted,'cornflowerblue',linestyle='--',linewidth=0.9,alpha=0.75,drawstyle='steps-post')
        ax2.plot(x[:-1],marginals_RAND_rounded_sorted,'g:',linewidth=0.9,alpha=0.75,drawstyle='steps-post')
        ax2.plot(x[:-1],marginals_BF_rounded_sorted,'orange',linestyle='-.',linewidth=0.9,alpha=0.75,drawstyle='steps-post')

        # draw lines from box corners
        xmin = 0
        xmax = int(n*pfi[4])

        ax1.plot([0,pfi[0]*n],[marginals_sorted[0],pfi[1]],'k--',linewidth=0.75)
        ax1.plot([xmax,(pfi[0]+pfi[2])*n],[marginals_sorted[xmax],pfi[1]],'k--',linewidth=0.75)

        ymin = min([min(marginals_sorted[xmin:xmax]),min(marginals_ILP_rounded_sorted[xmin:xmax]),min(marginals_RAND_rounded_sorted[xmin:xmax]), min(marginals_BF_rounded_sorted[xmin:xmax])])
        ymax = max([max(marginals_sorted[xmin:xmax]),max(marginals_ILP_rounded_sorted[xmin:xmax]),max(marginals_RAND_rounded_sorted[xmin:xmax]), max(marginals_BF_rounded_sorted[xmin:xmax])])
        ax2.set_xlim(xmin,xmax)
        ax2.set_ylim(ymin*(0.99),ymax*(1.01))
        ax2.set_xticks([])

        # showing span of deviation
        ax2.set_yticks([])

        textx = (pfi[0]-0.045)*n
        texty = pfi[3]/2 + pfi[1] 
        spanlabel = str(int((ymax*1.01 - ymin*0.99)*1000))+'/m'
        ax1.text(textx,texty,spanlabel,ha = 'center',va='center',fontsize=8)
        ax1.plot([(pfi[0]-0.02)*n,(pfi[0]-0.01)*n],[pfi[1],pfi[1]],'k',linewidth=0.75)
        ax1.plot([(pfi[0]-0.02)*n,(pfi[0]-0.01)*n],[pfi[1]+pfi[3],pfi[1]+pfi[3]],'k',linewidth=0.75)
        ax1.arrow((pfi[0]-0.015)*n,(pfi[3]/2 + pfi[1]),0,pfi[3]/2-0.03,width=0.001/pfi[4],head_width=0.15/pfi[4],head_length=0.007)
        ax1.arrow((pfi[0]-0.015)*n,(pfi[3]/2 + pfi[1]),0,-(pfi[3]/2-0.03),width=0.001/pfi[4],head_width=0.15/pfi[4],head_length=0.007)

        plt.savefig('../m'+str(M)+'_'+instance+'_leximin_marginals_comparison.pdf',bbox_inches='tight')




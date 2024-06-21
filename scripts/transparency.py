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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


#import planar
#from planar import BoundingBox as Bbox


# # # # # # # # # # # # # # PARAMETERS # # # # # # # # # # # # # # # # #

# number of panels desired in the lottery
M = 1000

# instances you want to run plots for
instances = ['hd_30', 'sf_a_35', 'sf_c_44', 'sf_b_20', 'sf_d_40', 'mass_24', 'obf_30', 'sf_e_110', 'cca_75']
instance_names_dict = {'sf_a_35':'sf(a)', 'sf_b_20': 'sf(b)', 'sf_c_44':'sf(c)', 'sf_d_40':'sf(d)', 'sf_e_110':'sf(e)', 'cca_75':'cca', 'hd_30':'hd', 'mass_24':'mass','nexus_170':'nexus','obf_30':'obf','newd_40':'ndem'}
instance_ids = {'hd_30': 1, 'sf_a_35':2, 'sf_c_44':3, 'sf_b_20': 4, 'sf_d_40': 5, 'mass_24':6, 'obf_30': 7, 'sf_e_110':8, 'cca_75': 9}

# # objectives (can only run one at a time)
# LEXIMIN = 0
# MAXIMIN = 0
# NASH = 0
# GOLDILOCKS = 1

# which rounding algorithms to analyze
ILP = 0
ILP_MINIMAX_CHANGE = 0               
BECK_FIALA = 0
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


def add_dset_to_plot(x,y,c,level, std_dev = None):
    for xcoord in x:
        if xcoord >= len(y):
            break
        leftend = xcoord+0.05+0.9/2*level
        rightend = leftend + 0.9/2
        # if std_dev is not None then plot error bars
        if std_dev is not None:
            plt.errorbar((leftend+rightend)/2,y[xcoord],yerr=std_dev[xcoord],c=c)
        plt.plot([leftend,rightend],[y[xcoord],y[xcoord]],c)




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# plot comparing fairness (Figure 1 & corresponding figure in Appendix)
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


######### Build Data ###########
opt_plot_data_max = []
ilp_plot_data_max = []
rand_plot_data_max = []
rand_std_dev_max = []
theory_plot_data_max = []

opt_plot_data_min = []
ilp_plot_data_min = []
rand_plot_data_min = []
rand_std_dev_min = []
theory_plot_data_min = []

best_possible_min = []
best_possible_max = []
no_ilp_instances = ['sf_a_35', 'sf_e_110', 'cca_75']

for instance in instances:
    inst_ilp = instance not in no_ilp_instances
    print("instance: "+instance+" ilp: "+str(inst_ilp))
    # construct all data
    categories_df = pd.read_csv('../input-data/'+instance+'/categories.csv')
    respondents_df = pd.read_csv('../input-data/'+instance+'/respondents.csv')
    n = len(respondents_df)
    k = int(instance[instance.rfind('_')+1:]) # get number of people wanted on panel

    # compute number of unique realized feature-vectors in pool
    categories = list(categories_df['category'].unique())
    C = len(respondents_df.groupby(categories))
    print(f"instance {instance}, n: {n}, k: {k}, C: {C}, id: {instance_ids[instance]}")

    # if MAXIMIN == 1:
    #     stub = '../intermediate_data/'+instance+'_m'+str(M)+'_maximin_'

    # elif NASH == 1:
    #     stub = '../intermediate_data/'+instance+'_m'+str(M)+'_nash_'

    # elif GOLDILOCKS == 1:
    #     print("setting stub")
    #     stub = '../intermediate_data/dropped_0/'+instance+'/'+instance+'_m'+str(M)+'_goldilocks_p=50_gamma_1_scale=1_'
    # else:
    #     break
    maximin_stub = '../intermediate_data/dropped_0/'+instance+'/'+instance+'_m'+str(M)+'_maximin_'
    minimax_stub = '../intermediate_data/dropped_0/'+instance+'/'+instance+'_m'+str(M)+'_minimax_'
    maximin_min = float(np.min(pd.read_csv(maximin_stub + 'opt_marginals.csv')['marginals'].values))
    minimax_max = float(np.max(pd.read_csv(minimax_stub + 'opt_marginals.csv')['marginals'].values))
    best_possible_min.append(maximin_min)
    best_possible_max.append(minimax_max)    

    stub = '../intermediate_data/dropped_0/'+instance+'/'+instance+'_m'+str(M)+'_true_goldilocks_'
    OPT_probabilities_df = pd.read_csv(stub+'opt_probabilities.csv')
    OPT_marginals_df = pd.read_csv(stub+'opt_marginals.csv')
    print("got opt")
    committees = [[int(OPT_probabilities_df['committees'].values[i][11:-2].split(',')[j]) for j in range(len(OPT_probabilities_df['committees'].values[i][11:-2].split(',')))] for i in range(len(list(OPT_probabilities_df['committees'].values)))]


    OPT_marginals = OPT_marginals_df['marginals']

    opt_plot_data_min.append(min(OPT_marginals))
    opt_plot_data_max.append(max(OPT_marginals))


    if ILP==1:
        if inst_ilp:
            stub='../intermediate_data/dropped_0/ILProunded/'+instance+'/'+instance+'_m'+str(M)+'_true_goldilocks_'
            ILP_marginals = pd.read_csv(stub+'ILProunded_marginals.csv')['marginals']
            ilp_plot_data_max.append(max(ILP_marginals))
            ilp_plot_data_min.append(min(ILP_marginals))
            print("got ilp")
        else:
            ilp_plot_data_max.append(None)
            ilp_plot_data_min.append(None)

    if RANDOMIZED==1:
        rand_data = []
        stub = '../intermediate_data/dropped_0/pipage/'+instance+'/'+instance+'_m'+str(M)+'_true_goldilocks_'
        RAND_marginals = pd.read_csv(stub+'RANDrounded_marginals_full_stats.txt')  
        # append to rand_plot_data_max and rand_plot_data_min the values that are Avg Max Marg and Avg Min Marg
        rand_plot_data_max.append(RAND_marginals['Avg Max Marg'].values[0])
        rand_plot_data_min.append(RAND_marginals['Avg Min Marg'].values[0])
        rand_std_dev_max.append(RAND_marginals['Std Max Marg'].values[0])
        rand_std_dev_min.append(RAND_marginals['Std Min Marg'].values[0])
        print("got rand")

    if THEORY==1:
        bounds = compute_theoretical_bounds_indloss(k,M,n,C)
        theory_plot_data_max.append((max(OPT_marginals) + min(bounds['bf'],bounds['panelLP'])))
        theory_plot_data_min.append((min(OPT_marginals) - min(bounds['bf'],bounds['panelLP'])))
        print("got theory")


###### Build plot ##########
# if NASH == 1 or MAXIMIN ==1 or GOLDILOCKS == 1:
x = list(range(len(instances)))
plt.figure(figsize=(8,4))
# MAX = 0
# if MAX==1:
#     opt_plot_data = opt_plot_data_max
#     ilp_plot_data = ilp_plot_data_max
#     rand_plot_data = rand_plot_data_max
#     theory_plot_data = theory_plot_data_max
# else:
#     opt_plot_data = opt_plot_data_min
#     ilp_plot_data = ilp_plot_data_min
#     rand_plot_data = rand_plot_data_min
#     theory_plot_data = theory_plot_data_min
    
if THEORY==1:
    # shade the region between optimal fairness and lower bound
    for xcoord in x:
        y1 = opt_plot_data_max[xcoord]
        y2 = theory_plot_data_max[xcoord]
        plt.fill_between([xcoord+0.05,xcoord+0.95],[y1,y1],[y2,y2],color='k', alpha = 0.15)

        # add lines at bottom for bound
        plt.plot([xcoord+0.049,xcoord+0.951],[y2,y2],color='k',linewidth=1)
        
        y3 = opt_plot_data_min[xcoord]
        y4 = theory_plot_data_min[xcoord]
        plt.fill_between([xcoord+0.05,xcoord+0.95],[y3,y3],[y4,y4],color='k', alpha = 0.15)

        # add lines at bottom for bound
        plt.plot([xcoord+0.049,xcoord+0.951],[y4,y4],color='k',linewidth=1)
        
        # plot best possible mins and maxes at this coordinate with dashed lines
        plt.plot([xcoord+0.05,xcoord+0.95],[best_possible_min[xcoord],best_possible_min[xcoord]],color="firebrick",linestyle='dashed',linewidth=1)
        plt.plot([xcoord+0.05,xcoord+0.95],[best_possible_max[xcoord],best_possible_max[xcoord]],color="cornflowerblue",linestyle='dashed',linewidth=1)
        

if ILP==1:
    print("plotting ILP with "+str(ilp_plot_data_max)+" and "+str(ilp_plot_data_min))
    # make ilp_x_coords the indices of instances where ilp was run
    # ilp_x_coords = [xcoord for xcoord in x if instances[xcoord] not in no_ilp_instances]
    # print("x", x)
    # print("ilp_x_coords", ilp_x_coords)
    add_dset_to_plot(x,ilp_plot_data_max,'b',0)
    add_dset_to_plot(x,ilp_plot_data_min,'b',0)

if RANDOMIZED==1:
    # print("maximum standard deviation (over randomized rounding replicates) in objective across instances: "+str(max(rand_std_dev)))
    add_dset_to_plot(x,rand_plot_data_max,'g',1, std_dev = rand_std_dev_max)
    add_dset_to_plot(x,rand_plot_data_min,'g',1, std_dev = rand_std_dev_min)


xticks = list(np.arange(0.5,len(instances)+0.5,1))
instance_names = [instance_names_dict[instance] for instance in instances]
plt.xticks(xticks,labels=[instance_ids[instance] for instance in instances],fontsize=10)
plt.xlim(-0.1,len(instances)+0.1)

# if MAXIMIN==1:
#     objname = 'IP-Maximin'
#     ymax = 0.4
# elif NASH==1:
#     objname = 'IP-NW'
#     ymax = 0.6
# else:
#     objname = 'Goldilocks'
#     ymax = 0.4
ymax = 1
plt.ylim(0,1)
plt.text(x[1],ymax*0.925,'m = '+str(M), ha='right')


legend_elements = [Line2D([0], [0], color='firebrick', lw=1, linestyle='dashed', label='Best Possible Min'),
                    Line2D([0], [0], color='cornflowerblue', lw=1, linestyle='dashed', label='Best Possible Max')]

if THEORY==1:
    legend_elements.append(Line2D([0], [0], color='k', lw=1, label='Theoretical Bound'))

if ILP==1:
    print("ILP legend value", ILP)
    legend_elements.append(Line2D([0], [0], color='b', lw=1, label='ILP'))
    print("adding legend element")

if RANDOMIZED==1:
    legend_elements.append(Line2D([0], [0], color='g', lw=1, label='Pipage'))

plt.legend(handles=legend_elements,fontsize=10,loc='upper right')

ymax = 1
# # write in text loss
yshift1 = 0.0525*ymax/0.4
yshift2 = 0.0375*ymax/0.4 
yshift3 = 0.0225*ymax/0.4 

plt.ylabel('Extremal Marginals', fontsize=10)
plt.xlabel('Instance', fontsize=10)
# if path doesn't exist, create it
if not os.path.exists('../cr_paper_plots/transparency'):
    os.makedirs('../cr_paper_plots/transparency')

plt.savefig('../cr_paper_plots/transparency/true_goldilocks.pdf',bbox_inches='tight')

# # put objective-specific labels 
# if MAXIMIN==1:
#     plt.ylabel('Maximin objective value')
#     plt.savefig('../m'+str(M)+'maximin_algos_comparison.pdf',bbox_inches='tight')
    
# elif NASH==1:
#     plt.ylabel('Nash Welfare objective value')
#     plt.savefig('../m'+str(M)+'nash_algos_comparison.pdf',bbox_inches='tight')
    
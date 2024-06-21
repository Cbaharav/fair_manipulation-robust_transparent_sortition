import pandas as pd 
import numpy as np 
import mip
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import csv
import math
import random
from scipy.stats.mstats import gmean
from matplotlib.lines import Line2D
from time import time
from scipy.stats.mstats import gmean
from matplotlib.ticker import MaxNLocator
from paper_data_analysis import build_dictionaries

alg_colors = {"minimax": "cornflowerblue", "maximin": "firebrick", "leximin": "red", "legacy": "limegreen", "nash": "gold", "goldilocks": "darkorchid", "true_goldilocks":"gold"}
instances = ['hd_30', 'sf_a_35', 'sf_c_44', 'sf_b_20', 'sf_d_40', 'mass_24', 'obf_30', 'sf_e_110', 'cca_75']
instance_ids = {'hd_30': 1, 'sf_a_35':2, 'sf_c_44':3, 'sf_b_20': 4, 'sf_d_40': 5, 'mass_24':6, 'obf_30': 7, 'sf_e_110':8, 'cca_75': 9}
group1 = ['hd_30', 'sf_a_35', 'sf_c_44']
group2 = ['sf_b_20', 'sf_d_40', 'mass_24']
group3 = ['obf_30', 'sf_e_110', 'cca_75']
# objs = ['maximin', 'minimax', 'goldilocks_p=50_gamma_1_scale=1000', 'goldilocks_p=50_gamma_2_scale=1000', 'goldilocks_p=50_gamma_3_scale=1000']
M = 1000

def gl_to_string(p, gamma, scale):
    return f"goldilocks_p={p}_gamma_{gamma}_scale={scale}"

def reformat_name(name, gl_details = False):
    if "goldilocks" not in name:
        return name
    if "true" in name:
        if gl_details:
            return "True GL: gamma 1"
        else:
            return "goldilocks"
    if gl_details:
        # name is of the form goldilocks_p=50_gamma_1_scale=1000
        p = name[name.find("p=")+2:name.find("_gamma")]
        gamma = name[name.find("gamma_")+6:name.find("_scale")]
        scale = name[name.find("scale=")+6:]
        return f"GL: gamma {gamma}, scale={scale}"
    else:
        return "goldilocks"

def min_max_table(algorithms, num_dropped_feats=0):
    '''Produce a table comparing the minimum and maximum selection probabilities of different algorithms across all of the instances with respect to minimax and maximin'''
    data = []
    for instance in instances:
        maximin_stub = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_maximin_'
        minimax_stub = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_minimax_'
        maximin_min = np.min(pd.read_csv(maximin_stub + 'opt_marginals.csv')['marginals'].values)
        minimax_max = np.max(pd.read_csv(minimax_stub + 'opt_marginals.csv')['marginals'].values)
        
        instance_dict = {"Instance": instance_ids[instance]}
        for algorithm in algorithms:
            stub = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_'+algorithm+'_'
            marginals_df = pd.read_csv(stub + 'opt_marginals.csv')
            marginals = marginals_df['marginals'].values
            # algorithm = reformat_name(algorithm, gl_details=True)
            instance_dict[algorithm] = (round(np.min(marginals)/maximin_min, 2), round(np.max(marginals)/minimax_max, 2))
        data.append(instance_dict)
    df = pd.DataFrame(data)
    df.to_csv(f"../cr_paper_plots/tables/min_max_table_gammas.csv", index=False)
    print(df.to_latex(index=False))
    
def plot_manip_exp(instances, pool_copies=[1,2,3]):
    # plot manipulability for goldilocks, maximin, and minimax for pool pool_copies
    for instance in instances:
        fig, ax = plt.subplots(1, 1)
        for algorithm in ["maximin", "minimax", "goldilocks"]:
            manip_data = []
            for pool_copy in pool_copies:
                manip_file = '../paper_manip_exp/vary_pool_'+instance+'/'+algorithm+'_'+str(pool_copy)+'.csv'
                manip_df = pd.read_csv(manip_file)
                # get the deviation benefit from manip_df and add to manip_data
                if 'Deviation Benefit' not in manip_df.columns:
                    manip_data.append(manip_df['deviation'].values[0])
                else: 
                    manip_data.append(manip_df['Deviation Benefit'].values[0])
            ax.plot(pool_copies, manip_data, label=algorithm, linewidth=2, color = alg_colors[algorithm], alpha=0.85)
        # make x axis only have integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # make y axis from 0 to 1
        ax.set_ylim(0, 1)
        ax.set_title(f"MU Manipulability on {instance}")
        ax.set_xlabel("Pool Copies")
        ax.set_ylabel("Maximum Deviation Benefit")
        ax.legend()
        fig.savefig(f"../paper_plots/manipulability/{instance}_manipulability.pdf")
        
def paper_ssb_drops(instances):
    algorithms = ["maximin", "minimax", "true_goldilocks"]
    # instances = ['hd_30', 'sf_a_35', 'sf_c_44']
    dropped_feats = [0, 1, 2, 3]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(20, 10), constrained_layout=True)
    for instance, ax in zip(instances, [ax1, ax2, ax3]):
        for algorithm in algorithms:
            mins = []
            maxes = []
            for num_dropped_feats in dropped_feats:
                stub = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_'+algorithm+'_'
                marginals_df = pd.read_csv(stub + 'opt_marginals.csv')
                marginals = marginals_df['marginals'].values
                mins += [np.min(marginals)]
                maxes += [np.max(marginals)]
            if algorithm == "maximin" or algorithm == "leximin":
                leximin_min = mins
            if algorithm == "minimax":
                minimax_max = maxes
            l1 = ax.plot(dropped_feats, maxes, label=reformat_name(algorithm,gl_details=False), linewidth=4, color = alg_colors[reformat_name(algorithm)], alpha=0.85)
            ax.plot(dropped_feats, mins, label='_nolegend_', linestyle='--', color=l1[0].get_color(), linewidth=4, alpha=0.85)
        ax.set_title(f"Instance {instance_ids[instance]}", fontsize=24)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.fill_between(dropped_feats, leximin_min, minimax_max, color='grey', alpha=0.2)
        ax.tick_params(axis='both', labelsize=24)
    for ax in [ax1, ax2, ax3]:
        ax.label_outer()
    ax1.set_ylabel("Probability", fontsize=26, labelpad=20)
    ax2.set_xlabel("Number of Features Dropped", fontsize=26, labelpad=20)
    ax2.legend(loc='upper right', fontsize=26)
    # add a title far enough away from the subplots
    # fig.suptitle("Extremal Probabilities", fontsize=30)
    # if file doesn't exist, create folders
    if not os.path.exists("../cr_paper_plots/min_ssb_drops"):
        os.makedirs("../cr_paper_plots/min_ssb_drops")
    fig.savefig(f"../cr_paper_plots/min_ssb_drops/feature_drops_together_{instances}.pdf")

def paper_manip():
    algorithms = ["maximin", "minimax", "goldilocks"]
    instances = ['hd_30', 'sf_a_35', 'sf_c_44']
    pool_copies = [1,2]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(20, 7), constrained_layout=True)
    ax1.set_ylim(0, 1)
    for instance, ax in zip(instances, [ax1, ax2, ax3]):
        for algorithm in algorithms:
            manip_data = []
            for pool_copy in pool_copies:
                manip_file = '../paper_manip_exp/vary_pool_'+instance+'/'+algorithm+'_'+str(pool_copy)+'.csv'
                if algorithm == "goldilocks":
                    manip_file = '../CR_parallel_manip_output/vary_pool_'+instance+'/true_goldilocks_'+str(pool_copy)+'.csv'
                manip_df = pd.read_csv(manip_file)
                # get the deviation benefit from manip_df and add to manip_data
                if 'Deviation Benefit' not in manip_df.columns:
                    manip_data.append(manip_df['deviation'].values[0])
                else: 
                    manip_data.append(manip_df['Deviation Benefit'].values[0])
            ax.plot(pool_copies, manip_data, label=algorithm, linewidth=4, color = alg_colors[algorithm], alpha=0.85)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"Instance {instance_ids[instance]}", fontsize=24)
        ax.tick_params(axis='both', labelsize=24)
    for ax in [ax1, ax2, ax3]:
        ax.label_outer()
    ax1.set_ylabel("Maximum Deviation Benefit", fontsize=26, labelpad=20)
    ax2.set_xlabel("Pool Copies", fontsize=26, labelpad=20)
    ax3.legend(loc='lower left', fontsize=26)
    # fig.suptitle("MU Manipulability", fontsize=30)
    fig.savefig(f"../cr_paper_plots/manipulability/manipulability_together.pdf")
        
# most general version
def plot_ssb_drops(algorithms, instances, dropped_feats=[0]):
    # make two different plots, one for the minimum selection probability and one for the maximum selection probability
    # for each algorithm, plot the minimum and maximum selection probability over different numbers of dropped features
    for instance in instances:
        fig, ax = plt.subplots(1, 1)
        leximin_min = None
        minimax_max = None
        for alg in algorithms:
            mins = []
            maxes = []
            for num_dropped_feats in dropped_feats:
                stub = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_'+alg+'_'
                marginals_df = pd.read_csv(stub + 'opt_marginals.csv')
                marginals = marginals_df['marginals'].values
                mins += [np.min(marginals)]
                maxes += [np.max(marginals)]
            if alg == "leximin":
                leximin_min = mins
            if alg == "minimax":
                minimax_max = maxes
            l1 = ax.plot(dropped_feats, maxes, label=reformat_name(alg,gl_details=False), linewidth=2, color = alg_colors[reformat_name(alg)], alpha=0.85)
            ax.plot(dropped_feats, mins, label='_nolegend_', linestyle='--', color=l1[0].get_color(), linewidth=2, alpha=0.85)
        # add a title
        ax.set_title(f"Extremal Probabilities on {instance}")
        ax.fill_between(dropped_feats, leximin_min, minimax_max, color='grey', alpha=0.2)    
        ax.set_xlabel("Number of Features Dropped")
        ax.set_ylabel("Probability")
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(f"../paper_plots/min_ssb_drops/{instance}_min_ssb_drops.pdf")
    
# def plot_probability_allocations(instance_name, marginals) -> None:
#     """Produce line graph comparing the probability allocations of different algorithms on the same instance.
#     For each algorithm, the selection probabilities are sorted in order of increasing selection probability.
#     Plot is created at ./analysis/[instance_name]_[k]_prob_allocs.pdf and a table with raw data for the plot is created
#     at ./analysis/[instance_name]_[k]_prob_allocs_data.csv 
#     """
    
#     data = []
#     plot_data = []
#     n = len(marginals["leximin"]) # kinda a jank way to get n
#     k = int(instance_name[instance_name.rfind('_')+1:])
#     # legacy_alloc = legacy_probabilities(instance_name, 1000, random_seed=10)
#     for algorithm, alloc in marginals.items():
#         # Go through agents in order of increasing selection probability
#         sorted_agents = sorted(alloc)
#         for i, prob in enumerate(sorted_agents):
#             data.append({"algorithm": algorithm, "percentile of pool members": i / n * 100,
#                          "selection probability": prob})
#             if i == n - 1:  # needed to draw the last step
#                 plot_data.append({"algorithm": algorithm, "percentile of pool members": 100,
#                                   "selection probability": prob})

#     # output_directory = "plots"
#     df = pd.DataFrame(data)
#     df.to_csv(f"plots/{instance_name}_prob_allocs_data.csv", index=False)
#     df = pd.DataFrame(data + plot_data)

#     max_percentile = 100
#     restricted_df = df[df['percentile of pool members'] <= max_percentile]
#     fig = sns.relplot(data=restricted_df, x="percentile of pool members",
#                       y='selection probability', hue="algorithm", kind="line",
#                       drawstyle='steps-post', height=6, aspect=1.8, legend=False)
#     fig.set_axis_labels("percentile of pool members (by selection probability)", "selection probability")

#     plt.xlim(left=0.0, right=max_percentile)
#     plt.ylim(bottom=0.0)
#     # add legend where each line is labeled with the algorithm name
#     plt.legend(title="Algorithm", labels=list(marginals.keys()), fontsize=12)
#     # add title to the plot 
#     plt.title(f"Probability allocations on instance {instance_name}")
    
#     # confidence intervals for LEGACY probability allocation
#     ax = fig.ax
#     # restricted_df = restricted_df[restricted_df["algorithm"] == "Legacy"]
#     # restricted_df.sort_values(by="percentile of pool members", inplace=True)
#     # times_selected = (10000 * restricted_df["selection probability"]).round()
#     # # 99% Jeffreys intervals, i.e., the 0.5th and the 99.5th quantile of Beta(1/2 + successes, 1/2 + failures), with
#     # # exceptions when the number of successes or failures is 0:
#     # # Brown, L. D., Cai, T. T. & DasGupta, A. Interval estimation for a binomial proportion. Statistical science 16,
#     # # 101–117 (2001).
#     # lower = beta.ppf(.005, .5 + times_selected, .5 + (10000 - times_selected))
#     # lower = np.where(times_selected == 0, 0., lower)
#     # upper = beta.ppf(.995, .5 + times_selected, .5 + (10000 - times_selected))
#     # upper = np.where(times_selected == 10000, 1., upper)
#     # ax.fill_between(restricted_df["percentile of pool members"], lower, upper,
#     #                 alpha=0.25, edgecolor='none', facecolor="#2077B4", step="post")

#     # plt.legend((ax.lines[0], fig.ax.lines[1]), ('LEGACY', 'LEXIMIN'), fontsize=12)

#     # ax.lines[2].set_linestyle("dotted")
#     # ax.lines[2].set_color("g")
#     # ax.set_xlim(left=0.)
#     # ax.set_ylim(bottom=0)

#     # equalized probability label, minimum probability line, and rectangle
#     ax.text(1, 1.03 * k / n, 'equalized probability (k/n)', color='g')
#     ax.axhline(y=k/(n*1.0), xmin=0, xmax=max_percentile, linestyle='--', color='black', linewidth=0.5)
#     # rect = patches.Rectangle((0, 0), share_below_leximin_min * 100, leximin_min, edgecolor='none', facecolor="black",
#     #                          alpha=0.1)
#     # ax.add_patch(rect)

#     # ax.set_title(f"Probability allocations on instance {instance_name}")

#     plot_path = f"plots/{instance_name}_{time()}.pdf"
#     fig.savefig(plot_path)
#     # return plot_path
    
def compare_gl_gammas(instance, ps, scales, gammas) -> None:
    """Produce line graph comparing the probability allocations of different algorithms on the same instance.
    For each algorithm, the selection probabilities are sorted in order of increasing selection probability.
    Plot is created at ./analysis/[instance_name]_[k]_prob_allocs.pdf and a table with raw data for the plot is created
    at ./analysis/[instance_name]_[k]_prob_allocs_data.csv 
    """
    
    data = []
    plot_data = []
    marginals = {}
    stub_maximin = '../intermediate_data/'+instance+'/'+instance+'_m'+str(M)+'_maximin_'
    stub_minimax = '../intermediate_data/'+instance+'/'+instance+'_m'+str(M)+'_minimax_'
    marginals_df_maximin = pd.read_csv(stub_maximin + 'opt_marginals.csv')
    marginals["maximin"] = marginals_df_maximin['marginals'].values
    marginals_df_minimax = pd.read_csv(stub_minimax + 'opt_marginals.csv')
    marginals["minimax"] = marginals_df_minimax['marginals'].values
    stub_og_gl = '../intermediate_data/'+instance+'/'+instance+'_m'+str(M)+'_goldilocks_'
    marginals_df_gl = pd.read_csv(stub_og_gl + 'opt_marginals.csv')
    marginals["GL: p=50 gamma=10000"] = marginals_df_gl['marginals'].values
    
    for p in ps:
        for gamma_scale in scales:
            for gamma_label in gammas:
                stub = '../intermediate_data/'+instance+'/'+instance+'_m'+str(M)+'_goldilocks_'+f'p={p}_{gamma_label}_scale={gamma_scale}_opt_'
                marginals_df = pd.read_csv(stub + 'marginals.csv')
                alg_marginals = marginals_df['marginals'].values
                marginals[f'GL: p={p} gamma={gamma_label} scale={gamma_scale}'] = alg_marginals
    
    n = len(marginals["minimax"]) # kinda a jank way to get n
    k = int(instance[instance.rfind('_')+1:])
    # legacy_alloc = legacy_probabilities(instance_name, 1000, random_seed=10)
    for algorithm, alloc in marginals.items():
        # Go through agents in order of increasing selection probability
        sorted_agents = sorted(alloc)
        for i, prob in enumerate(sorted_agents):
            data.append({"algorithm": algorithm, "percentile of pool members": i / n * 100,
                         "selection probability": prob})
            if i == n - 1:  # needed to draw the last step
                plot_data.append({"algorithm": algorithm, "percentile of pool members": 100,
                                  "selection probability": prob})

    # output_directory = "plots"
    df = pd.DataFrame(data)
    df.to_csv(f"plots/{instance}_compare_gl_gammas_scales={scales}.csv", index=False)
    df = pd.DataFrame(data + plot_data)

    max_percentile = 100
    restricted_df = df[df['percentile of pool members'] <= max_percentile]
    fig = sns.relplot(data=restricted_df, x="percentile of pool members",
                      y='selection probability', hue="algorithm", kind="line",
                      drawstyle='steps-post', height=6, aspect=1.8, legend=False)
    fig.set_axis_labels("percentile of pool members (by selection probability)", "selection probability")

    plt.xlim(left=0.0, right=max_percentile)
    plt.ylim(bottom=0.0)
    # add legend where each line is labeled with the algorithm name
    plt.legend(title="Algorithm", labels=list(marginals.keys()), fontsize=12)
    # add title to the plot 
    plt.title(f"Probability allocations on instance {instance}")
    
    # confidence intervals for LEGACY probability allocation
    ax = fig.ax

    # equalized probability label, minimum probability line, and rectangle
    ax.text(1, 1.03 * k / n, 'equalized probability (k/n)', color='g')
    ax.axhline(y=k/(n*1.0), xmin=0, xmax=max_percentile, linestyle='--', color='black', linewidth=0.5)
    # rect = patches.Rectangle((0, 0), share_below_leximin_min * 100, leximin_min, edgecolor='none', facecolor="black",
    #                          alpha=0.1)
    # ax.add_patch(rect)

    # ax.set_title(f"Probability allocations on instance {instance_name}")

    plot_path = f"plots/{instance}_compare_gl_gammas_scales={scales}.csv.pdf"
    fig.savefig(plot_path)
    # return plot_path

def prob_stats_plots(algorithms, instances, to_plot=["gini", "geom"], num_dropped_feats=0, tables = False):
    gini_data = {}
    geom_data = {}
    var_data = {}
    
    for algorithm in algorithms:
        gini_alg = []
        geom_alg = []
        var_alg = []
        for instance in instances:
            stub = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_'+algorithm+'_'
            marginals_df = pd.read_csv(stub + 'opt_marginals.csv')
            marginals = marginals_df['marginals'].values
            gini, geometric_mean, mini, variance = compute_prob_allocation_stats(marginals, algorithm == "LEGACY")
            gini_alg.append(round(100*gini, 0))
            geom_alg.append(round(100*geometric_mean, 1))
            var_alg.append(round(variance, 4))
        algorithm = reformat_name(algorithm)
        gini_data[algorithm] = gini_alg
        geom_data[algorithm]  = geom_alg
        var_data[algorithm] = var_alg
    
    data = {"gini": gini_data, "geom": geom_data, "var": var_data}
    # if tables:
    #     gini_df = pd.DataFrame(gini_data)
    #     geom_df = pd.DataFrame(geom_data)
    #     gini_df.to_csv(f"tables/gini_table.csv", index=False)
    #     geom_df.to_csv(f"tables/geom_table.csv", index=False)
    #     print(gini_df.to_latex(index=False))
    #     print("********************")
    #     print(geom_df.to_latex(index=False))
    for vals in to_plot:
        plot_data = data[vals]
        x = np.arange(len(instances))  # the label locations
        width = 0.18  # the width of the bars
        multiplier = 0
        
        fig, ax = plt.subplots(figsize=(20, 9), constrained_layout=True)
        
        for alg, val in plot_data.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, val, width, label=alg, color=alg_colors[alg])
            # ax.bar_label(rects, padding=3, rotation='vertical', fontsize=16)
            multiplier += 1

        # Increase font size of labels and title
        ax.tick_params(axis='both', labelsize=24)
        if vals == "gini":
            ax.set_ylabel("Gini Coefficient (%)", fontsize=26)
            # ax.set_title('Gini Coefficients', fontsize=30)
            ax.set_ylim(0, 119)
        elif vals == "geom":
            ax.set_ylabel("Geometric Mean (%)", fontsize=26)
            # ax.set_title('Geometric Mean', fontsize=30)
            ax.set_ylim(0, 50)
        # ax.set_ylabel(vals, fontsize=16)
        ax.set_xlabel('Instance', fontsize=26)

        # Increase font size of x-axis tick labels
        ax.set_xticks(x + width)
        ax.set_xticklabels([instance_ids[instance] for instance in instances], fontsize=24)

        ax.legend(loc='upper left', fontsize=26)

        plot_path = f"../cr_paper_plots/fairness/{vals}.pdf"
        fig.savefig(plot_path)

def check_anon(instances, algorithm, num_dropped_feats=0):
    for instance in instances:
        categories_df = pd.read_csv('../input-data/'+instance+'/categories.csv')
        respondents_df = pd.read_csv('../input-data/'+instance+'/respondents.csv')
        n = len(respondents_df)
        k = int(instance[instance.rfind('_')+1:])        

        number_people_wanted = int(instance[instance.rfind('_')+1:]) # get number of people on panel from instance name
        categories, people, columns_data = build_dictionaries(categories_df,respondents_df, k, dropped_feats=num_dropped_feats)
        F = len(categories)
        
        ordered_features = list(categories.keys())
        people_tups = {}
        unique_fvs = set()
        for i in range(n):
            feature_vector = [0]*F
            for f_ind in range(F):
                feature = ordered_features[f_ind]
                feature_value = people[i][feature]
                feature_vector[f_ind] = feature_value
            feature_vector = tuple(feature_vector)
            unique_fvs.add(feature_vector)
            people_tups[i] = feature_vector
            
        
        stub = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_'+algorithm+'_'
        marginals_df = pd.read_csv(stub + 'opt_marginals.csv')
        marginals = marginals_df['marginals'].values
        anon_flag = True
        for tup in unique_fvs:
            people_with_tup = [i for i in range(n) if people_tups[i] == tup]
            # tup_marginals = [marginals[i] for i in people_with_tup]
            # check if all values in tup_marginals are the same
            first_marg = marginals[people_with_tup[0]]
            # print(f"TUPLE: {tup}, PEOPLE: {people_with_tup}, MARGINALS: {[marginals[i] for i in people_with_tup]}")
            for i in people_with_tup:
                if abs(marginals[i] - first_marg) > 0.015:
                    print(f"{instance} NOT ANON")
                    print(f"person {i} has far marginal ({marginals[i]}) from person {people_with_tup[0]} ({first_marg})")
                    anon_flag = False

        if anon_flag: print(f"{instance} ANON")
        # return True

def number_unique_fvs(instances):
    for instance in instances:
        categories_df = pd.read_csv('../input-data/'+instance+'/categories.csv')
        respondents_df = pd.read_csv('../input-data/'+instance+'/respondents.csv')
        unique_fvs = set()
        for nationbuilder_id in list(respondents_df['nationbuilder_id'].values):
            fv = []
            for category in list(categories_df['category'].unique()):
                fv.append(respondents_df[respondents_df['nationbuilder_id']==nationbuilder_id][category].values[0])
            fv = tuple(fv)
            unique_fvs.add(fv)
        print(f"instance: {instance}, num unique fvs: {len(unique_fvs)}, num people: {len(list(respondents_df['nationbuilder_id'].values))}")
    
def compute_prob_allocation_stats(alloc, cap_for_geometric_mean):
    """For an probability allocation, compute three measures of inequality: the Gini coefficient, the geometric mean,
    and the minimum selection probability.
    If `cap_for_geometric_mean` is True, probabilities below 1/10,000 are set to 1/10,000 before the calculation of the
    geometric mean to prevent the geometric mean from becoming zero. As described in the Methods section, we only give
    this advantage to the LEGACY benchmark.
    """
    n = len(alloc)
    k = round(sum(alloc))

    # selection probabilities in increasing order
    sorted_probs = sorted(alloc)
    # Formulation for Gini coefficient adapted from:
    # Damgaard, C., & Weiner, J. (2000). Describing inequality in plant size or fecundity. Ecology, 81(4), 1139-1142.
    gini = sum((2 * i - n + 1) * prob for i, prob in enumerate(sorted_probs)) / (n * k)

    if cap_for_geometric_mean:
        capped_probs = [max(prob, 1 / 10000) for prob in alloc]
        geometric_mean = gmean(capped_probs)
    else:
        geometric_mean = gmean(sorted_probs)

    mini = min(sorted_probs)
    
    variance = np.var(sorted_probs)

    return gini, geometric_mean, mini, variance

def plot_cdf(instance, num_dropped_feats=0):
    k = int(instance[instance.rfind('_')+1:])
    
    leximin_stub = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_leximin_'
    lex_marginals_df = pd.read_csv(leximin_stub + 'opt_marginals.csv')
    lex_marginals = lex_marginals_df['marginals'].values
    n = len(lex_marginals) # kinda a jank way to get n
    
    gl_stub = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_'+gl_to_string(50, 1, 1000)+'_'
    gl_marginals_df = pd.read_csv(gl_stub + 'opt_marginals.csv')
    gl_marginals = gl_marginals_df['marginals'].values
    
    lex_sums = np.cumsum(sorted(lex_marginals))
    gl_sums = np.cumsum(sorted(gl_marginals))
    eq_sums = np.cumsum([k/n]*n)
    wacky = np.cumsum([0]*(n-k)+[1]*k)
    plt.plot(lex_sums, label="leximin")
    plt.plot(gl_sums, label="goldilocks")
    plt.plot(eq_sums, label="equalized probability")
    plt.plot(wacky, label="wacky")
    plt.legend()
    plt.title(f"{instance}")
    plt.savefig(f"plots/{instance}_cdf.pdf")

def plot_probability_allocations(instance_name, algs) -> None:
    """Produce line graph comparing the probability allocations of different algorithms on the same instance.
    For each algorithm, the selection probabilities are sorted in order of increasing selection probability.
    Plot is created at ./analysis/[instance_name]_[k]_prob_allocs.pdf and a table with raw data for the plot is created
    at ./analysis/[instance_name]_[k]_prob_allocs_data.csv 
    """
    marginals = {}
    for alg in algs:
        stub = '../intermediate_data/dropped_0/'+instance_name+'/'+instance_name+'_m1000_'+alg+'_'
        marginals_df = pd.read_csv(stub + 'opt_marginals.csv')
        marginals[alg] = marginals_df['marginals'].values
    data = []
    plot_data = []
    n = len(marginals['true_goldilocks']) # kinda a jank way to get n
    k = int(instance_name[instance_name.rfind('_')+1:])
    # legacy_alloc = legacy_probabilities(instance_name, 1000, random_seed=10)
    for algorithm, alloc in marginals.items():
        # Go through agents in order of increasing selection probability
        sorted_agents = sorted(alloc)
        for i, prob in enumerate(sorted_agents):
            data.append({"algorithm": algorithm, "percentile of pool members": i / n * 100,
                         "selection probability": prob})
            if i == n - 1:  # needed to draw the last step
                plot_data.append({"algorithm": algorithm, "percentile of pool members": 100,
                                  "selection probability": prob})

    # output_directory = "plots"
    df = pd.DataFrame(data)
    df.to_csv(f"plots/{instance_name}_prob_allocs_data.csv", index=False)
    df = pd.DataFrame(data + plot_data)

    max_percentile = 100
    restricted_df = df[df['percentile of pool members'] <= max_percentile]
    fig = sns.relplot(data=restricted_df, x="percentile of pool members",
                      y='selection probability', hue="algorithm", kind="line",
                      drawstyle='steps-post', height=6, aspect=1.8, legend=False)
    fig.set_axis_labels("percentile of pool members (by selection probability)", "selection probability")

    plt.xlim(left=0.0, right=max_percentile)
    plt.ylim(bottom=0.0)
    # add legend where each line is labeled with the algorithm name
    plt.legend(title="Algorithm", labels=list(marginals.keys()), fontsize=12)
    # add title to the plot 
    plt.title(f"Probability allocations on instance {instance_name}")

    ax = fig.ax

    # equalized probability label, minimum probability line, and rectangle
    ax.text(1, 1.03 * k / n, 'equalized probability (k/n)', color='g')
    ax.axhline(y=k/(n*1.0), xmin=0, xmax=max_percentile, linestyle='--', color='black', linewidth=0.5)

    plot_path = f"plots/{instance_name}_{algs}.pdf"
    fig.savefig(plot_path)
    print("saving at plot path", plot_path)
    # return plot_path    

if __name__ == "__main__":
    # paper_manip()
    # min_max_table(['legacy', 'minimax', 'maximin', 'leximin', 'nash', 'true_goldilocks', 'gold24k'])
    # min_max_table(['minimax', 'leximin', gl_to_string(50, 1, 1), gl_to_string(50, 2, 1), gl_to_string(50, 3, 1)])
    min_max_table(['minimax', 'leximin', 'true_goldilocks', 'true_goldilocks_gamma2', 'true_goldilocks_gamma3'])
    # prob_stats_plots(['legacy', 'minimax', 'leximin', 'true_goldilocks'], instances)
    # for instance in ['sf_e_110', 'cca_75']:
    #     plot_probability_allocations(instance, ['leximin', 'minimax', gl_to_string(50, 1, 1), 'true_goldilocks', 'gold24k'])
    # paper_ssb_drops(group3)
    # paper_manip()
    # instances = ['hd_30']
    # plot_ssb_drops(['leximin', 'minimax', gl_to_string(50, 1, 1)], instances, dropped_feats=[0, 1, 2, 3, 4])
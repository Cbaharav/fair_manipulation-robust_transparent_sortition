# Given the pool composition and the population fractions, which are inferred from the quotas
# we determine the amount an agent can benefit from deviating

import sys
import os
from optimizer import Instance
import strategies
from data_extract_pool import extract_pool

import numpy as np
import pandas as pd
import random
import csv
import copy
import os
import re
import ast

seed = 20
np.random.seed(seed)
random.seed(seed)

# Type of strategy
# Tests all possible deviations for the player with all overrepresented feature values
OPTIMAL = 0
# Deviates from 0 to 1 for every player
HEURISTIC_ONE = 0
# Tests all possible deviatinos by one feature for every player
OPTIMAL_ONE_FEATURE = 0
# Tests heuristic of deviating to highest probability for every player
HEURISTIC_TWO = 0
# Tests all possible deviations for every player
OPTIMAL_ALL = 0

method = None

def run_deviations(stub, instance_name, objective, pool_copies, pool_info):
    result = {'instance': instance_name, 'objective': objective, 'pool_copies':pool_copies}
    old_n = pool_info["n"]
    k = pool_info["k"]
    F = pool_info["F"]
    people = pool_info["people"]
    categories = pool_info["categories"]
    column_data = pool_info["column_data"]
    underrepresented_fv = pool_info["underrepresented_fv"]
    people_tups = pool_info["people_tups"] #TODO: also create reverse dict mapping from tuple to people with it 
    
    # then for each tuple, find all the people with it and only check people with min probability there
    
    unique_fvs = set()
    for i in range(old_n):
        unique_fvs.add(people_tups[i])
    num_unique_vectors = len(unique_fvs)
    
    tuple_to_people = {}
    for tup in unique_fvs:
        tuple_to_people[tup] = [i for i in range(old_n) if people_tups[i] == tup]
    
    ## multiply by pool_copies
    new_n = old_n*pool_copies
    for dup in range(pool_copies-1):
        for i in range(old_n):
            people[(dup+1)*old_n+i] = copy.deepcopy(people[i])
            people_tups[(dup+1)*old_n+i] = people_tups[i]
            tuple_to_people[people_tups[i]].append((dup+1)*old_n+i)

    print("tuple_to_people: ", tuple_to_people)
     ######################################################################################################
    instance = Instance(instance_name, old_n, new_n, k, categories, people, tuple_to_people, objective, column_data, num_unique_vectors)

     # Generate marginals for original instance
    
    oldMarginals = None
    if os.path.exists(stub + "old_marginals.csv"):
        marginals_df = pd.read_csv(stub + 'old_marginals.csv')
        oldMarginals = marginals_df['marginals'].values
    else:
        oldMarginals = instance.calculateMarginals(stub+"old_")
    old_marg_avgs = []
    deviating_person = []
    
    if os.path.exists(stub+"manipvec_mappings.txt"):
        # open the file and iterate through all of the lines after the first, parsing it as manip_vec_number, person, starting_tup
        old_tuple_ordering = []
        with open(stub+"manipvec_mappings.txt", 'r') as f:
            # ignore the first line in f
            f.readline()
            for line in f:
                manip_vec_num, person, starting_tup = re.split(r',(?![^(]*\))', line)
                i = int(manip_vec_num)
                person = int(person)
                tup = ast.literal_eval(starting_tup)
                old_tuple_ordering.append(tup)
                margs_for_tup = [oldMarginals[i] for i in tuple_to_people[tup]]
                old_marg_avgs.append(sum(margs_for_tup)/float(len(margs_for_tup)))
                deviating_person.append(person)
                print("manip_vec_num: ", manip_vec_num, " person: ", person, " starting_tup: ", starting_tup)
        instance.fixed_tuples = old_tuple_ordering
    else:
        with open(stub+"manipvec_mappings.txt", 'a') as f: 
            f.write('manip_vec_num,person,starting_tup\n')
            for i, tup in enumerate(instance.fixed_tuples):
                # fixing for degenerate objs where people with same fv may get different marginals (minimax, maximin)
                margs_for_tup = [oldMarginals[i] for i in tuple_to_people[tup]]
                old_marg_avgs.append(sum(margs_for_tup)/float(len(margs_for_tup)))
                deviating_person.append(tuple_to_people[tup][0])
                f.write('%s,%s,%s\n' % (i, tuple_to_people[tup][0], tup))
        
    instance.old_marg_avgs = old_marg_avgs
    instance.deviating_people = deviating_person
    instance.underrepped_fv = underrepresented_fv
    instance.stub = stub
    instance.underrepped_people = [i for i in range(instance.new_n) if instance.people[i] == instance.underrepped_fv]
    
    best_deviation = None
    starting_vector = None
    ending_vector = None
    # if HEURISTIC_ONE == 1:
    print("Heuristic 1: underrepresented value strategy for all vectors")
    strategies.strategy_underrepresented_feature_values_all(instance)
    
    # iterate through all of the manipulation vectors, read the file, and find the best deviation
    # for i in range(num_unique_vectors):
    #     person_num = instance.deviating_people[i]
    #     with open(stub+"manipvec" + str(i)+"_person"+str(person_num)+"_deviation_benefit.txt", 'r') as f:
    #         for line in f:
    #             new_marg_avgs, old_marg_avgs, deviation_benefit = line.split(',')
    #             new_marg_avgs = float(new_marg_avgs)
    #             old_marg_avgs = float(old_marg_avgs)
    #             deviation_benefit = float(deviation_benefit)
    #             if deviation_benefit > best_deviation:
    #                 best_deviation = deviation_benefit
    #                 starting_vector = instance.fixed_tuples[i]
    #                 ending_vector = instance.underrepped_fv



    # if best_deviation is not None:
    #     print("Deviate from: ", starting_vector, " to ", ending_vector)
    #     print("Benefit: ", best_deviation)
    #     result['deviation'] = best_deviation
    #     result['method'] = method
    #     result['starting_vector'] = starting_vector
    #     result['ending_vector'] = ending_vector
    # else:
    #     print("Unable to calculate deviation")
    # return result
    

if __name__== "__main__":
    instance_name = sys.argv[1]
    objective = sys.argv[2]
    pool_copies = int(sys.argv[3]) 
    
    mydict =[]
    print("Instance: ", instance_name)
    pool_info = extract_pool(instance_name)
    # print(pool_info["fv_pop"])
    print("Objective: ", objective)
    print("Pool copies:", pool_copies)
    stub = "../CR_parallel_manip_output/vary_pool_" + instance_name +"/"+ objective+"_"+str(pool_copies) +"/"
    # if stub folder doesn't exist, create one
    if not os.path.exists(stub):
        os.makedirs(stub)
    run_deviations(stub, instance_name, objective, pool_copies, pool_info)
    # mydict.append(result)


    # # field names 
    # fields = ['instance', 'objective','starting_vector','ending_vector', 'pool_copies','method','deviation','min','gini'] 
        
    # # name of csv file 
    # filename = "../manip_output/vary_pool_" + instance_name +"/"+ objective+"_"+str(pool_copies) +".csv"
    # # writing to csv file 
    # with open(filename, 'w') as csvfile: 
    #     # creating a csv dict writer object 
    #     writer = csv.DictWriter(csvfile, fieldnames = fields) 
            
    #     # writing headers (field names) 
    #     writer.writeheader() 
            
    #     # writing data rows 
    #     writer.writerows(mydict) 
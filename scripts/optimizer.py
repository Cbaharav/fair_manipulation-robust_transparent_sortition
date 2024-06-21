import numpy as np
from typing import Dict
import gurobipy as grb
import itertools
from paper_data_analysis import save_timings, gold_rush, find_opt_distribution_leximin, find_opt_distribution_maximin, find_opt_distribution_minimax, find_opt_distribution_nash, compute_marginals, save_results, find_opt_distribution_true_gold
import os

# Instance
class Instance:
    
    # Constructor: note that pool is not yet set
    def __init__(self, instance_name, old_n, new_n, k, categories, people, tuple_to_people, obj, columns_data, num_unique_vectors, num_dropped_feats = 0):
        self.instance_name = instance_name
        self.old_n = old_n
        self.new_n = new_n
        self.k = k
        self.categories = categories
        self.people = people
        self.tuple_to_people = tuple_to_people
        self.fixed_tuples = list(tuple_to_people.keys()) # fixing the order of the tuples
        
        self.num_dropped_feats = num_dropped_feats
        self.obj = obj
        self.columns_data = columns_data
        self.num_unique_vectors = num_unique_vectors
        
        assert (list(people.keys()) == np.arange(new_n)).all() # rely on this assumption for marginals ordering

    # Set pool
    def setPeople(self, people):
        self.people = people
  
    # Calculates the marginal probabilities of each type of volunteer
    # Optimize based on an objective function (set above by flags: NASH, MAXIMIN, LEXIMIN, MINIMAX, GOLDILOCKS)
    # https://github.com/Gurobi/modeling-examples/blob/master/marketing_campaign_optimization/marketing_campaign_optimization.ipynb
    def calculateMarginals(self, filename):
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        obj = self.obj
        categories = self.categories
        people = self.people
        number_people_wanted = self.k
        check_same_address = False
        check_same_address_columns = None
        columns_data = self.columns_data
        runtime = None
        if obj =='leximin':
            committees, probabilities, output_lines, runtime = find_opt_distribution_leximin(categories, people,
                                                        columns_data, number_people_wanted, check_same_address, check_same_address_columns)
        elif obj == 'maximin':
            committees, probabilities, output_lines, infeasible, runtime = find_opt_distribution_maximin(categories, people,
                                                        columns_data, number_people_wanted, check_same_address, check_same_address_columns)
        elif obj == 'nash':
            committees, probabilities, output_lines, runtime = find_opt_distribution_nash(categories, people, columns_data, 
                                                        number_people_wanted, check_same_address, check_same_address_columns)
        elif obj == 'minimax':
            committees, probabilities, output_lines, infeasible, runtime = find_opt_distribution_minimax(categories, people,
                                                        columns_data, number_people_wanted, check_same_address, check_same_address_columns)
        elif obj == 'true_goldilocks':
            EPS_TRUE_GL = 1.0
            committees, probabilities, output_lines, runtime = find_opt_distribution_true_gold(number_people_wanted, 1, categories, people, columns_data, check_same_address, check_same_address_columns, anon=True, EPS_TRUE_GL=EPS_TRUE_GL)
        else:
            assert obj == 'goldilocks'
            committees, probabilities, output_lines, gl_timings = gold_rush(None, {"gamma_1": 1}, [1], [50], categories, people, number_people_wanted, check_same_address, save_res=False, anon=False, verbose=True)
            runtime = list(gl_timings.values())[0]
        marginals = compute_marginals(committees, probabilities, self.new_n)
        num_copies = self.new_n/self.old_n
        save_timings({self.instance_name+"_"+str(num_copies):{obj:runtime}}, "manip_timings.txt")
        save_results(committees, probabilities, filename, self.new_n)
        return marginals
        
from typing import List, FrozenSet, Dict

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from scipy.stats import beta
from scipy.stats.mstats import gmean
import mip
import cvxpy as cp
import os
import random
import pyomo.environ as pyo
from pyomo.opt import *
from pyomo.environ import *
import math
import gurobipy as grb
from time import time
import datetime
import csv

from copy import deepcopy
from random import seed
from collections import Counter
from stratification import find_random_sample_legacy, SelectionError, check_min_cats

os.system("export GUROBI_HOME=\"/Library/gurobi911/mac64\"")
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

np.random.seed(1)
random.seed(1)

# # # # # # # # # # # # # # PARAMETERS # # # # # # # # # # # # # # # # #

# number of panels desired in the lottery
M = 1000

# which instances to analyze
instances = ['hd_30', 'mass_24', 'obf_30', 'sf_a_35', 'sf_b_20', 'sf_c_44', 'sf_d_40', 'sf_e_110', 'cca_75']


# which objective you want to optimize
LEXIMIN = 0
MAXIMIN = 0
NASH = 0
MINIMAX = 0
GOLDILOCKS = 0 # right now doing to the p
LEGACY = 0
TRUE_GOLD = 1
GOLD24K = 0

# flags for which types of lotteries you want to compute
OPT = 1                      # computes unconstrained optimal distribution - need to run before any others

ILP = 0                      # computes both optimal unconstrained and near-optimal unconstrained, wrt to fairness notion specified below
BECK_FIALA = 0               # computes uniform rounded from OPT via beck-fiala (must run OPT first)
RANDOMIZED = 0               # computes uniform rounded from OPT via randomized rounding (must run OPT first) 
RANDOMIZED_REPLICATES = 1000 # runs randomized a bunch of times -> report avg and stdev of loss
ILP_MINIMIAX_CHANGE = 0      # takes input distribution specified by fairness objectives and computes minimum change in anyone's probability

RERUN = 1 # if 0 will not rerun if instance already exists

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# optimization / other run parameters
EPS = 0.0005 
EPS_NASH = 1
EPS_GL = 2
# EPS2 = 0.00000001
EPS2 = 0.00000000001

# set run parameters
check_same_address = False
check_same_address_columns = [] # unset because never used, for now
debug = 0
discrete_number = M



def _print(message: str) -> str:
    print(message)
    return message


def _setup_committee_generation(categories, people, number_people_wanted, check_same_address,households):
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"
    
    model = mip.Model(sense=mip.MAXIMIZE)
    model.verbose = debug

    # for every person, we have a binary variable indicating whether they are in the committee
    agent_vars = {id: model.add_var(var_type=mip.BINARY) for id in people}

    # we have to select exactly `number_people_wanted` many persons
    model.add_constr(mip.xsum(agent_vars.values()) == number_people_wanted)

    # we have to respect quotas
    for feature in categories:
        for value in categories[feature]:
            number_feature_value_agents = mip.xsum(agent_vars[id] for id, person in people.items()
                                                   if person[feature] == value)
            model.add_constr(number_feature_value_agents >= categories[feature][value]["min"])
            model.add_constr(number_feature_value_agents <= categories[feature][value]["max"])

    # we might not be able to select multiple persons from the same household
    if check_same_address:
        people_by_household = {}
        for id, household in households.items():
            if household not in people_by_household:
                people_by_household[household] = []
            people_by_household[household].append(id)

        for household, members in people_by_household.items():
            if len(members) >= 2:
                model.add_constr(mip.xsum(agent_vars[id] for id in members) <= 1)

    # Optimize once without any constraints to check if no feasible committees exist at all.
    status = model.optimize()
    if status == mip.OptimizationStatus.INFEASIBLE:
        print("infeasible")
        return None, None, True
        #new_quotas, output_lines = _relax_infeasible_quotas(categories, people, number_people_wanted,check_same_address, households,((),))
        raise InfeasibleQuotasError(new_quotas, output_lines)
    elif status != mip.OptimizationStatus.OPTIMAL:
        raise SelectionError(f"No feasible committees found, solver returns code {status} (see "
                             "https://docs.python-mip.com/en/latest/classes.html#optimizationstatus).")

    return model, agent_vars, False

def _generate_initial_committees(new_committee_model, agent_vars,multiplicative_weights_rounds):
    """To speed up the main iteration of the maximin and Nash algorithms, start from a diverse set of feasible
    committees. In particular, each agent that can be included in any committee will be included in at least one of
    these committees.
    """
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"
    new_output_lines = []
    committees: Set[FrozenSet[str]] = set()  # Committees discovered so far
    covered_agents: Set[str] = set()  # All agents included in some committee

    # We begin using a multiplicative-weight stage. Each agent has a weight starting at 1.
    weights = {id: 1 for id in agent_vars}
    for i in range(multiplicative_weights_rounds):
        # In each round, we find a
        # feasible committee such that the sum of weights of its members is maximal.
        new_committee_model.objective = mip.xsum(weights[id] * agent_vars[id] for id in agent_vars)
        new_committee_model.optimize()
        new_committee_model.write("cbc_fail.mps")  # TODO remove
        new_committee_model.write("cbc_fail.sol")
        new_set = _ilp_results_to_committee(agent_vars)

        # We then decrease the weight of each agent in the new committee by a constant factor. As a result, future
        # rounds will strongly prioritize including agents that appear in few committees.
        for id in new_set:
            weights[id] *= 0.8
        # We rescale the weights, which does not change the conceptual algorithm but prevents floating point problems.
        coefficient_sum = sum(weights.values())
        for id in agent_vars:
            weights[id] *= len(agent_vars) / coefficient_sum

        if new_set not in committees:
            # We found a new committee, and repeat.
            committees.add(new_set)
            for id in new_set:
                covered_agents.add(id)
        else:
            # If our committee is already known, make all weights a bit more equal again to mix things up a little.
            for id in agent_vars:
                weights[id] = 0.9 * weights[id] + 0.1

        # print(f"Multiplicative weights phase, round {i+1}/{multiplicative_weights_rounds}. Discovered {len(committees)}"
        #       " committees so far.")

    # If there are any agents that have not been included so far, try to find a committee including this specific agent.
    for id in agent_vars:
        if id not in covered_agents:
            new_committee_model.objective = agent_vars[id]  # only care about agent `id` being included.
            new_committee_model.optimize()
            new_set: FrozenSet[str] = _ilp_results_to_committee(agent_vars)
            if id in new_set:
                committees.add(new_set)
                for id2 in new_set:
                    covered_agents.add(id2)
            else:
                new_output_lines.append(_print(f"Agent {id} not contained in any feasible committee."))
                assert False # crash code if not all agents are covered
    # We assume in this stage that the quotas are feasible.
    assert len(committees) >= 1

    if len(covered_agents) == len(agent_vars):
        new_output_lines.append(_print("All agents are contained in some feasible committee."))

    return committees, frozenset(covered_agents), new_output_lines


def _ilp_results_to_committee(variables):
    try:
        res = frozenset(id for id in variables if variables[id].x > 0.5)
    except Exception as e:  # unfortunately, MIP sometimes throws generic Exceptions rather than a subclass.
        raise ValueError(f"It seems like some variables does not have a value. Original exception: {e}.")

    return res

def _dual_leximin_stage(people, committees,fixed_probabilities):
    """This implements the dual LP described in `find_distribution_leximin`, but where P only ranges over the panels
    in `committees` rather than over all feasible panels:
    minimize ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ
    s.t.     Σ_{i ∈ P} yᵢ ≤ ŷ                              ∀ P
             Σ_{i not in fixed_probabilities} yᵢ = 1
             ŷ, yᵢ ≥ 0                                     ∀ i

    Returns a Tuple[grb.Model, Dict[str, grb.Var], grb.Var]   (not in type signature to prevent global gurobi import.)
    """
    assert len(committees) != 0

    model = grb.Model()
    agent_vars = {person: model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.) for person in people}  # yᵢ
    cap_var = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.)  # ŷ
    model.addConstr(grb.quicksum(agent_vars[person] for person in people if person not in fixed_probabilities) == 1)
    for committee in committees:
        model.addConstr(grb.quicksum(agent_vars[person] for person in committee) <= cap_var)
    model.setObjective(cap_var - grb.quicksum(
                                    fixed_probabilities[person] * agent_vars[person] for person in fixed_probabilities),
                       grb.GRB.MINIMIZE)

    # Change Gurobi configuration to encourage strictly complementary (“inner”) solutions. These solutions will
    # typically allow to fix more probabilities per outer loop of the leximin algorithm.
    model.setParam("Method", 2)  # optimize via barrier only
    model.setParam("Crossover", 0)  # deactivate cross-over

    return model, agent_vars, cap_var

def find_opt_distribution_leximin(categories, people,columns_data, number_people_wanted,check_same_address, check_same_address_columns):
    """Find a distribution over feasible committees that maximizes the minimum probability of an agent being selected
    (just like maximin), but breaks ties to maximize the second-lowest probability, breaks further ties to maximize the
    third-lowest probability and so forth.

    Arguments follow the pattern of `find_random_sample`.

    Returns:
        (committees, probabilities, output_lines)
        `committees` is a list of feasible committees, where each committee is represented by a frozen set of included
            agent ids.
        `probabilities` is a list of probabilities of equal length, describing the probability with which each committee
            should be selected.
        `output_lines` is a list of debug strings.
    """
    start = time()
    output_lines = ["Using leximin algorithm."]
    grb.setParam("OutputFlag", 0)


    assert not check_same_address
    households = None

    # Set up an ILP `new_committee_model` that can be used for discovering new feasible committees maximizing some
    # sum of weights over the agents.
    new_committee_model, agent_vars, infeasible = _setup_committee_generation(categories, people, number_people_wanted,
                                                                  check_same_address, households)

    # Start by finding some initial committees, guaranteed to cover every agent that can be covered by some committee
    committees: Set[FrozenSet[str]]  # set of feasible committees, add more over time
    covered_agents: FrozenSet[str]  # all agent ids for agents that can actually be included
    committees, covered_agents, new_output_lines = _generate_initial_committees(new_committee_model, agent_vars,
                                                                                3 * len(people))
    output_lines += new_output_lines

    # Over the course of the algorithm, the selection probabilities of more and more agents get fixed to a certain value
    fixed_probabilities: Dict[str, float] = {}

    reduction_counter = 0

    # The outer loop maximizes the minimum of all unfixed probabilities while satisfying the fixed probabilities.
    # In each iteration, at least one more probability is fixed, but often more than one.
    while len(fixed_probabilities) < len(people):
        print(f"Fixed {len(fixed_probabilities)}/{len(people)} probabilities.")

        dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(people, committees, fixed_probabilities)
        # In the inner loop, there is a column generation for maximizing the minimum of all unfixed probabilities
        while True:
            """The primal LP being solved by column generation, with a variable x_P for each feasible panel P:
            
            maximize z
            s.t.     Σ_{P : i ∈ P} x_P ≥ z                         ∀ i not in fixed_probabilities
                     Σ_{P : i ∈ P} x_P ≥ fixed_probabilities[i]    ∀ i in fixed_probabilities
                     Σ_P x_P ≤ 1                                   (This should be thought of as equality, and wlog.
                                                                   optimal solutions have equality, but simplifies dual)
                     x_P ≥ 0                                       ∀ P
                     
            We instead solve its dual linear program:
            minimize ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ
            s.t.     Σ_{i ∈ P} yᵢ ≤ ŷ                              ∀ P
                     Σ_{i not in fixed_probabilities} yᵢ = 1
                     ŷ, yᵢ ≥ 0                                     ∀ i
            """
            dual_model.optimize()
            if dual_model.status != grb.GRB.OPTIMAL:
                # In theory, the LP is feasible in the first iterations, and we only add constraints (by fixing
                # probabilities) that preserve feasibility. Due to floating-point issues, however, it may happen that
                # Gurobi still cannot satisfy all the fixed probabilities in the primal (meaning that the dual will be
                # unbounded). In this case, we slightly relax the LP by slightly reducing all fixed probabilities.
                for agent in fixed_probabilities:
                    # Relax all fixed probabilities by a small constant
                    fixed_probabilities[agent] = max(0., fixed_probabilities[agent] - 0.0001)
                    dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(people, committees,
                                                                                    fixed_probabilities)
                print(dual_model.status, f"REDUCE PROBS for {reduction_counter}th time.")
                reduction_counter += 1
                continue

            # Find the panel P for which Σ_{i ∈ P} yᵢ is largest, i.e., for which Σ_{i ∈ P} yᵢ ≤ ŷ is tightest
            agent_weights = {person: agent_var.x for person, agent_var in dual_agent_vars.items()}
            new_committee_model.objective = mip.xsum(agent_weights[person] * agent_vars[person] for person in people)
            new_committee_model.optimize()
            new_set = _ilp_results_to_committee(agent_vars)  # panel P
            value = new_committee_model.objective_value  # Σ_{i ∈ P} yᵢ

            upper = dual_cap_var.x  # ŷ
            dual_obj = dual_model.objVal  # ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ

            output_lines.append(_print(f"Maximin is at most {dual_obj - upper + value:.2%}, can do {dual_obj:.2%} with "
                                       f"{len(committees)} committees. Gap {value - upper:.2%}."))
            if value <= upper + EPS:
                # Within numeric tolerance, the panels in `committees` are enough to constrain the dual, i.e., they are
                # enough to support an optimal primal solution.
                for person, agent_weight in agent_weights.items():
                    if agent_weight > EPS and person not in fixed_probabilities:
                        # `agent_weight` is the dual variable yᵢ of the constraint "Σ_{P : i ∈ P} x_P ≥ z" for
                        # i = `person` in the primal LP. If yᵢ is positive, this means that the constraint must be
                        # binding in all optimal solutions [1], and we can fix `person`'s probability to the
                        # optimal value of the primal/dual LP.
                        # [1] Theorem 3.3 in: Renato Pelessoni. Some remarks on the use of the strict complementarity in
                        # checking coherence and extending coherent probabilities. 1998.
                        fixed_probabilities[person] = max(0, dual_obj)
                break
            else:
                # Given that Σ_{i ∈ P} yᵢ > ŷ, the current solution to `dual_model` is not yet a solution to the dual.
                # Thus, add the constraint for panel P and recurse.
                assert new_set not in committees
                committees.add(new_set)
                dual_model.addConstr(grb.quicksum(dual_agent_vars[id] for id in new_set) <= dual_cap_var)

    # The previous algorithm computed the leximin selection probabilities of each agent and a set of panels such that
    # the selection probabilities can be obtained by randomizing over these panels. Here, such a randomization is found.
    primal = grb.Model()
    # Variables for the output probabilities of the different panels
    committee_vars = [primal.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.) for _ in committees]
    # To avoid numerical problems, we formally minimize the largest downward deviation from the fixed probabilities.
    eps = primal.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.)
    primal.addConstr(grb.quicksum(committee_vars) == 1)  # Probabilities add up to 1
    for person, prob in fixed_probabilities.items():
        person_probability = grb.quicksum(comm_var for committee, comm_var in zip(committees, committee_vars)
                                          if person in committee)
        primal.addConstr(person_probability >= prob - eps)
    primal.setObjective(eps, grb.GRB.MINIMIZE)
    primal.optimize()

    # Bound variables between 0 and 1 and renormalize, because np.random.choice is sensitive to small deviations here
    probabilities = np.array([comm_var.x for comm_var in committee_vars]).clip(0, 1)
    probabilities = list(probabilities / sum(probabilities))
    end = time()
    runtime = end-start
    return list(committees), probabilities, output_lines, runtime

def _find_minimax_primal(committees, covered_agents):

    model = mip.Model(sense=mip.MINIMIZE)

    committee_variables = [model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=1.) for _ in committees]
    model.add_constr(mip.xsum(committee_variables) == 1)
    
    agent_panel_variables = {id: [] for id in covered_agents} # for each agent, stores list of panels they are on
    for committee, var in zip(committees, committee_variables):
        for id in committee:
            if id in covered_agents:
                agent_panel_variables[id].append(var)

    upper = model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=1.)
    
    for agent_variables in agent_panel_variables.values():
        model.add_constr(upper >= mip.xsum(agent_variables)) # upper is at least the prob of a given agent
    
    model.objective = upper
    model.optimize()

    probabilities = [var.x for var in committee_variables]
    probabilities = [max(p, 0) for p in probabilities] # isn't this always the case given the lb on committee vars?
    sum_probabilities = sum(probabilities)
    probabilities = [p / sum_probabilities for p in probabilities] # normalizing

    return probabilities

def _find_maximin_primal(committees, covered_agents):

    model = mip.Model(sense=mip.MAXIMIZE)

    committee_variables = [model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=1.) for _ in committees]
    model.add_constr(mip.xsum(committee_variables) == 1)
    
    agent_panel_variables = {id: [] for id in covered_agents}
    for committee, var in zip(committees, committee_variables):
        for id in committee:
            if id in covered_agents:
                agent_panel_variables[id].append(var)

    lower = model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=1.)
    
    for agent_variables in agent_panel_variables.values():
        model.add_constr(lower <= mip.xsum(agent_variables))
    
    model.objective = lower
    model.optimize()

    probabilities = [var.x for var in committee_variables]
    probabilities = [max(p, 0) for p in probabilities]
    sum_probabilities = sum(probabilities)
    probabilities = [p / sum_probabilities for p in probabilities]

    return probabilities

def _find_minimax_primal_discrete(committees, covered_agents, discrete_number):
    """ finds uniform lottery that minimizes the maximum probability of any agent being selected by solving ILP.
        inputs: committees = list of committees in support of optimal unconstrained distribution
                covered_agents = list of agents included on any committee in committees (should be all agents)
                discrete_number = M, the number of panels over which you want a uniform lottery
        outputs: vector of probabilities, one assigned to each committee (in order of committees list)
    """
    model = mip.Model(sense=mip.MINIMIZE)

    committee_variables = [model.add_var(var_type=mip.INTEGER, lb=0., ub=mip.INF) for _ in committees]
    model.add_constr(mip.xsum(committee_variables) == discrete_number)
    
    agent_panel_variables = {id: [] for id in covered_agents}
    for committee, var in zip(committees, committee_variables):
        for id in committee:
            if id in covered_agents:
                agent_panel_variables[id].append(var)

    upper = model.add_var(var_type=mip.INTEGER, lb=0.)

    for agent_variables in agent_panel_variables.values():
        model.add_constr(upper >= mip.xsum(agent_variables))

    model.objective = upper
    
    model.optimize(max_seconds=1800)

    probabilities = [round(var.x) / discrete_number for var in committee_variables]


def _find_maximin_primal_discrete(committees, covered_agents, discrete_number):
    """ finds uniform lottery that maximizes the minimum probability of any agent being selected by solving ILP.
        inputs: committees = list of committees in support of optimal unconstrained distribution
                covered_agents = list of agents included on any committee in committees (should be all agents)
                discrete_number = M, the number of panels over which you want a uniform lottery
        outputs: vector of probabilities, one assigned to each committee (in order of committees list)
    """
    model = mip.Model(sense=mip.MAXIMIZE)

    committee_variables = [model.add_var(var_type=mip.INTEGER, lb=0., ub=mip.INF) for _ in committees]
    model.add_constr(mip.xsum(committee_variables) == discrete_number)
    
    agent_panel_variables = {id: [] for id in covered_agents}
    for committee, var in zip(committees, committee_variables):
        for id in committee:
            if id in covered_agents:
                agent_panel_variables[id].append(var)

    lower = model.add_var(var_type=mip.INTEGER, lb=0.)

    for agent_variables in agent_panel_variables.values():
        model.add_constr(lower <= mip.xsum(agent_variables))

    model.objective = lower
    
    model.optimize(max_seconds=1800)

    probabilities = [round(var.x) / discrete_number for var in committee_variables]


    return probabilities

def find_opt_distribution_minimax(categories, people, columns_data, number_people_wanted, check_same_address, check_same_address_columns):
    """Find a distribution over feasible committees that maximizes the minimum probability of an agent being selected.

        Arguments follow the pattern of `find_random_sample`.

        Returns:
            (committees, probabilities, output_lines)
            `committees` is a list of feasible committees, where each committee is represented by a frozen set of included
                agent ids.
            `probabilities` is a list of probabilities of equal length, describing the probability with which each committee
                should be selected.
            `output_lines` is a list of debug strings.
            boolean flag denoting infeasibility
    """
    start = time()
    output_lines = [_print("Using maximin algorithm.")]

    assert not check_same_address
    households = None

    # Set up an ILP `new_committee_model` that can be used for discovering new feasible committees maximizing some
    # sum of weights over the agents.
    new_committee_model, agent_vars, infeasible = _setup_committee_generation(categories, people, number_people_wanted,
                                                                  check_same_address, households)
    if infeasible==True:
        return None,None,None,None,True
    # Start by finding some initial committees, guaranteed to cover every agent that can be covered by some committee
    committees: Set[FrozenSet[str]]  # set of feasible committees, add more over time
    covered_agents: FrozenSet[str]  # all agent ids for agents that can actually be included
    committees, covered_agents, new_output_lines = _generate_initial_committees(new_committee_model, agent_vars,
                                                                                len(people))
    output_lines += new_output_lines

    # The incremental model is an LP with a variable y_e for each entitlement e and one more variable z.
    # For an agent i, let e(i) denote her entitlement. Then, the LP is:
    #
    # maximize  z
    # s.t.      Σ_{i ∈ B} y_{e(i)} \geq z   ∀ feasible committees B (*)
    #           Σ_e y_e = 1
    #           y_e ≥ 0                  ∀ e
    #
    # At any point in time, constraint (*) is only enforced for the committees in `committees`. By linear-programming
    # duality, if the optimal solution with these reduced constraints satisfies all possible constraints, the committees
    # in `committees` are enough to find the maximin distribution among them.
    incremental_model = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GUROBI)
    incremental_model.verbose = debug

    lower_bound = incremental_model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=mip.INF)  # variable z
    # variables y_e
    incr_agent_vars = {id: incremental_model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=1.) for id in covered_agents}

    # Σ_e y_e = 1
    incremental_model.add_constr(mip.xsum(incr_agent_vars.values()) == 1)
    # minimize z
    incremental_model.objective = lower_bound

    for committee in committees:
        committee_sum = mip.xsum([incr_agent_vars[id] for id in committee])
        # Σ_{i ∈ B} y_{e(i)} \geq z   ∀ B ∈ `committees`
        incremental_model.add_constr(committee_sum >= lower_bound)

    while True:
        status = incremental_model.optimize()
        assert status == mip.OptimizationStatus.OPTIMAL

        entitlement_weights = {id: incr_agent_vars[id].x for id in covered_agents}  # currently optimal values for y_e
        lower = lower_bound.x  # currently optimal value for z

        # For these fixed y_e, find the feasible committee B with minimal Σ_{i ∈ B} y_{e(i)} (so tightest constraint)
        new_committee_model.objective = -mip.xsum(entitlement_weights[id] * agent_vars[id] for id in covered_agents)
        new_committee_model.optimize()
        new_set = _ilp_results_to_committee(agent_vars)
        value = sum(entitlement_weights[id] for id in new_set)

        output_lines.append(_print(f"Minimax is at least {value:.2%}, can do {lower:.2%} with {len(committees)} "
                                   f"committees. Gap {lower - value:.2%}{'≤' if lower-value <= EPS else '>'}{EPS:%}."))
        if value >= lower - EPS:
            # No feasible committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z (at least up to EPS, to prevent rounding errors).
            # Thus, we have enough committees.
            committee_list = list(committees)
            probabilities = _find_minimax_primal(committee_list, covered_agents)
           
            equality_thresh = float(number_people_wanted) / len(people) # k/n
            if value >= 2*equality_thresh:
                output_lines.append(_print("Minimax is also infinity norm."))
            else:
                output_lines.append(_print("Minimax is not infinity norm."))
            end = time()
            runtime = end-start
            return committee_list, probabilities, output_lines, False, runtime
        
        else:
            # Some committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z. We add B to `committees` and recurse.
            assert new_set not in committees
            committees.add(new_set)
            incremental_model.add_constr(mip.xsum(incr_agent_vars[id] for id in new_set) >= lower_bound)

            # Heuristic for better speed in practice:
            # Because optimizing `incremental_model` takes a long time, we would like to get multiple committees out
            # of a single run of `incremental_model`. Rather than reoptimizing for optimal y_e and z, we find some
            # feasible values y_e and z by modifying the old solution.
            # This heuristic only adds more committees, and does not influence correctness.
            
            # Ok I really don't get this heuristic at the moment
            counter = 0
            for _ in range(10):
                # scale down the y_{e(i)} for i ∈ `new_set` to make Σ_{i ∈ `new_set`} y_{e(i)} ≤ z true.
                for id in new_set:
                    entitlement_weights[id] *= value / lower
                # This will change Σ_e y_e to be less than 1. We rescale the y_e and z.
                sum_weights = sum(entitlement_weights.values())
                if sum_weights < EPS:
                    break
                for id in entitlement_weights:
                    entitlement_weights[id] *= sum_weights
                lower *= sum_weights

                new_committee_model.objective = mip.xsum(entitlement_weights[id] * agent_vars[id]
                                                         for id in covered_agents)
                new_committee_model.optimize()
                new_set = _ilp_results_to_committee(agent_vars)
                value = sum(entitlement_weights[id] for id in new_set)
                if value >= lower - EPS or new_set in committees:
                    break
                else:
                    committees.add(new_set)
                    incremental_model.add_constr(mip.xsum(incr_agent_vars[id] for id in new_set) >= lower_bound)
                counter += 1
            if counter > 0:
                print(f"Heuristic successfully generated {counter} additional committees.")
                

def find_opt_distribution_maximin(categories, people, columns_data, number_people_wanted, check_same_address, check_same_address_columns):
    """Find a distribution over feasible committees that maximizes the minimum probability of an agent being selected.

        Arguments follow the pattern of `find_random_sample`.

        Returns:
            (committees, probabilities, output_lines)
            `committees` is a list of feasible committees, where each committee is represented by a frozen set of included
                agent ids.
            `probabilities` is a list of probabilities of equal length, describing the probability with which each committee
                should be selected.
            `output_lines` is a list of debug strings.
            boolean flag denoting infeasibility
    """
    start = time()
    output_lines = [_print("Using maximin algorithm.")]

    assert not check_same_address
    households = None

    # Set up an ILP `new_committee_model` that can be used for discovering new feasible committees maximizing some
    # sum of weights over the agents.
    new_committee_model, agent_vars, infeasible = _setup_committee_generation(categories, people, number_people_wanted,
                                                                  check_same_address, households)
    if infeasible==True:
        return None,None,None,None,True
    # Start by finding some initial committees, guaranteed to cover every agent that can be covered by some committee
    committees: Set[FrozenSet[str]]  # set of feasible committees, add more over time
    covered_agents: FrozenSet[str]  # all agent ids for agents that can actually be included
    committees, covered_agents, new_output_lines = _generate_initial_committees(new_committee_model, agent_vars,
                                                                                len(people))
    output_lines += new_output_lines

    # The incremental model is an LP with a variable y_e for each entitlement e and one more variable z.
    # For an agent i, let e(i) denote her entitlement. Then, the LP is:
    #
    # minimize  z
    # s.t.      Σ_{i ∈ B} y_{e(i)} ≤ z   ∀ feasible committees B (*)
    #           Σ_e y_e = 1
    #           y_e ≥ 0                  ∀ e
    #
    # At any point in time, constraint (*) is only enforced for the committees in `committees`. By linear-programming
    # duality, if the optimal solution with these reduced constraints satisfies all possible constraints, the committees
    # in `committees` are enough to find the maximin distribution among them.
    incremental_model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.GUROBI)
    incremental_model.verbose = debug

    upper_bound = incremental_model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=mip.INF)  # variable z
    # variables y_e
    incr_agent_vars = {id: incremental_model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=1.) for id in covered_agents}

    # Σ_e y_e = 1
    incremental_model.add_constr(mip.xsum(incr_agent_vars.values()) == 1)
    # minimize z
    incremental_model.objective = upper_bound

    for committee in committees:
        committee_sum = mip.xsum([incr_agent_vars[id] for id in committee])
        # Σ_{i ∈ B} y_{e(i)} ≤ z   ∀ B ∈ `committees`
        incremental_model.add_constr(committee_sum <= upper_bound)

    while True:
        status = incremental_model.optimize()
        assert status == mip.OptimizationStatus.OPTIMAL

        entitlement_weights = {id: incr_agent_vars[id].x for id in covered_agents}  # currently optimal values for y_e
        upper = upper_bound.x  # currently optimal value for z

        # For these fixed y_e, find the feasible committee B with maximal Σ_{i ∈ B} y_{e(i)}.
        new_committee_model.objective = mip.xsum(entitlement_weights[id] * agent_vars[id] for id in covered_agents)
        new_committee_model.optimize()
        new_set = _ilp_results_to_committee(agent_vars)
        value = sum(entitlement_weights[id] for id in new_set)

        output_lines.append(_print(f"Maximin is at most {value:.2%}, can do {upper:.2%} with {len(committees)} "
                                   f"committees. Gap {value - upper:.2%}{'≤' if value-upper <= EPS else '>'}{EPS:%}."))
        if value <= upper + EPS:
            # No feasible committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z (at least up to EPS, to prevent rounding errors).
            # Thus, we have enough committees.
            committee_list = list(committees)
            probabilities = _find_maximin_primal(committee_list, covered_agents)
            end = time()
            runtime = end-start
            return committee_list, probabilities, output_lines, False, runtime
        
        else:
            # Some committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z. We add B to `committees` and recurse.
            assert new_set not in committees
            committees.add(new_set)
            incremental_model.add_constr(mip.xsum(incr_agent_vars[id] for id in new_set) <= upper_bound)

            # Heuristic for better speed in practice:
            # Because optimizing `incremental_model` takes a long time, we would like to get multiple committees out
            # of a single run of `incremental_model`. Rather than reoptimizing for optimal y_e and z, we find some
            # feasible values y_e and z by modifying the old solution.
            # This heuristic only adds more committees, and does not influence correctness.
            counter = 0
            for _ in range(10):
                # scale down the y_{e(i)} for i ∈ `new_set` to make Σ_{i ∈ `new_set`} y_{e(i)} ≤ z true.
                for id in new_set:
                    entitlement_weights[id] *= upper / value
                # This will change Σ_e y_e to be less than 1. We rescale the y_e and z.
                sum_weights = sum(entitlement_weights.values())
                if sum_weights < EPS:
                    break
                for id in entitlement_weights:
                    entitlement_weights[id] /= sum_weights
                upper /= sum_weights

                new_committee_model.objective = mip.xsum(entitlement_weights[id] * agent_vars[id]
                                                         for id in covered_agents)
                new_committee_model.optimize()
                new_set = _ilp_results_to_committee(agent_vars)
                value = sum(entitlement_weights[id] for id in new_set)
                if value <= upper + EPS or new_set in committees:
                    break
                else:
                    committees.add(new_set)
                    incremental_model.add_constr(mip.xsum(incr_agent_vars[id] for id in new_set) <= upper_bound)
                counter += 1
            if counter > 0:
                print(f"Heuristic successfully generated {counter} additional committees.")

def legacy_find(feature_info, agents, k):
    """Produce a single panel using the LEGACY algorithm, restarting until a panel is found."""
    columns_data = {agent_id: {} for agent_id in agents}
    for feature in feature_info:
        for value in feature_info[feature]:
            fv_agents = [id for id, person in people.items() if person[feature] == value]
            num_fv_agents = len(fv_agents)
            feature_info[feature][value]["selected"] = 0
            feature_info[feature][value]["remaining"] = num_fv_agents
    # print(feature_info)
    # print(agents)        
    while True:
        # print("in legacy find loop")
        cat_copy = deepcopy(feature_info)
        agents_copy = deepcopy(agents)
        try:
            people_selected, new_output_lines = find_random_sample_legacy(cat_copy, agents_copy, columns_data, k, False,
                                                                          [])
        except SelectionError:
            continue
        # check we have reached minimum needed in all cats
        check_min_cat, new_output_lines = check_min_cats(cat_copy)
        if check_min_cat:
            return list(people_selected.keys())

def legacy_probabilities(random_seed, iterations, categories, people, columns_data, number_people_wanted, check_same_address, check_same_address_columns):
    """Estimate the probability allocation for LEGACY by drawing `legacy` many random panels in a row."""
    print("running legacy probabilities")
    seed(random_seed)
    np.random.seed(random_seed)

    for category in categories:
        assert sum(fv_info["min"] for fv_info in categories[category].values()) <= number_people_wanted
        assert sum(fv_info["max"] for fv_info in categories[category].values()) >= number_people_wanted

    # For each agent, count how frequently she was selected among `iterations` many panels produced by LEGACY
    agent_appearance_counter = Counter()
    for i in range(iterations):
        if i % 100 == 0:
            print(f"Running iteration {i} out of {iterations}.")
        agent_appearance_counter.update(legacy_find(categories, people, number_people_wanted))
    return {agent_id: agent_appearance_counter[agent_id] / iterations for agent_id in people}


def Objrule(model):
  return pyo.quicksum(log(model.marginals[i]) for i in model.marginals)



def _define_entitlements(covered_agents):
    entitlements = list(covered_agents)
    contributes_to_entitlement = {}
    for id in covered_agents:
        contributes_to_entitlement[id] = entitlements.index(id)

    return entitlements, contributes_to_entitlement


def _committees_to_matrix(committees, entitlements,
                          contributes_to_entitlement):
    columns = []
    for committee in committees:
        column = [0 for _ in entitlements]
        for id in committee:
            column[contributes_to_entitlement[id]] += 1
        columns.append(np.array(column))
    return np.column_stack(columns)



def find_rounded_distribution_nash(committees, covered_agents, discrete_number):
    """ finds uniform lottery that maximizes the geometric mean of agents' marginals. does so via Baron solver, implemented with pyomo.
        inputs: committees = list of committees in support of optimal unconstrained distribution
                covered_agents = list of agents included on any committee in committees (should be all agents)
                discrete_number = M, the number of panels over which you want a uniform lottery
        outputs: vector of probabilities, one assigned to each committee (in order of committees list)
    """

    n_committees = len(committees)
    n_agents = len(list(covered_agents))

    # define the model - variables for committee probabilities (multiplied by m, the discrete number) and for individuals' marginals
    model = pyo.ConcreteModel()
    model.probs = pyo.Var(range(n_committees), domain=pyo.NonNegativeIntegers)
    model.marginals = pyo.Var(range(n_agents), domain=pyo.NonNegativeReals)

    # must be a valid distribution over m panels
    model.committeedist_constr = pyo.Constraint(rule=(pyo.summation(model.probs)==discrete_number))

    # agents' marginals must equal sum of probs of committees theyre on
    model.marginals_constrs = pyo.ConstraintList()
    for agent in list(covered_agents):
        expr = 0
        for i in range(n_committees):
            committee = list(committees)[i]
            if agent in committee:
                expr += model.probs[i]

        model.marginals_constrs.add(model.marginals[agent]==expr)

    # objective is product of marginals
    model.obj = pyo.Objective(rule=Objrule,sense=pyo.maximize)


    # objective: product of individual probabilities
    opt = SolverFactory('baron',executable='/usr/local/bin/baron-osx64/baron')
    results = opt.solve(model)

    results.write()

    probabilities_rounded = []
    for i in model.probs:
        probabilities_rounded.append(model.probs[i].value/discrete_number)

    return probabilities_rounded

# alternate function, which finds nash-optimal uniform lottery via ILP using gurobi solver
def _find_nash_primal_discrete_gurobi(committees, covered_agents, discrete_number):
    """ finds uniform lottery that maximizes the geometric mean of agents' marginals. does so via Gurobi solver.
        inputs: committees = list of committees in support of optimal unconstrained distribution
                covered_agents = list of agents included on any committee in committees (should be all agents)
                discrete_number = M, the number of panels over which you want a uniform lottery
        outputs: vector of probabilities, one assigned to each committee (in order of committees list)
    """
    model = grb.Model()

    committee_variables = [model.addVar(vtype=grb.GRB.INTEGER, lb=0.) for _ in committees]
    model.addConstr(grb.quicksum(committee_variables) == discrete_number)
    
    agent_panel_variables = {id: [] for id in covered_agents}
    for committee, var in zip(committees, committee_variables):
        for id in committee:
            if id in covered_agents:
                agent_panel_variables[id].append(var)
                
    agent_utils = {id: model.addVar(vtype=grb.GRB.INTEGER, lb=0., name=f"u_{id}") for id in covered_agents}
    agent_log_utils = {id: model.addVar(vtype=grb.GRB.CONTINUOUS, name=f"log_u_{id}") for id in covered_agents}
    for id in covered_agents:
        model.addConstr(agent_utils[id] == grb.quicksum(agent_panel_variables[id]))
        model.addGenConstrLog(agent_utils[id], agent_log_utils[id], options="FuncPieces=-1 FuncPieceError=0.0001")

    model.setObjective(grb.quicksum(agent_log_utils.values()), grb.GRB.MAXIMIZE)
    model.write("test.lp")
    
    model.setParam('MIPGap', 0.0005)
    model.setParam('TimeLimit', 7200)
    model.optimize()

    probabilities = [round(var.x) / discrete_number for var in committee_variables]

    return probabilities

def _find_goldilocks_primal_discrete_gurobi(n, k, committees, covered_agents, discrete_number, p=50, gamma=1):
    """ finds uniform lottery that maximizes goldilocks objective.
        inputs: committees = list of committees in support of optimal unconstrained distribution
                covered_agents = list of agents included on any committee in committees (should be all agents)
                discrete_number = M, the number of panels over which you want a uniform lottery
        outputs: vector of probabilities, one assigned to each committee (in order of committees list)
    """
    model = grb.Model()

    committee_variables = [model.addVar(vtype=grb.GRB.INTEGER, lb=0.) for _ in committees]
    model.addConstr(grb.quicksum(committee_variables) == discrete_number)
    
    agent_panel_variables = {id: [] for id in covered_agents}
    for committee, var in zip(committees, committee_variables):
        for id in committee:
            if id in covered_agents:
                agent_panel_variables[id].append(var)
                
    agent_utils = {id: model.addVar(vtype=grb.GRB.INTEGER, lb=0., name=f"u_{id}") for id in covered_agents} # not dividing by num panels because would cancel out
    agent_reciprocal_utils = {id: model.addVar(vtype=grb.GRB.CONTINUOUS, name=f"recip_u_{id}") for id in covered_agents}
    for id in covered_agents:
        model.addConstr(agent_utils[id] == grb.quicksum(agent_panel_variables[id]))
        model.addGenConstrPow(agent_utils[id], agent_reciprocal_utils[id], -1, options="FuncPieces=-1 FuncPieceError=0.0001")
    
    # create variable for the p norm of the agent utilities
    agent_norm = model.addVar(vtype=grb.GRB.CONTINUOUS, name="agent_norm")
    recip_norm = model.addVar(vtype=grb.GRB.CONTINUOUS, name="recip_norm")
    model.addGenConstrNorm(agent_norm, agent_utils.values(), grb.GRB.INFINITY, name="max_norm_constr")  
    model.addGenConstrNorm(recip_norm, agent_reciprocal_utils.values(), grb.GRB.INFINITY, name="min_norm_constr")
    balance_factor = float(n)/(k*discrete_number) # putting m in bottom of fraction to account
    
    # set objective to minimize balance_factor * agent_norm + (gamma/balance_factor) * recip_norm
    model.setObjective(balance_factor * agent_norm + (gamma/balance_factor) * recip_norm, grb.GRB.MINIMIZE)
    model.write("test.lp")
    
    model.setParam('MIPGap', 0.0005)
    model.setParam('TimeLimit', 21000)
    model.optimize()

    probabilities = [round(var.x) / discrete_number for var in committee_variables]

    return probabilities

def find_opt_distribution_nash(categories, people, columns_data, number_people_wanted, check_same_address, check_same_address_columns):
    """Find a distribution over feasible committees that maximizes the so-called Nash welfare, i.e., the product of
    selection probabilities over all persons.

    Arguments follow the pattern of `find_random_sample`.

    Returns:
        (committees, probabilities, output_lines)
        `committees` is a list of feasible committees, where each committee is represented by a frozen set of included
            agent ids.
        `probabilities` is a list of probabilities of equal length, describing the probability with which each committee
            should be selected.
        `output_lines` is a list of debug strings.

    The following gives more details about the algorithm:
    Instead of directly maximizing the product of selection probabilities Πᵢ pᵢ, we equivalently maximize
    log(Πᵢ pᵢ) = Σᵢ log(pᵢ). If some person/household i is not included in any feasible committee, their pᵢ is 0, and
    this sum is -∞. We will then try to maximize Σᵢ log(pᵢ) where i is restricted to range over persons/households that
    can possibly be included.
    """
    start = time()
    output_lines = ["Using Nash algorithm."]

    assert not check_same_address
    households = None

    # `new_committee_model` is an integer linear program (ILP) used for discovering new feasible committees.
    # We will use it many times, putting different weights on the inclusion of different agents to find many feasible
    # committees.
    new_committee_model, agent_vars, infeasible = _setup_committee_generation(categories, people, number_people_wanted, check_same_address, households)

    # Start by finding committees including every agent, and learn which agents cannot possibly be included.
    committees: List[FrozenSet[str]]  # set of feasible committees, add more over time
    covered_agents: FrozenSet[str]  # all agent ids for agents that can actually be included
    committee_set, covered_agents, new_output_lines = _generate_initial_committees(new_committee_model, agent_vars,
                                                                                   2 * len(people))
    committees = list(committee_set)
    output_lines += new_output_lines

    # Map the covered agents to indices in a list for easier matrix representation.
    entitlements: List[str]
    contributes_to_entitlement: Dict[str, int]  # for id of a covered agent, the corresponding index in `entitlements`
    entitlements, contributes_to_entitlement = _define_entitlements(covered_agents)

    # Now, the algorithm proceeds iteratively. First, it finds probabilities for the committees already present in
    # `committees` that maximize the sum of logarithms. Then, reusing the old ILP, it finds the feasible committee
    # (possibly outside of `committees`) such that the partial derivative of the sum of logarithms with respect to the
    # probability of outputting this committee is maximal. If this partial derivative is less than the maximal partial
    # derivative of any committee already in `committees`, the Karush-Kuhn-Tucker conditions (which are sufficient in
    # this case) imply that the distribution is optimal even with all other committees receiving probability 0.
    start_lambdas = [1 / len(committees) for _ in committees]
    while True:
        lambdas = cp.Variable(len(committees))  # probability of outputting a specific committee
        lambdas.value = start_lambdas
        # A is a binary matrix, whose (i,j)th entry indicates whether agent `feasible_agents[i]`
        matrix = _committees_to_matrix(committees, entitlements, contributes_to_entitlement)
        assert matrix.shape == (len(entitlements), len(committees))

        objective = cp.Maximize(cp.sum(cp.log(matrix @ lambdas)))
        constraints = [0 <= lambdas, sum(lambdas) == 1]
        problem = cp.Problem(objective, constraints)
        # TODO: test relative performance of both solvers, see whether warm_start helps.
        try:
            nash_welfare = problem.solve(solver=cp.SCS, warm_start=True)
        except cp.SolverError:
            # At least the ECOS solver in cvxpy crashes sometimes (numerical instabilities?). In this case, try another
            # solver. But hope that SCS is more stable.
            output_lines.append(_print("Had to switch to ECOS solver."))
            nash_welfare = problem.solve(solver=cp.ECOS, warm_start=True)
        scaled_welfare = nash_welfare - len(entitlements) * log(number_people_wanted / len(entitlements))
        output_lines.append(_print(f"Scaled Nash welfare is now: {scaled_welfare}."))

        assert lambdas.value.shape == (len(committees),)
        entitled_utilities = matrix.dot(lambdas.value)
        assert entitled_utilities.shape == (len(entitlements),)
        assert (entitled_utilities > EPS2).all()
        entitled_reciprocals = 1 / entitled_utilities
        assert entitled_reciprocals.shape == (len(entitlements),)
        differentials = entitled_reciprocals.dot(matrix)
        assert differentials.shape == (len(committees),)

        obj = []
        for id in covered_agents:
            obj.append(entitled_reciprocals[contributes_to_entitlement[id]] * agent_vars[id])
        new_committee_model.objective = mip.xsum(obj)
        new_committee_model.optimize()

        new_set = _ilp_results_to_committee(agent_vars)
        value = sum(entitled_reciprocals[contributes_to_entitlement[id]] for id in new_set)
        if value <= differentials.max() + EPS_NASH:
            probabilities = np.array(lambdas.value).clip(0, 1)
            probabilities = list(probabilities / sum(probabilities))
            end = time()
            runtime = end-start
            return committees, probabilities, output_lines, runtime
        else:
            print(value, differentials.max(), value - differentials.max())
            assert new_set not in committees
            committees.append(new_set)
            start_lambdas = np.array(lambdas.value).resize(len(committees))
            
def compute_gammas(n, k, categories, people, instance, num_dropped_feats):
    gamma_1 = 1
    
    stub_maximin = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_maximin_'
    stub_minimax = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_minimax_'
    marginals_df_maximin = pd.read_csv(stub_maximin + 'opt_marginals.csv')
    marginals_maximin = marginals_df_maximin['marginals'].values
    marginals_df_minimax = pd.read_csv(stub_minimax + 'opt_marginals.csv')
    marginals_minimax = marginals_df_minimax['marginals'].values
    gamma_2 = (n**2 / k**2) * (np.min(marginals_maximin) * np.max(marginals_minimax))
    
    ratios = []
    for feature in categories:
        for value in categories[feature]:
            fv_agents = [id for id, person in people.items() if person[feature] == value]
            frac_fv_agents = len(fv_agents)/n
            p_fv = (categories[feature][value]['min'] + categories[feature][value]['max']) / (2.0*k) # doing avg of min and max bc don't want 0s or ns
            print(f"feature = {feature}, value = {value}, p_fv = {p_fv}, frac_fv_agents = {frac_fv_agents}")
            if frac_fv_agents > 0 and p_fv > 0: ratios += [p_fv / frac_fv_agents]
    gamma_3 = np.min(ratios) * np.max(ratios)

    gammas = {"gamma_1": gamma_1, "gamma_2": gamma_2, "gamma_3": gamma_3}
    with open('../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/gammas.txt', 'w') as f: 
        for key, value in gammas.items():
            f.write('%s:%s\n' % (key, value))
    return gammas
    
def gold_rush(stub, gammas, gamma_scales, ps, categories, people, k, check_same_address, save_res = True, anon=True, verbose=True):
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"
    output_lines = []
    assert not check_same_address
    households = None
    timings = {}

    # `new_committee_model` is an integer linear program (ILP) used for discovering new feasible committees.
    # We will use it many times, putting different weights on the inclusion of different agents to find many feasible
    # committees.
    new_committee_model, agent_vars, infeasible = _setup_committee_generation(categories, people, k, check_same_address, households)

    # Start by finding committees including every agent, and learn which agents cannot possibly be included.
    committees: List[FrozenSet[str]]  # set of feasible committees, add more over time
    covered_agents: FrozenSet[str]  # all agent ids for agents that can actually be included
    committee_set, covered_agents, new_output_lines = _generate_initial_committees(new_committee_model, agent_vars,
                                                                                   2 * len(people))
    committees = list(committee_set)
    probabilities = None
    output_lines += new_output_lines
    for gamma_scale in gamma_scales:
        for gamma_label, gamma_value in gammas.items():
            for p in ps:
                # passing in the precomputed committees each time lol
                committees, probabilities, new_output_lines, runtime = find_opt_distribution_gold(k, gamma_scale * gamma_value, p, people, new_committee_model, agent_vars, committees, covered_agents, anon=anon, verbose=verbose)
                timings[f"GL:p={p}_{gamma_label}_scale={gamma_scale}"] = runtime
                output_lines += new_output_lines
                if save_res: 
                    save_results(committees, probabilities, stub + f'p={p}_{gamma_label}_scale={gamma_scale}_opt_',n)

    return committees, probabilities, output_lines, timings # return the last set of committees and probabilities so that if only doing one run it's normal           

def save_timings(timings, filename):
    #write timings to file:
    with open(filename, 'a') as f: 
        for instance, obj_dicts in timings.items():
            for obj, timing in obj_dicts.items():
                current_day = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                f.write('%s,%s,%s,%s\n' % (instance, obj, timing, current_day))
                
def fv_vecs(people, entitlements, contributes_to_entitlement):
    # fix an ordering of the features
    feat_list = list(people[entitlements[0]].keys())
    fv_vec_to_agents = {}
    for id in people:
        fv_vec = []
        for feat in feat_list:
            fv_vec.append(people[id][feat])
        fv_vec = tuple(fv_vec)
        if fv_vec not in fv_vec_to_agents:
            fv_vec_to_agents[fv_vec] = [contributes_to_entitlement[id]]
        else:
            fv_vec_to_agents[fv_vec].append(contributes_to_entitlement[id])
    return fv_vec_to_agents  

def find_opt_distribution_gold24k(k, gamma, categories, people, columns_data, check_same_address, check_same_address_columns, verbose = True, anon = True, EPS2=0.00000000001, EPS_TRUE_GL=0.0001):
    # do true max and min using SOCP
    output_lines = []
    assert not check_same_address
    households = None
    start = time()

    # `new_committee_model` is an integer linear program (ILP) used for discovering new feasible committees.
    # We will use it many times, putting different weights on the inclusion of different agents to find many feasible
    # committees.
    new_committee_model, agent_vars, infeasible = _setup_committee_generation(categories, people, k, check_same_address, households)

    # Start by finding committees including every agent, and learn which agents cannot possibly be included.
    committees: List[FrozenSet[str]]  # set of feasible committees, add more over time
    covered_agents: FrozenSet[str]  # all agent ids for agents that can actually be included
    committee_set, covered_agents, new_output_lines = _generate_initial_committees(new_committee_model, agent_vars,
                                                                                2 * len(people))
    committees = list(committee_set)
    
    output_lines += new_output_lines
    output_lines += [f"Using 24k goldilocks with gamma={gamma}."]
    entitlements: List[str]
    contributes_to_entitlement: Dict[str, int]  # for id of a covered agent, the corresponding index in `entitlements`
    entitlements, contributes_to_entitlement = _define_entitlements(covered_agents)
    fv_vec_to_agents = fv_vecs(people, entitlements, contributes_to_entitlement)
    
    start_lambdas = [1 / len(committees) for _ in committees]
    iter_num = 0
    while True:
        iter_num += 1
        lambdas = cp.Variable(len(committees))  # probability of outputting a specific committee
        x = cp.Variable(1) # max term  
        y = cp.Variable(1) # gamma/min
        z = cp.Variable(1) # - max(x/(k/n), (k/n)/y)
        pis = cp.Variable(len(entitlements))
        lambdas.value = start_lambdas
        # A is a binary matrix, whose (i,j)th entry indicates whether agent `feasible_agents[i]`
        matrix = _committees_to_matrix(committees, entitlements, contributes_to_entitlement)
        assert matrix.shape == (len(entitlements), len(committees))

        balance_factor = len(people)/float(k)
        objective = cp.Maximize(z)
        constraints = [z <= 0, 0 <= lambdas, sum(lambdas) == 1, matrix @ lambdas == pis]
        num_pre_aux_constraints = len(constraints)
        constraints.append(balance_factor * x + z <= 0)
        constraints.append(y/balance_factor + z <= 0)
        
        y_constr_inds = []
        x_constr_inds = []
        for ent in range(len(entitlements)):
            y_constr_inds.append(len(constraints))
            constraints.append(cp.SOC(pis[ent] + y, cp.vstack([2*math.sqrt(gamma), pis[ent] - y])))
            x_constr_inds.append(len(constraints))
            constraints.append(pis[ent] <= x)

        num_pre_anon_constraints = len(constraints)
        # copy constraints list
        constraints_copy = constraints.copy()
        if anon:
            # add constraints so that all vectors with the same feature values have the same probability
            constraint_inds = [] # stores the tuples of indices of agents that we're constraining
            for fv_vec, agents in fv_vec_to_agents.items():
                if len(agents) == 1: continue
                for i in range(1, len(agents)):
                    constraints.append(pis[agents[0]] - pis[agents[i]] <= 0.01)
                    constraints.append(pis[agents[i]] - pis[agents[0]] <= 0.01)
                    constraint_inds.append((agents[0], agents[i]))
                    constraint_inds.append((agents[i], agents[0]))
        
        problem = cp.Problem(objective, constraints)
        gl_obj = problem.solve(solver=cp.ECOS, warm_start=False, verbose=False)
        # if the problem is infeasible, run it again but change the constraints to just be the first two
        curr_anon = anon
        if problem.status == "infeasible" and anon:
            # sometimes not feasible to have anonymity with start set
            print("infeasible problem, running again without anonymity")
            constraints = constraints_copy
            problem = cp.Problem(objective, constraints)
            gl_obj = problem.solve(solver=cp.ECOS, warm_start=True, verbose=False)
            curr_anon = False
            
        if verbose:
            output_lines.append(_print(f"Current 24k GL objective is now: {-gl_obj}."))

        assert lambdas.value.shape == (len(committees),)

        entitled_utilities = pis.value
        assert entitled_utilities.shape == (len(entitlements),)
        assert (entitled_utilities > EPS2).all()
        dhdp = np.zeros(len(entitlements))
        sum_dgdp_terms = np.zeros(len(entitlements))
        
        max_agents = [] # list of agents with max pi
        min_agents = [] # list of agents with min pi
        w = 0
        q = 0
        A_constr_agents = [0 for _ in range(len(entitlements))]
        B_constr_agents = [0 for _ in range(len(entitlements))]
        for agent in range(len(entitlements)):
            pi = entitled_utilities[agent]
            is_max_pi = int(pi == entitled_utilities.max())
            is_min_pi = int(pi == entitled_utilities.min())
            if is_max_pi: max_agents.append(agent)
            if is_min_pi: min_agents.append(agent)
            
            dxdp = is_max_pi
            dydp = (-gamma/(pi**2)) * is_min_pi
            x_is_max = (balance_factor * x.value >= y.value/balance_factor)
            y_is_max = (y.value/balance_factor >= balance_factor * x.value) # not doing 1-x_is_max bc they could be equal
            dzdp = -balance_factor * int(x_is_max) * dxdp - (1.0/balance_factor) * int(y_is_max) * dydp
            # print("x is max", x_is_max)
            # print("y is max", y_is_max)
            if is_max_pi: print("MAX agent", agent, "pi", pi, "dxdp", dxdp, "dydp", dydp, "dzdp", dzdp)
            if is_min_pi: print("MIN agent", agent, "pi", pi, "dxdp", dxdp, "dydp", dydp, "dzdp", dzdp)
            dhdp[agent] = dzdp
            
            # z constraints
            mu_z_x_constraint = constraints[num_pre_aux_constraints].dual_value
            mu_z_y_constraint = constraints[num_pre_aux_constraints + 1].dual_value
            sum_dgdp_terms[agent] += mu_z_x_constraint * (balance_factor * dxdp + dzdp)
            sum_dgdp_terms[agent] += mu_z_y_constraint * (dydp/balance_factor + dzdp)
            if is_max_pi:
                print("z x constraint", mu_z_x_constraint * (balance_factor * dxdp + dzdp))
                print("z y constraint", mu_z_y_constraint * (dydp/balance_factor + dzdp))
            
            # x & y constraints            
            mu_x_constraint = constraints[x_constr_inds[agent]].dual_value
            mu_y_constraint = constraints[y_constr_inds[agent]].dual_value[0]
            
            q += mu_x_constraint
            w -= mu_y_constraint * (((4 * gamma) + (pi - y.value)**2)**(-0.5) * (pi - y.value) + 1)
            
            sum_dgdp_terms[agent] += mu_x_constraint
            B_constr_agents[agent] += mu_x_constraint
            sum_dgdp_terms[agent] += dydp * mu_y_constraint * (((4 * gamma) + (pi - y.value)**2)**(-0.5) * (pi - y.value) + 1)
            sum_dgdp_terms[agent] += mu_y_constraint * (((4 * gamma) + (pi - y.value)**2)**(-0.5) - 1)
            A_constr_agents[agent]+= dydp * mu_y_constraint * (((4 * gamma) + (pi - y.value)**2)**(-0.5) * (pi - y.value) + 1) + mu_y_constraint * (((4 * gamma) + (pi - y.value)**2)**(-0.5) - 1)
        
        # correcting max and min dgdp
        for agent in max_agents:
            sum_dgdp_terms[agent] -= q
            B_constr_agents[agent] -= q
        for agent in min_agents:
            sum_dgdp_terms[agent] += dydp * w
            A_constr_agents[agent] += dydp * w
        
        if curr_anon:
            for ind, (i, j) in enumerate(constraint_inds):
                constr_dual_val = constraints[ind + num_pre_anon_constraints].dual_value
                sum_dgdp_terms[i] += constr_dual_val
                sum_dgdp_terms[j] -= constr_dual_val
        etas = dhdp - sum_dgdp_terms
        # dhdp_dict = {i: dhdp[i] for i in range(len(entitlements))}
        # A_dict = {i: A_constr_agents[i] for i in range(len(entitlements))}
        # B_dict = {i: B_constr_agents[i] for i in range(len(entitlements))}
        # sum_dgdp_terms_dict = {i: sum_dgdp_terms[i] for i in range(len(entitlements))}
        # eta_dict = {i: etas[i] for i in range(len(entitlements))}
        # print('dhdp_dict', dhdp_dict)
        # print('A dict', A_dict)
        # print('B dict', B_dict)
        
        differentials = etas.dot(matrix) # sum for each committee
        assert differentials.shape == (len(committees),)

        obj = []
        for id in covered_agents:
            obj.append(etas[contributes_to_entitlement[id]] * agent_vars[id])
        new_committee_model.objective = mip.xsum(obj)
        new_committee_model.optimize()

        new_set = _ilp_results_to_committee(agent_vars)
        value = sum(etas[contributes_to_entitlement[id]] for id in new_set)
        
        # compared_value = sum(etas[contributes_to_entitlement[id]] for id in gold_rush_committees[2*len(people)+iter_num])
        print(f'value = {value}')
        # print(f'compared value = {compared_value}')
        print(f'new committee = {sorted(list(new_set))}')
        # print(f'gold rush committee = {sorted(list(gold_rush_committees[2*len(people)+iter_num]))}')
        print(f'differentials max = {differentials.max()}')
        if value <= differentials.max() + EPS_TRUE_GL:
            probabilities = np.array(lambdas.value).clip(0, 1)
            probabilities = list(probabilities / sum(probabilities))
            end = time()
            runtime = end-start
            print("k/n", k/len(people))
            return committees, probabilities, output_lines, runtime
        else:
            if verbose:
                print("adding new committee to 24k gold")
                print(value, differentials.max(), value - differentials.max())
            assert new_set not in committees
            committees.append(new_set)
            start_lambdas = np.array(lambdas.value).resize(len(committees))

def find_opt_distribution_true_gold(k, gamma, categories, people, columns_data, check_same_address, check_same_address_columns, verbose = True, anon = True, EPS2=0.00000000001, EPS_TRUE_GL=1):
    # do true max and min using SOCP
    output_lines = []
    assert not check_same_address
    households = None
    start = time()

    # `new_committee_model` is an integer linear program (ILP) used for discovering new feasible committees.
    # We will use it many times, putting different weights on the inclusion of different agents to find many feasible
    # committees.
    new_committee_model, agent_vars, infeasible = _setup_committee_generation(categories, people, k, check_same_address, households)

    # Start by finding committees including every agent, and learn which agents cannot possibly be included.
    committees: List[FrozenSet[str]]  # set of feasible committees, add more over time
    covered_agents: FrozenSet[str]  # all agent ids for agents that can actually be included
    committee_set, covered_agents, new_output_lines = _generate_initial_committees(new_committee_model, agent_vars,
                                                                                2 * len(people))
    committees = list(committee_set)
    
    output_lines += new_output_lines
    output_lines += [f"Using true goldilocks with gamma={gamma}."]
    entitlements: List[str]
    contributes_to_entitlement: Dict[str, int]  # for id of a covered agent, the corresponding index in `entitlements`
    entitlements, contributes_to_entitlement = _define_entitlements(covered_agents)
    fv_vec_to_agents = fv_vecs(people, entitlements, contributes_to_entitlement)
    
    start_lambdas = [1 / len(committees) for _ in committees]
    iter_num = 0
    while True:
        iter_num += 1
        lambdas = cp.Variable(len(committees))  # probability of outputting a specific committee
        x = cp.Variable(1) # max term  
        y = cp.Variable(1) # gamma/min
        pis = cp.Variable(len(entitlements))
        lambdas.value = start_lambdas
        # A is a binary matrix, whose (i,j)th entry indicates whether agent `feasible_agents[i]`
        matrix = _committees_to_matrix(committees, entitlements, contributes_to_entitlement)
        assert matrix.shape == (len(entitlements), len(committees))

        balance_factor = len(people)/float(k)
        objective = cp.Maximize(-balance_factor * x - y /balance_factor)
        constraints = [0 <= lambdas, sum(lambdas) == 1, matrix @ lambdas == pis]
        num_pre_aux_constraints = len(constraints)
        y_constr_inds = []
        x_constr_inds = []
        for ent in range(len(entitlements)):
            y_constr_inds.append(len(constraints))
            constraints.append(cp.SOC(pis[ent] + y, cp.vstack([2*math.sqrt(gamma), pis[ent] - y])))
            # constraints.append(cp.norm(cp.vstack([2*math.sqrt(gamma), pis[ent] - y]), 2) <= pis[ent] + y)
            x_constr_inds.append(len(constraints))
            constraints.append(pis[ent] <= x)

        num_pre_anon_constraints = len(constraints)
        # copy constraints list
        constraints_copy = constraints.copy()
        # anon = False
        if anon:
            # add constraints so that all vectors with the same feature values have the same probability
            constraint_inds = [] # stores the tuples of indices of agents that we're constraining
            for fv_vec, agents in fv_vec_to_agents.items():
                if len(agents) == 1: continue
                for i in range(1, len(agents)):
                    constraints.append(pis[agents[0]] - pis[agents[i]] <= 0.01)
                    constraints.append(pis[agents[i]] - pis[agents[0]] <= 0.01)
                    constraint_inds.append((agents[0], agents[i]))
                    constraint_inds.append((agents[i], agents[0]))
        
        problem = cp.Problem(objective, constraints)
        gl_obj = problem.solve(solver=cp.ECOS, warm_start=False, verbose=False)
        # if the problem is infeasible, run it again but change the constraints to just be the first two
        curr_anon = anon
        if problem.status == "infeasible" and anon:
            # sometimes not feasible to have anonymity with start set
            print("infeasible problem, running again without anonymity")
            constraints = constraints_copy
            problem = cp.Problem(objective, constraints)
            gl_obj = problem.solve(solver=cp.ECOS, warm_start=True, verbose=False)
            curr_anon = False
            
        if verbose:
            output_lines.append(_print(f"Current GL objective is now: {-gl_obj}."))

        assert lambdas.value.shape == (len(committees),)

        entitled_utilities = pis.value
        assert entitled_utilities.shape == (len(entitlements),)
        assert (entitled_utilities > EPS2).all()
        dhdp = np.zeros(len(entitlements))
        sum_dgdp_terms = np.zeros(len(entitlements))
        for agent in range(len(entitlements)):
            pi = entitled_utilities[agent]
            is_max_pi = int(pi == entitled_utilities.max())
            min_partial, max_partial, extra_min_partials, extra_max_partials = 0, 0, 0, 0
            if is_max_pi:
                dhdp[agent] = -balance_factor
                print(f"agent {agent} is max pi")
                for ag_2 in range(len(entitlements)):
                    if ag_2 != agent:
                        mu_x_2_constraint = constraints[x_constr_inds[ag_2]].dual_value
                        max_x_2_partial = -is_max_pi
                        sum_dgdp_terms[agent] += max_x_2_partial * mu_x_2_constraint
                        extra_max_partials += max_x_2_partial * mu_x_2_constraint
                print(f"extra max partials = {extra_max_partials}")
                print(f"dhdp[agent] = {dhdp[agent]}")        
            is_min_pi = int(pi == entitled_utilities.min())
            if is_min_pi:
                dhdp[agent] = float(gamma)/(balance_factor * pi**2)
                print(f"agent {agent} is min pi")
                for ag_2 in range(len(entitlements)):
                    if ag_2 != agent:
                        mu_y_2_constraint = constraints[y_constr_inds[ag_2]].dual_value[0]
                        mu_y_2_partial = (float(gamma)/(pi**2)) * (1 + (entitled_utilities[ag_2] - y.value) * (4 * gamma + (entitled_utilities[ag_2] - y.value)**2)**(-0.5))
                        sum_dgdp_terms[agent] += mu_y_2_partial * mu_y_2_constraint
                        extra_min_partials += mu_y_2_partial * mu_y_2_constraint
            mu_x_constraint = constraints[x_constr_inds[agent]].dual_value
            mu_y_constraint = constraints[y_constr_inds[agent]].dual_value[0]
            min_partial = (4*gamma + (pi - y.value)**2)**(-0.5) * (pi - y.value) * (1 + gamma * is_min_pi/(pi**2)) + (gamma * is_min_pi/(pi**2)) - 1
            sum_dgdp_terms[agent] += min_partial * mu_y_constraint
            max_partial = 1 - is_max_pi
            sum_dgdp_terms[agent] += max_partial * mu_x_constraint
        if curr_anon:
            for ind, (i, j) in enumerate(constraint_inds):
                constr_dual_val = constraints[ind + num_pre_anon_constraints].dual_value
                sum_dgdp_terms[i] += constr_dual_val
                sum_dgdp_terms[j] -= constr_dual_val
        etas = dhdp - sum_dgdp_terms
        dhdp_dict = {i: dhdp[i] for i in range(len(entitlements))}
        sum_dgdp_terms_dict = {i: sum_dgdp_terms[i] for i in range(len(entitlements))}
        eta_dict = {i: etas[i] for i in range(len(entitlements))}
        # print('dhdp_dict', dhdp_dict)
        # print('sum_dgdp_terms_dict', sum_dgdp_terms_dict)
        # print('eta_dict', eta_dict)
            
        differentials = etas.dot(matrix) # sum for each committee
        assert differentials.shape == (len(committees),)

        obj = []
        for id in covered_agents:
            obj.append(etas[contributes_to_entitlement[id]] * agent_vars[id])
        new_committee_model.objective = mip.xsum(obj)
        new_committee_model.optimize()

        new_set = _ilp_results_to_committee(agent_vars)
        value = sum(etas[contributes_to_entitlement[id]] for id in new_set)
        
        # compared_value = sum(etas[contributes_to_entitlement[id]] for id in gold_rush_committees[2*len(people)+iter_num])
        print(f'value = {value}')
        # print(f'compared value = {compared_value}')
        print(f'new committee = {sorted(list(new_set))}')
        # print(f'gold rush committee = {sorted(list(gold_rush_committees[2*len(people)+iter_num]))}')
        print(f'differentials max = {differentials.max()}')
        if value <= differentials.max() + EPS_TRUE_GL:
            probabilities = np.array(lambdas.value).clip(0, 1)
            probabilities = list(probabilities / sum(probabilities))
            end = time()
            runtime = end-start
            return committees, probabilities, output_lines, runtime
        else:
            if verbose:
                print("adding new committee to goldilocks")
                print(value, differentials.max(), value - differentials.max())
            assert new_set not in committees
            committees.append(new_set)
            start_lambdas = np.array(lambdas.value).resize(len(committees))

def find_opt_distribution_gold(k, gamma, p, people, new_committee_model, agent_vars, committees, covered_agents, anon=True, verbose = True):
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"
    start = time()
    output_lines = [f"Using Goldilocks algorithm with parameter p ={p} and gamma={gamma}."]

    # Map the covered agents to indices in a list for easier matrix representation.
    entitlements: List[str]
    contributes_to_entitlement: Dict[str, int]  # for id of a covered agent, the corresponding index in `entitlements`
    entitlements, contributes_to_entitlement = _define_entitlements(covered_agents)
    fv_vec_to_agents = fv_vecs(people, entitlements, contributes_to_entitlement)
    # Now, the algorithm proceeds iteratively. First, it finds probabilities for the committees already present in
    # `committees` that maximize the sum of logarithms. Then, reusing the old ILP, it finds the feasible committee
    # (possibly outside of `committees`) such that the partial derivative of the sum of logarithms with respect to the
    # probability of outputting this committee is maximal. If this partial derivative is less than the maximal partial
    # derivative of any committee already in `committees`, the Karush-Kuhn-Tucker conditions (which are sufficient in
    # this case) imply that the distribution is optimal even with all other committees receiving probability 0.
    start_lambdas = [1 / len(committees) for _ in committees]
    while True:
        lambdas = cp.Variable(len(committees))  # probability of outputting a specific committee
        lambdas.value = start_lambdas
        # A is a binary matrix, whose (i,j)th entry indicates whether agent `feasible_agents[i]`
        matrix = _committees_to_matrix(committees, entitlements, contributes_to_entitlement)
        assert matrix.shape == (len(entitlements), len(committees))

        balance_factor = len(people)/float(k)
        # create an objective that maximizes the sum of p_i^c for each agent i
        objective = cp.Maximize(-balance_factor * cp.pnorm(matrix @ lambdas, p) - gamma * cp.pnorm(cp.power(matrix @ lambdas, -1), p) /balance_factor)
        constraints = [0 <= lambdas, sum(lambdas) == 1]
        if anon:
            # add constraints so that all vectors with the same feature values have the same probability
            constraint_inds = [] # stores the tuples of indices of agents that we're constraining
            for fv_vec, agents in fv_vec_to_agents.items():
                if len(agents) == 1: continue
                for i in range(1, len(agents)):
                    constraints.append(matrix[agents[0]]@lambdas - matrix[agents[i]]@lambdas <= 0.01)
                    constraints.append(matrix[agents[i]]@lambdas - matrix[agents[0]]@lambdas <= 0.01)
                    constraint_inds.append((agents[0], agents[i]))
                    constraint_inds.append((agents[i], agents[0]))
            # print(f"contributes_to_entitlement = {contributes_to_entitlement}")
            # print(f"constraint_inds = {constraint_inds}")
        
        problem = cp.Problem(objective, constraints)
        gl_obj = problem.solve(solver=cp.ECOS, warm_start=True, verbose=False)
        # if the problem is infeasible, run it again but change the constraints to just be the first two
        curr_anon = anon
        if problem.status == "infeasible" and anon:
            # sometimes not feasible to have anonymity with start set
            print("infeasible problem, running again without anonymity")
            constraints = [0 <= lambdas, sum(lambdas) == 1]
            problem = cp.Problem(objective, constraints)
            gl_obj = problem.solve(solver=cp.ECOS, warm_start=True, verbose=False)
            curr_anon = False
            
        if verbose:
            output_lines.append(_print(f"Current GL objective is now: {-gl_obj}."))

        assert lambdas.value.shape == (len(committees),)

        entitled_utilities = matrix.dot(lambdas.value)
        assert entitled_utilities.shape == (len(entitlements),)
        # print(f'min of entitled utilities = {entitled_utilities.min()}')
        assert (entitled_utilities > EPS2).all()
        max_term_factor = -balance_factor*(1.0/p) * (np.sum(entitled_utilities**p))**(1.0/p - 1)
        min_term_factor = ((1.0*gamma)/balance_factor) * (np.sum(1/entitled_utilities**p))**(1.0/p - 1)
        dhdp = max_term_factor * p * (entitled_utilities ** (p-1)) + min_term_factor * (1/(entitled_utilities**(p+1)))
        assert dhdp.shape == (len(entitlements),)
        sum_dgdp_terms = np.zeros(len(entitlements))
        if curr_anon:
            for ind, (i, j) in enumerate(constraint_inds):
                constr_dual_val = constraints[ind + 2].dual_value
                sum_dgdp_terms[i] += constr_dual_val
                sum_dgdp_terms[j] -= constr_dual_val
        etas = dhdp - sum_dgdp_terms
            
        differentials = etas.dot(matrix) # sum for each committee
        assert differentials.shape == (len(committees),)

        obj = []
        for id in covered_agents:
            obj.append(etas[contributes_to_entitlement[id]] * agent_vars[id])
        new_committee_model.objective = mip.xsum(obj)
        new_committee_model.optimize()

        new_set = _ilp_results_to_committee(agent_vars)
        value = sum(etas[contributes_to_entitlement[id]] for id in new_set)
        if value <= differentials.max() + EPS_GL:
            probabilities = np.array(lambdas.value).clip(0, 1)
            probabilities = list(probabilities / sum(probabilities))
            end = time()
            runtime = end-start
            return committees, probabilities, output_lines, runtime
        else:
            if verbose:
                print("adding new committee to goldilocks")
                print(value, differentials.max(), value - differentials.max())
            assert new_set not in committees
            committees.append(new_set)
            start_lambdas = np.array(lambdas.value).resize(len(committees))

def build_dictionaries(categories_df,respondents_df, k = 0.5, dropped_feats = 0):
    """ reads data into dictionaries
         categories: categories["feature"]["value"] is a dictionary with keys "min", "max", "selected", "remaining".
         people: people["nationbuilder_id"] is dictionary mapping "feature" to "value" for a person.
         columns_data: columns_data["nationbuilder_id"] is dictionary mapping "contact_field" to "value" for a person.
         k: number of people wanted on panel, only need this as input if dropped_feats > 0
         dropped_feats: how many features to drop with maximal ssb"""
    categories = {}
    people = {}
    columns_data = None # unset because never used
    
    # assert that k is not None if dropped_feats > 0
    assert dropped_feats == 0 or k != 0.5 
        
    p_fv = {}
    feature_ssb = {}
    
    # fill categories
    for category in list(categories_df['category'].unique()):
        p_fv[category] = {}
        categories[category] = {}
        for value in list(categories_df[categories_df['category']==category]['name'].unique()):
            lower_quota = int(categories_df[(categories_df['category']==category) & (categories_df['name']==value)]['min'].values[0])
            upper_quota = int(categories_df[(categories_df['category']==category) & (categories_df['name']==value)]['max'].values[0])
            categories[category][value] = {"min":lower_quota, "max":upper_quota, "selected":0, "remaining":respondents_df.count()}
            p_fv[category][value] = (lower_quota + upper_quota)/(2.0 * k)
    
    assert dropped_feats < len(categories)
    
    # fill people
    for nationbuilder_id in list(respondents_df['nationbuilder_id'].values):
        people[nationbuilder_id] = {}
        for category in list(categories_df['category'].unique()):
            people[nationbuilder_id][category] = respondents_df[respondents_df['nationbuilder_id']==nationbuilder_id][category].values[0]

    # print("p_fv", p_fv)
    # print("p keys ", p_fv.keys())
    for feature in categories:
        fv_ssb = []
        for value in categories[feature]:
            fv_agents = [id for id, person in people.items() if person[feature] == value]
            num_fv_agents = (len(fv_agents))/(len(people)*1.0)
            # print("feature, value, num_fv_agents: ", feature, value, num_fv_agents)
            if p_fv[feature][value] > 0 and num_fv_agents > 0:
                fv_ssb += [p_fv[feature][value] / num_fv_agents]
        # print(f"feature: {feature}, fv_ssb: {fv_ssb}")
        feature_ssb[feature] = np.max(fv_ssb) - np.min(fv_ssb)
    
    
    while dropped_feats > 0:
        # find feature with max ssb
        remove_feat = None
        max_ssb = -1
        for feature in categories:
            if feature_ssb[feature] > max_ssb:
                remove_feat = feature
                max_ssb = feature_ssb[feature]
        # print("remove_feat: ", remove_feat)
        categories.pop(remove_feat)
        # for category in categories:
        #     if category != remove_feat:
        #         new_cats[category] = categories[category]
        new_people = {}
        for nationbuilder_id in list(respondents_df['nationbuilder_id'].values):
            new_people[nationbuilder_id] = {}
            for category in categories:
                new_people[nationbuilder_id][category] = respondents_df[respondents_df['nationbuilder_id']==nationbuilder_id][category].values[0]
        people = new_people
        # print(f"after removing {remove_feat} we have categories: {categories} and people: {people}")
        dropped_feats -= 1
    
    return categories, people, columns_data





def randomized_round_pipage(probabilities,M):
    """implements pipage rounding as in Gandhi et al 2006.
       inputs: probabilities - probabilities associated with each panel
               M - number of panels over which you want the uniform lottery to be
    """
    # NOTE: had a precision issue where I wasn't getting >= 0.99999 for the last probability left with obf, was getting >= 0.9999, so changed to that 
    floor_scaled_probs = [math.floor(p*M) for p in probabilities]
    rem_scaled_probs = [a_i - b_i for a_i, b_i in zip([M*p for p in probabilities], floor_scaled_probs)]

    unround_inds = list(range(len(probabilities)))
    # want to randomly round remainders of scaled probabilities to 0 / 1 while preserving their sum & having negative correlation: pipage rounding
    while len(unround_inds)>0:

        # set p1, p2. if either one is already rounded, take it out and reset loop.
        p1 = rem_scaled_probs[unround_inds[0]]
        if (p1>=0.9999 and p1 <=1.0000001) or (p1 >= -0.0000001 and p1 <= 0.0000001):
            unround_inds.pop(0)
            continue 

        p2 = rem_scaled_probs[unround_inds[1]]
        if (p2>=0.9999 and p2 <=1.0000001) or (p2 >= -0.0000001 and p2 <= 0.0000001):
            unround_inds.pop(1)
            continue


        alpha = min(1-p1, p2)  # will be added to p1, subtracted rom p2
        beta = min(p1,1-p2)    # will be added to p2, subtracted from p1


        # with probability alpha / (alpha + beta), do beta modification:
        d = random.uniform(0,1)
        if d <= alpha / (alpha+beta):
            rem_scaled_probs[unround_inds[0]] = p1 - beta
            rem_scaled_probs[unround_inds[1]] = p2 + beta
        else:
            rem_scaled_probs[unround_inds[0]] = p1 + alpha
            rem_scaled_probs[unround_inds[1]] = p2 - alpha

        if (p1>=0.9999 and p1 <=1.0000001) or (p1 >= -0.0000001 and p1 <= 0.0000001):
            unround_inds.pop(0)
        if (p2>=0.9999 and p2 <=1.0000001) or (p2 >= -0.0000001 and p2 <= 0.0000001):
            unround_inds.pop(1)

    result = [(floor_scaled_probs[i]+rem_scaled_probs[i])/M for i in range(len(probabilities))]

    return result


def beckfiala_round(committees,probabilities,people,M,k):
    """implements dependent rounding as in Flanigan et al 2020.
       inputs: committees - list of all panels in support of optimal unconstrained distribution
               probabilities - probabilities associated with each panel in committees
               people - list of people in all committees
               M - number of panels over which you want the uniform lottery to be
               k - panel size
    """

    probs_round = [int(p*M) for p in probabilities]
    curr_probs = [probabilities[i]*M - probs_round[i]for i in range(len(probabilities))]

    # find value of target probability of each agent
    target_agent_probs = {id: 0 for id in people}
    num_active_committees_agent = {id: 0 for id in people}
    for c in range(len(committees)):
        for id in committees[c]:
            target_agent_probs[id] = target_agent_probs[id] + curr_probs[c]
            num_active_committees_agent[id] += 1

    model = grb.Model()

    # VARIABLES
    committee_variables = [model.addVar(lb=0.,ub=1) for c in committees] 

    agent_panel_variables = {id: [] for id in people}
    for committee, var in zip(committees, committee_variables):
        for id in committee:
            agent_panel_variables[id].append(var)

    # LP
    model.addConstr(grb.quicksum(committee_variables)==sum(curr_probs)) # sum must be preserved

    agent_constraints = {}
    for id in people:
        agent_constraints[id] = model.addConstr(grb.quicksum(agent_panel_variables[id])==target_agent_probs[id])
        

    optimistic_marginals = {id : num_active_committees_agent[id] for id in people}
    pessimistic_marginals = {id : 0 for id in people}


    # Iteratively solve the LP, dropping the second type of constraints as we go
    determined_variables = {}
    while True:

        model.optimize()
        assert model.status == grb.GRB.OPTIMAL


        for cnum,C in enumerate(committee_variables):
            if cnum in determined_variables:
                continue
            
            lp_value = C.X
            if lp_value < EPS: 
                determined_variables[cnum] = False 
                model.addConstr(C==0.)
                for id in committees[cnum]:
                    optimistic_marginals[id] -=1
                    num_active_committees_agent[id] -= 1


            elif lp_value > 1-EPS:
                determined_variables[cnum] = True
                model.addConstr(C==1.)

                for id in committees[cnum]:
                    pessimistic_marginals[id] += 1
                    num_active_committees_agent[id] -= 1


        if len(determined_variables) == len(committees):
            # construct rounded probabilities:
            for cnum in range(len(committees)):
                probs_round[cnum] = (probs_round[cnum] + committee_variables[cnum].X)/M

            return probs_round

        # drop any constraints that are almost satisfied, within tolerance of k
        constraints_to_delete = []
        for id in agent_constraints:

            if (pessimistic_marginals[id] >= target_agent_probs[id] - k) and (optimistic_marginals[id] <= target_agent_probs[id] + k):
                constraints_to_delete.append(id)

            # if agent is on all active panels
            elif num_active_committees_agent[id] == len(committees) - len(determined_variables):
                constraints_to_delete.append(id)

        assert len(constraints_to_delete) > 0

        # remove constraints listed for deletion
        for id in constraints_to_delete:
            model.remove(agent_constraints[id])
            del agent_constraints[id]
        


def minimax_change_round(committees,probabilities,people,marginals,M):
    """ finds uniform lottery that minimizes the maximum deivation of any agent's marginal from those implied by optimal distribution 
        inputs: committees = list of committees in support of optimal unconstrained distribution
                probabilities = probabilities of choosing all panels in optimal unconstrained distribution
                people = list of agents included on any committee in committees (should be all agents)
                marginals = marginals given by probabilities, the optimal distribution over panels
                M = the number of panels over which you want a uniform lottery
        outputs: vector of probabilities, one assigned to each committee (in order of committees list)
    """

    model = mip.Model(sense=mip.MINIMIZE)

    # will assign integer between 1 and M to every committee, they add to M
    committee_variables = [model.add_var(var_type=mip.INTEGER, lb=0., ub=mip.INF) for _ in committees]
    model.add_constr(mip.xsum(committee_variables) == M)
    
    # people-on-panel variables
    agent_panel_variables = {id: [] for id in people}
    for committee, var in zip(committees, committee_variables):
        for id in committee:
            if id in people:
                agent_panel_variables[id].append(var)

    upper = model.add_var(var_type=mip.CONTINUOUS, lb=0.)

    # sum of all variables pertaining to a given agent 
    for id in people:
        model.add_constr(marginals[id]*M - mip.xsum(agent_panel_variables[id]) <= upper)
        model.add_constr(mip.xsum(agent_panel_variables[id]) - marginals[id]*M <= upper)


    model.objective = upper

    
    model.optimize(max_seconds=1800)
    rounded_probabilities = [round(var.x) / M for var in committee_variables]

    return rounded_probabilities




def compute_marginals(committees,probabilities,n):
    marginals = [0 for i in range(n)]

    for c in range(len(committees)):
        committee = committees[c]

        for i in committee:
                marginals[i] = marginals[i] + probabilities[c]

    # print max and min marginals
    print("max marginal:", max(marginals))
    print("min marginal:", min(marginals))
    return marginals


def save_results(committees,probabilities,filestem,n,rep=None):

    # save panel distribution
    results_df = pd.DataFrame({'committees':committees, 'probabilities':probabilities})
    print("filestem:", filestem)
    if rep==None:   
        results_df.to_csv(filestem+'probabilities.csv')
    else:
        results_df.to_csv(filestem+'probabilities_rep'+str(rep)+'.csv')

    # compute and save marginals
    marginals = compute_marginals(committees,probabilities,n)
    marginals_df = pd.DataFrame({'marginals':marginals})
    if rep==None:
        marginals_df.to_csv(filestem+'marginals.csv')
    else:
        marginals_df.to_csv(filestem+'marginals_rep'+str(rep)+'.csv')
        


# # # # # # # # # # # # # # # # MAIN # # # # # # # # # # # # # # # # # # #

if __name__== "__main__":
    timings = {}

    # min_max_table(100)
    for instance in instances:
        timings[instance] = {}
        
        

        # read in & construct necessary information about instance
        categories_df = pd.read_csv('../input-data/'+instance+'/categories.csv')
        respondents_df = pd.read_csv('../input-data/'+instance+'/respondents.csv')
        n = len(respondents_df)
        k = int(instance[instance.rfind('_')+1:])
        num_dropped_feats = 0

        categories, people, columns_data = build_dictionaries(categories_df,respondents_df, k, dropped_feats=num_dropped_feats)



        objectives = {}
        if LEXIMIN==1:
            objectives['leximin'] = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_leximin_'
        if MAXIMIN==1:
            objectives['maximin'] = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_maximin_'
        if NASH==1:
            objectives['nash'] = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_nash_'
        if MINIMAX==1:
            objectives['minimax'] = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_minimax_'
        if GOLDILOCKS==1:
            objectives['goldilocks'] = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_goldilocks_'
        if LEGACY==1:
            objectives['legacy'] = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_legacy_'
        if TRUE_GOLD==1:
            objectives['true_goldilocks'] = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_true_goldilocks_'
        if GOLD24K==1:
            objectives['gold24k'] = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/'+instance+'/'+instance+'_m'+str(M)+'_gold24k_'
           
        alg_probs = {}
        for obj in objectives:
            stub = objectives[obj]
            if OPT == 1:
                if RERUN == 0 and os.path.isfile(stub + 'opt_marginals.csv'):
                    print("already have optimal solution for", obj, "on", instance)
                else: 
                    if obj =='leximin':
                        committees, probabilities, output_lines, runtime = find_opt_distribution_leximin(categories, people,
                                                                    columns_data, k, check_same_address, check_same_address_columns)
                        timings[instance]["leximin"] = runtime
                    if obj == 'maximin':
                        committees, probabilities, output_lines, infeasible, runtime = find_opt_distribution_maximin(categories, people,
                                                                    columns_data, k, check_same_address, check_same_address_columns)
                        timings[instance]["maximin"] = runtime
                    if obj == 'nash':
                        committees, probabilities, output_lines, runtime = find_opt_distribution_nash(categories, people, columns_data, 
                                                                    k, check_same_address, check_same_address_columns)
                        timings[instance]["nash"] = runtime
                    if obj == 'minimax':
                        committees, probabilities, output_lines, infeasible, runtime = find_opt_distribution_minimax(categories, people,
                                                                    columns_data, k, check_same_address, check_same_address_columns)
                        timings[instance]["minimax"] = runtime
                    if obj == 'legacy':
                        output_lines = []
                        marginals = legacy_probabilities(0, 10000, categories, people,
                                                                    columns_data, k, check_same_address, check_same_address_columns)
                        marginals_df = pd.DataFrame({'marginals':marginals})
                        marginals_df.to_csv(stub+'opt_marginals.csv')
                    if obj == 'goldilocks':
                        ps = [50]
                        scales = [1]
                        # gammas = compute_gammas(n, k, categories, people, instance, num_dropped_feats)
                        gammas = {"gamma_1":1}
                        committees, probabilities, output_lines, gl_timings = gold_rush(stub, gammas, scales, ps, categories, people, k, check_same_address, anon=True, save_res=False)    
                        # timings[instance]["goldilocks"] = value
                        for key, value in gl_timings.items():
                            timings[instance]["goldilocks"] = value
                    if obj == 'true_goldilocks':
                        EPS_TRUE_GL = 1.0
                        # gammas = compute_gammas(n, k, categories, people, instance, num_dropped_feats)
                        committees, probabilities, output_lines, runtime = find_opt_distribution_true_gold(k, 1, categories, people, columns_data, check_same_address, check_same_address_columns, anon=True, EPS_TRUE_GL=EPS_TRUE_GL)
                        timings[instance]["true_gold"] = runtime
                    if obj == 'gold24k':
                        EPS_TRUE_GL = 1.0
                        committees, probabilities, output_lines, runtime = find_opt_distribution_gold24k(k, 1, categories, people, columns_data, check_same_address, check_same_address_columns, anon=True, EPS_TRUE_GL=EPS_TRUE_GL)
                    print(output_lines)
                    if obj not in ['legacy', 'goldilocks']: save_results(committees, probabilities, stub + 'opt_',n)

            # read in committees from OPT solution for rest of rounding computations
            if obj != 'legacy':
                if obj == 'goldilocks':
                    stub += 'p=50_gamma_1_scale=1_'
                results_df = pd.read_csv(stub + 'opt_probabilities.csv')
                committees = [[int(results_df['committees'].values[i][11:-2].split(',')[j]) for j in range(len(results_df['committees'].values[i][11:-2].split(',')))] for i in range(len(list(results_df['committees'].values)))]
                probabilities = results_df['probabilities']
                marginals_df = pd.read_csv(stub + 'opt_marginals.csv')
                marginals = marginals_df['marginals'].values

            if ILP == 1: # note: ILP is only a valid choice for NASH or MAXIMIN
                lowercase_obj = obj.lower()
                round_dir = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/ILProunded/'+instance+'/'
                if not os.path.exists(round_dir):
                    os.makedirs(round_dir)
                round_stub = round_dir+instance+'_m'+str(M)+'_'+ lowercase_obj + '_'
                if obj =='maximin':
                    probabilities_rounded = _find_maximin_primal_discrete(committees, people, M)
                if obj =='nash':
                    probabilities_rounded = _find_nash_primal_discrete_gurobi(committees,people,M)
                if obj == 'minimax':
                    probabilities_rounded = _find_minimax_primal_discrete(committees, people, M)
                    #probabilities_rounded = find_rounded_distribution_nash(committees,people,M) # solve with baron solver instead
                if obj == 'goldilocks' or obj == 'true_goldilocks':
                    probabilities_rounded = _find_goldilocks_primal_discrete_gurobi(n, k, committees, people, M) # discrete sol uses true gold anyways

                save_results(committees,probabilities_rounded, round_stub+'ILProunded_',n)
                # get marginals from saved file
                marginals_df = pd.read_csv(round_stub + 'ILProunded_marginals.csv')
                marginals = marginals_df['marginals'].values
                with open(round_stub + 'ILProunded_marginals_stats.txt', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Min Marg", "Max Marg"])
                    writer.writerow([min(marginals), max(marginals)])
                    print("writing to ", round_stub + 'ILProunded_marginals_stats.txt')
                
                

            if ILP_MINIMIAX_CHANGE == 1:   
                probabilities_rounded = minimax_change_round(committees,probabilities,people,marginals,M)
                save_results(committees,probabilities_rounded,stub + 'ILP_MMC_rounded_',n)


            if BECK_FIALA == 1:
                probabilities_rounded = beckfiala_round(committees,probabilities,people,M,k)
                save_results(committees,probabilities_rounded, stub+'BFrounded_',n)

            if RANDOMIZED == 1:
                DELETE_INTERMED = 1
                # make a directory for the instance if it doesn't exist
                lowercase_obj = obj.lower()
                round_dir = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/pipage/'+instance+'/'+lowercase_obj+'/'
                if not os.path.exists(round_dir):
                    os.makedirs(round_dir)
                round_stub = round_dir+instance+'_m'+str(M)+'_'+ lowercase_obj + '_'
                min_marg = []
                max_marg = []
                avg_margs = [0.0]*n # running average for each agent
                for rep in range(RANDOMIZED_REPLICATES):
                    probabilities_rounded = randomized_round_pipage(probabilities,M)
                    save_results(committees,probabilities_rounded,round_stub + 'RANDrounded_',n,rep)
                    # get marginals from saved file
                    marginals_df = pd.read_csv(round_stub + 'RANDrounded_marginals_rep'+str(rep)+'.csv')
                    marginals = marginals_df['marginals'].values
                    min_marg.append(min(marginals))
                    max_marg.append(max(marginals))
                    avg_margs = [(avg_margs[i]*rep + marginals[i])/(rep+1) for i in range(n)]
                    if DELETE_INTERMED:
                        os.remove(round_stub + 'RANDrounded_probabilities_rep'+str(rep)+'.csv')
                        os.remove(round_stub + 'RANDrounded_marginals_rep'+str(rep)+'.csv')
                    if rep%100==0:
                        print(rep)
                summ_file = '../intermediate_data/dropped_'+str(num_dropped_feats)+'/pipage/'+instance+'/'+instance+'_m'+str(M)+'_'+ lowercase_obj + '_RANDrounded_marginals_full_stats.txt'
                # save a file that has the average min marg, average max marg, and min of the avg_margs, and max of the avg_margs
                with open(summ_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Avg Min Marg", "Std Min Marg", "Avg Max Marg", "Std Max Marg", "Min Avg Marg", "Max Avg Marg"])
                    writer.writerow([np.average(min_marg), np.std(min_marg), np.average(max_marg), np.std(max_marg), min(avg_margs), max(avg_margs)])
                    print("writing to ", summ_file)
    save_timings(timings, "../new_timings.txt")

    



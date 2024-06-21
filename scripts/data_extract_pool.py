import sys
import numpy as np
import pandas as pd
import csv
from paper_data_analysis import build_dictionaries

# reads datasets and gets the necessary info from the datasets
def extract_pool(instance):
    k = int(instance[instance.rfind('_')+1:])    
    categories_df = pd.read_csv('../input-data/'+instance+'/categories.csv')
    respondents_df = pd.read_csv('../input-data/'+instance+'/respondents.csv')
    categories, people, column_data = build_dictionaries(categories_df, respondents_df, k, dropped_feats=0)

    n = len(people)
    F = len(categories)
    
    # make sure people are indexed by 0, 1, 2, ...
    assert (list(people.keys()) == np.arange(n)).all()
    # Create Dict[Dict(Str, Float)] of f -> v -> p_fv (population freq)
    p_fv = {}
    # Create Dict[Dict(Str, Float)] of f -> v -> nu_fv (pool freq)
    nu_fv = {}
    ratios_fv = {}
    # construct most underrepresented vector
    underrepresented_fv = {}
    for feature, feature_values in categories.items():
        nu_fv[feature] = {}
        p_fv[feature] = {}
        ratios_fv[feature] = {}
        for value in feature_values.keys():
            fv_agents = [id for id, person in people.items() if person[feature] == value]
            nu_fv[feature][value] = len(fv_agents)/n
            p_fv[feature][value] = (categories[feature][value]['min'] + categories[feature][value]['max']) / (2.0*k) # doing avg of min and max bc don't want 0s or ns
            if p_fv[feature][value] == 0: 
                continue
            ratios_fv[feature][value] = nu_fv[feature][value] / p_fv[feature][value]
        # print("nu_fv", nu_fv)
        # print("p_fv", p_fv)
        # print("ratios_fv", ratios_fv)
        underrepresented_fv[feature] = min(ratios_fv[feature], key=ratios_fv[feature].get)

    # Create fixed ordering of features so we can check if we've already checked a vector
    ordered_features = sorted(list(categories.keys()))
    people_tups = {} # mapping people to tuples
    for i in range(n):
        feature_vector = [0]*F
        for f_ind in range(F):
            feature = ordered_features[f_ind]
            feature_value = people[i][feature]
            feature_vector[f_ind] = feature_value
        feature_vector = tuple(feature_vector)
        people_tups[i] = feature_vector

    return {
        "n":n,
        "k":k,
        "F":F,
        "p_fv":p_fv,
        "nu_fv":nu_fv,
        "ratios_fv":ratios_fv,
        "underrepresented_fv":underrepresented_fv,
        "people":people,
        "people_tups":people_tups,
        "categories":categories,
        "column_data":column_data,
    }
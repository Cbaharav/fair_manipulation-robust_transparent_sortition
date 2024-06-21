import multiprocessing as mp
from optimizer import Instance
import utils
import copy
from tqdm import tqdm
import itertools
import os

def copyInstance(instance: Instance):
    """ Copy an instance """
    newInstance = Instance(instance.instance_name, instance.old_n, instance.new_n, instance.k, instance.categories, instance.people, instance.tuple_to_people, instance.obj, instance.columns_data, instance.num_unique_vectors, instance.num_dropped_feats)
    return newInstance

def deviator(args):
    """ Calculate deviation for a single defection """
    # make sure that the code is just running on one cpu
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    # Construct instance post-defection
    (manip_vec_num, old_instance) = args
    person_num = old_instance.deviating_people[manip_vec_num]
    # check if a file already exists for this manipulation vector and person
    if os.path.exists(old_instance.stub+"manipvec" + str(manip_vec_num)+"_person"+str(person_num)+"_deviation_benefit.txt"):
        print("manipvec" + str(manip_vec_num)+"_person"+str(person_num)+"_deviation_benefit.txt already exists, returning")
        return
    newInstance = copyInstance(old_instance)
    newPool = copy.deepcopy(old_instance.people)
    newPool[person_num] = old_instance.underrepped_fv # one person with this vector defects
    newInstance.setPeople(newPool)
    # Find new marginal probabilities
    newMarginals = newInstance.calculateMarginals(old_instance.stub+"manipvec" + str(manip_vec_num)+"_person"+str(person_num)+"_")
    margs_for_tup = [newMarginals[i] for i in old_instance.underrepped_people]
    if person_num not in old_instance.underrepped_people:
        margs_for_tup.append(newMarginals[person_num])
    new_marg_avgs = sum(margs_for_tup)/float(len(margs_for_tup))
    deviation_benefit = new_marg_avgs - old_instance.old_marg_avgs[manip_vec_num]
    with open(old_instance.stub+"manipvec" + str(manip_vec_num)+"_person"+str(person_num)+"_deviation_benefit.txt", 'a') as f:
        # write in new_marg_avgs, old_marg_avgs[starting_vector], deviation_benefit
        f.write('%s,%s,%s,%s,%s\n' % (old_instance.fixed_tuples[manip_vec_num], old_instance.underrepped_fv, new_marg_avgs, old_instance.old_marg_avgs[manip_vec_num], deviation_benefit)) 

################################################################################################################
def strategy_underrepresented_feature_values_all(instance: Instance):
    """ Determine manipulation gain from deviating to the most underrepresented feature values

    instance - pool instance
    underrepresented_fv - feature vector with the most underrepresented feature values
    feature_vectors - list of feature vectors that exist in the pool
    """
    
    args = [(i, instance) for i in range(len(instance.fixed_tuples))]
    with mp.Pool(processes=3) as pool:
        pool.map(deviator, args)


    
    

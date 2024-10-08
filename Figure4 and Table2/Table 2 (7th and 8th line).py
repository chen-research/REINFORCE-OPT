from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import os,gc

def objective_2d(sol,m1=np.array([-0.5,-0.5]),m2=np.array([0.5,0.5])):
    """
    This function computes the objective to be minimized.
    It process multiple solutions.
    
    Inputs
    --------
    solution: np.ndarray, each row is a x-instance.
    
    Outputs
    --------
    output: np.1darray, the ith entry is the objective value of the ith row of solution.
    """
    output = -np.log(np.sum((sol-m1)**2,axis=1)+0.00001)-np.log(np.sum((sol-m2)**2,axis=1)+0.01)
    return output

def objective_6d(sol,m1=np.array([-0.5]*6),m2=np.array([0.5]*6)):
    """
    This function computes the objective to be minimized.
    It process multiple solutions.
    
    Inputs
    --------
    solution: np.ndarray, each row is a x-instance.
    
    Outputs
    --------
    output: np.1darray, the ith entry is the objective value of the ith row of solution.
    """
    output = -np.log(np.sum((sol-m1)**2,axis=1)+0.00001)-np.log(np.sum((sol-m2)**2,axis=1)+0.01)
    return output


#Cross entropy method for optimization
def cem_gaussian(init_generation, fitness_fcn, elite_ratio=0.1, 
                 update_weight=0.4, generation_num=1000, verbose=False):
    """
    This function performs the cross-entropy method for optimization, for a 2D case.
    Reference: The Cross-Entropy Method for Optimization - Z. I. Botev1 et al. Handbook of Statistics.
    Algorithm 4.1, p50.
    
    Inputs.
    ---------
    init_generation:np.2darray, each row is a candidate solution.
    fitness_fcn, the fitness function.
    elite_ratio:float, the portion of best candidates in each generation used to produce a Gaussian distribution.
    update_weight:float, mu = (1-update_weight)*mu + update_weight*updated_mu
    generation_num:number of generations to be generated.
    verbose:binary, whether to show the training process.
    
    Outputs.
    ---------
    best_solution:np.1darray, the solution with the largest fitness value among all the generations.
    best_fitness:float, fitness of the best solution.
    """
    dim = init_generation.shape[1]
    pop_size = init_generation.shape[0]
    elite_num = int(pop_size*elite_ratio) #number of best x-values to keep in each generation
    init_fitness = fitness_fcn(init_generation)
    best_index = np.argsort(init_fitness)[-elite_num:]
    elites = init_generation[best_index]
    mu = np.mean(elites,axis=0)
    sigma = np.std(elites,axis=0)
    generation = np.zeros((pop_size,dim),dtype=np.float) #updated generation holder
    update_weight = 0.4
    best_solution = elites[-1]
    best_fitness = init_fitness[best_index[-1]]

    for n in range(generation_num):
        #produce a new generation
        for d in range(dim):
            generation[:,d] = np.random.normal(loc=mu[d], scale=sigma[d]+1e-17, size=(pop_size))
            
        fitness = fitness_fcn(generation)
        best_index = np.argsort(fitness)[-elite_num:]
        elites = generation[best_index]
        mu = (1-update_weight)*mu + update_weight*np.mean(elites,axis=0)
        sigma = (1-update_weight)*sigma + update_weight*np.std(elites,axis=0)    
        current_best_fit = fitness[best_index[-1]] #best fitness of this generation
    
        if current_best_fit > best_fitness:
            best_fitness = current_best_fit
            best_solution = elites[-1]
            if verbose:
                print("n:",n,"best_fitness:","mu",mu.round(3), best_fitness.round(3), "best solution:", best_solution.round(2))
    
    return [best_solution,best_fitness]

#Hyper_parameter space
elite_ratio_space = (0.05,0.1,0.2,0.5) 
update_weight_space = (0.9, 0.7, 0.5, 0.2)

################################################### - 7th Line in Table-2 
r = np.random.RandomState(0)
pop_size = 600 #number of individuals in a generation
cem_init_population = r.uniform(low=-1.0, high=1.0, size=(pop_size,2))
final_fitness = -1000
final_param = {'elite ratio':None, 'update weight':None}
for er in elite_ratio_space:
    for up_w in update_weight_space:
        best_sol, best_fit = cem_gaussian(
                                          init_generation=cem_init_population,
                                          fitness_fcn=objective_2d,
                                          elite_ratio=er, 
                                          update_weight=up_w, 
                                          generation_num=1000,
                                          verbose=False
                                          )
        if best_fit>final_fitness:
            final_param['elite ratio'] = er
            final_param['update weight'] = up_w
            final_solution = best_sol
            final_fitness = best_fit

print("Final Solution:", final_solution, "Final Fitness:", final_fitness, final_param)


################ - 8th Line in Table-2 
r = np.random.RandomState(0)
pop_size = 600 #number of individuals in a generation
cem_init_population = r.uniform(low=0.0, high=1.0, size=(pop_size,2))
final_fitness = -1000
final_param = {'elite ratio':None, 'update weight':None}
for er in elite_ratio_space:
    for up_w in update_weight_space:
        best_sol, best_fit = cem_gaussian(
                                          init_generation=cem_init_population,
                                          fitness_fcn=objective_2d,
                                          elite_ratio=er, 
                                          update_weight=up_w, 
                                          generation_num=1000,
                                          verbose=False
                                          )
        if best_fit>final_fitness:
            final_param['elite ratio'] = er
            final_param['update weight'] = up_w
            final_solution = best_sol
            final_fitness = best_fit

print("Final Solution:", final_solution, "Final Fitness:",final_fitness, final_param)
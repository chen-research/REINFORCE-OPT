#import driver
from tf_agents.drivers import py_driver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

#import environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment #allows parallel computing for generating experiences
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment

#import replay buffer
from tf_agents import replay_buffers as rb

#import agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.utils import value_ops
from tf_agents.trajectories import StepType


#other used packages
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.axes import Axes as ax
import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.networks import actor_distribution_network
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.networks.categorical_projection_network import CategoricalProjectionNetwork
import os,gc
import pygad
from pyswarms.single.global_best import GlobalBestPSO
import time


#To limit TensorFlow to a CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cpus = tf.config.experimental.list_physical_devices('CPU') 
tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
#enable multiprocessing for parallel computing
tf_agents.system.multiprocessing.enable_interactive_mode()
gc.collect()

###############
#Define the objective f to be maximized
N = 1   #This the dimension number
state_dim = N
def f(x,lambda_coef=0.3):
    return -(x**2 -1)**2 - lambda_coef*(x-1)**2 + 5

################
#Set initial x-values for all three algorithms
x0_reinforce = np.array([-1.5],dtype=np.float32)
sub_episode_length = 35 #number of time_steps in a sub-episode. 
episode_length = sub_episode_length*6  #an trajectory starts from the initial timestep and has no other initial timesteps
                                      #each trajectory will be split to multiple episodes
env_num = 10  #Number of parallel environments, each environment is used to generate an episode
print('x0', x0_reinforce)

################
#Set hyper-parameters for REINFORCE-OPT
generation_num = 3500  #number of theta updates for REINFORCE-IP, also serves as the number
                      #of generations for GA, and the number of iterations for particle swarm optimization

disc_factor = 1.0
alpha = 0.2 #regularization coefficient
param_alpha = 0.15 #regularization coefficient for actor_network #tested value: 0.02
sub_episode_num = int(env_num*(episode_length/sub_episode_length)) #number of sub-episodes used for a single update of actor_net params
print("number of sub_episodes used for a single param update:", sub_episode_num)

#Learning Schedule = initial_lr * (C/(step+C))
class lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, C):
        self.initial_learning_rate = initial_lr
        self.C = C
    def __call__(self, step):
        return self.initial_learning_rate*self.C/(self.C+step)
lr = lr_schedule(initial_lr=0.0001, C=50000)   
opt = tf.keras.optimizers.legacy.SGD(learning_rate=lr)
#opt = tf.keras.optimizers.Adam( )

train_step_num = 0


act_min = -1
act_max = 1
step_size = 0.1
def compute_reward(x):
    return f(x)
    
#Define the Environment
class Env(py_environment.PyEnvironment):
    def __init__(self):
        '''The function to initialize an Env obj.
        '''
        #Specify the requirement for the value of action, (It is a 2d-array for this case)
        #which is an argument of _step(self, action) that is later defined.
        #tf_agents.specs.BoundedArraySpec is a class.
        #_action_spec.check_array( arr ) returns true if arr conforms to the specification stored in _action_spec
        self._action_spec = array_spec.BoundedArraySpec(
                            shape=(state_dim,), dtype=np.int32, minimum=0, maximum=act_max-act_min, name='action') #a_t is an 2darray
    
        #Specify the format requirement for observation (It is a 2d-array for this case), 
        #i.e. the observable part of S_t, and it is stored in self._state
        self._observation_spec = array_spec.BoundedArraySpec(
                                 shape=(state_dim,), dtype=np.float32, name='observation') #default max and min is None
        self._state = x0_reinforce
        #self.A = mat
        self._episode_ended = False
        #stop_threshold is a condition for terminating the process for looking for the solution
        #self._stop_threshold = 0.01
        self._step_counter = 0

    def action_spec(self):
        #return the format requirement for action
        return self._action_spec

    def observation_spec(self):
        #return the format requirement for observation
        return self._observation_spec

    def _reset(self):
        self._state = x0_reinforce  #initial state
        self._episode_ended = False
        self._step_counter = 0
        
        #Reward
        initial_r = np.float32(0.0)
        
        #return ts.restart(observation=np.array(self._state, dtype=np.float32))
        return ts.TimeStep(step_type=StepType.FIRST, 
                           reward=initial_r, 
                           discount=np.float32(disc_factor), 
                           observation=np.array(self._state, dtype=np.float32)
                           )
    
    def set_state(self,new_state):
        self._state = new_state
    
    def get_state(self):
        return self._state
    
    def _step(self, action):
        '''
        The function for the transtion from (S_t, A_t) to (R_{t+1}, S_{t+1}).
    
        Input.
        --------
        self: contain S_t.
        action: A_t.
    
        Output.
        --------
        an TimeStep obj, TimeStep(step_type_{t+1}, R_{t+1}, discount_{t+1}, observation S_{t+1})
        ''' 
        # Suppose that we are at the beginning of time t 
        
        ################## --- Determine whether we should end the episode.
        if self._episode_ended:  # its time-t value is set at the end of t-1
            return self.reset()
        # Move on to the following if self._episode_ended=False
        
        
        ################# --- Compute S_{t+1} 
        #Note that we set the action space as a set of non-negative vectors
        #action-act_max converts the set to the desired set of negative vectors.
        self._state = self._state + (action-act_max)*step_size    
        self._step_counter +=1
        
        ################# --- Compute R_{t+1}=R(S_t,A_t)
        R = compute_reward(self._state[0])
        
        #Set conditions for termination
        if self._step_counter>=sub_episode_length-1:
            self._episode_ended = True  #value for t+1

        #Now we are at the end of time t, when self._episode_ended may have changed
        if self._episode_ended:
            #if self._step_counter>100:
            #    reward += np.float32(-100)
            #ts.termination(observation,reward,outer_dims=None): Returns a TimeStep obj with step_type set to StepType.LAST.
            return ts.termination(np.array(self._state, dtype=np.float32), reward=R)
        else:
            #ts.transition(observation,reward,discount,outer_dims=None): Returns 
            #a TimeStep obj with step_type set to StepType.MID.
            return ts.transition(np.array(self._state, dtype=np.float32), reward=R, discount=disc_factor)



#Create a sequence of parallel environments and batch them, for later use by driver to generate parallel trajectories.
parallel_env = ParallelPyEnvironment(env_constructors=[Env]*env_num, 
                                     start_serially=False,
                                     blocking=False,
                                     flatten=False
                                    )
#Use the wrapper to create a TFEnvironments obj. (so that parallel computation is enabled)
train_env = tf_py_environment.TFPyEnvironment(parallel_env, check_dims=True) #instance of parallel environments
eval_env = tf_py_environment.TFPyEnvironment(Env(), check_dims=False) #instance
# train_env.batch_size: The batch size expected for the actions and observations.  
print('train_env.batch_size = parallel environment number = ', env_num)


#actor_distribution_network outputs a distribution
#it is a neural net which outputs the parameter (mean and sd, named as loc and scale) for a normal distribution
tf.random.set_seed(0)
actor_net = actor_distribution_network.ActorDistributionNetwork(   
                                         train_env.observation_spec(),
                                         train_env.action_spec(),
                                         fc_layer_params=(10,5), #Hidden layers
                                         seed=0, #seed used for Keras kernal initializers for NormalProjectionNetwork.
                                         discrete_projection_net=CategoricalProjectionNetwork,
                                         activation_fn = tf.math.tanh,
                                         #continuous_projection_net=(NormalProjectionNetwork)
                                         )


#Create the  REINFORCE_agent
train_step_counter = tf.Variable(0)
tf.random.set_seed(0)
REINFORCE_agent = reinforce_agent.ReinforceAgent(
        time_step_spec = train_env.time_step_spec(),
        action_spec = train_env.action_spec(),
        actor_network = actor_net,
        value_network = None,
        value_estimation_loss_coef = 0.2,
        optimizer = opt,
        advantage_fn = None,
        use_advantage_loss = False,
        gamma = 1.0, #discount factor for future returns
        normalize_returns = False, #The instruction says it's better to normalize
        gradient_clipping = None,
        entropy_regularization = None,
        train_step_counter = train_step_counter
        )
       
REINFORCE_agent.initialize()

#################
#replay_buffer is used to store policy exploration data
#################
replay_buffer = rb.TFUniformReplayBuffer(
                data_spec = REINFORCE_agent.collect_data_spec,  # describe spec for a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                batch_size = env_num,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                         # One batch corresponds to one parallel environment
                max_length = episode_length*100    # The maximum number of items that can be stored in a single batch segment of the buffer.     
                                                    # if exceeding this number previous trajectories will be dropped
)

#test_buffer is used for evaluating a policy
test_buffer = rb.TFUniformReplayBuffer(
                data_spec= REINFORCE_agent.collect_data_spec,  # describe a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                batch_size= 1,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                                    # train_env.batch_size: The batch size expected for the actions and observations.  
                                                    # batch_size: Batch dimension of tensors when adding to buffer. 
                max_length = episode_length         # The maximum number of items that can be stored in a single batch segment of the buffer.     
                                                    # if exceeding this number previous trajectories will be dropped
)

#A driver uses an agent to perform its policy in the environment.
#The trajectory is saved in replay_buffer
collect_driver = DynamicEpisodeDriver(
                                     env = train_env, #train_env contains parallel environments (no.: env_num)
                                     policy = REINFORCE_agent.collect_policy,
                                     observers = [replay_buffer.add_batch],
                                     num_episodes = sub_episode_num   #SUM_i (number of episodes to be performed in the ith parallel environment)
                                    )

#For policy evaluation
test_driver = py_driver.PyDriver(
                                     env = eval_env, #PyEnvironment or TFEnvironment class
                                     policy = REINFORCE_agent.policy,
                                     observers = [test_buffer.add_batch],
                                     max_episodes=1, #optional. If provided, the data generation ends whenever
                                                      #either max_steps or max_episodes is reached.
                                     max_steps=sub_episode_length
                                )

#Functions needed for training
def extract_episode(traj_batch,epi_length,attr_name = 'observation'):
    """
    This function extract episodes (each episode consists of consecutive time_steps) from a batch of trajectories.
    Inputs.
    -----------
    traj_batch:replay_buffer.gather_all(), a batch of trajectories
    epi_length:int, number of time_steps in each extracted episode
    attr_name:str, specify which data from traj_batch to extract
    
    Outputs.
    -----------
    tf.constant(new_attr,dtype=attr.dtype), shape = [new_batch_size, epi_length, state_dim]
                                         or shape = [new_batch_size, epi_length]
    """
    attr = getattr(traj_batch,attr_name)
    original_batch_dim = attr.shape[0]
    traj_length = attr.shape[1]
    epi_num = int(traj_length/epi_length) #number of episodes out of each trajectory
    batch_dim = int(original_batch_dim*epi_num) #new batch_dim
    
    if len(attr.shape)==3:
        stat_dim = attr.shape[2]
        new_attr = np.zeros([batch_dim, epi_length, state_dim])
    else:
        new_attr = np.zeros([batch_dim, epi_length])
        
    for i in range(original_batch_dim):
        for j in range(epi_num):
            new_attr[i*epi_num+j] = attr[i,j*epi_length:(j+1)*epi_length].numpy()
        
    return tf.constant(new_attr,dtype=attr.dtype)

# Please also see the metrics module for standard implementations of different
# metrics.

##### - Train REINFORCE_agent's actor_network multiple times.
update_num = generation_num 
eval_intv = 100 #number of updates required before each policy evaluation
REINFORCE_logs = [] #for logging the best objective value of the best solution among all the solutions used for one update of theta
final_reward = -1000
plot_intv = 500

tf.random.set_seed(0)
for n in range(0,update_num):
    #Generate Trajectories
    replay_buffer.clear()
    collect_driver.run()  #a batch of trajectories will be saved in replay_buffer
    
    experience = replay_buffer.gather_all() #get the batch of trajectories, shape=(batch_size, episode_length)
    rewards = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'reward') #shape=(sub_episode_num, sub_episode_length)
    observations = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)
    actions = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'action') #shape=(sub_episode_num, sub_episode_length, state_dim)
    step_types = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'step_type')
    discounts = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'discount')
    
    time_steps = ts.TimeStep(step_types,
                             tf.zeros_like(rewards),
                             tf.zeros_like(discounts),
                             observations
                            )
    
    rewards_sum = tf.reduce_sum(rewards, axis=1) #shape=(sub_episode_num,)
    
    with tf.GradientTape() as tape:
        #trainable parameters in the actor_network in REINFORCE_agent
        variables_to_train = REINFORCE_agent._actor_network.trainable_weights
    
        ###########Compute J_loss = -J
        actions_distribution = REINFORCE_agent.collect_policy.distribution(
                               time_steps, policy_state=None).action
    
        #log(pi(action|state)), shape = (batch_size, epsode_length)
        action_log_prob = common.log_probability(actions_distribution, 
                                                 actions,
                                                 REINFORCE_agent.action_spec)
    
        J = tf.reduce_sum(tf.reduce_sum(action_log_prob,axis=1)*rewards_sum)/sub_episode_num
        
        ###########Compute regularization loss from actor_net params
        regu_term = tf.reduce_sum(variables_to_train[0]**2)
        num = len(variables_to_train) #number of vectors in variables_to_train
        for i in range(1,num):
            regu_term += tf.reduce_sum(variables_to_train[i]**2)
        
        total = -J + param_alpha*regu_term
    
    #update parameters in the actor_network in the policy
    grads = tape.gradient(total, variables_to_train)
    grads_and_vars = list(zip(grads, variables_to_train))
    opt.apply_gradients(grads_and_vars=grads_and_vars)
    train_step_num += 1
    
    batch_rewards = rewards.numpy()
    batch_rewards[:,-1] = -np.power(10,8) #The initial reward is set as 0, we set it as this value to not affect the best_obs_index 
    best_step_reward = np.max(batch_rewards)
    best_step_index = [int(batch_rewards.argmax()/sub_episode_length),batch_rewards.argmax()%sub_episode_length+1]
    best_step = observations[best_step_index[0],best_step_index[1],:] #best solution
    #best_step_reward = f(best_solution)
    avg_step_reward = np.mean(batch_rewards[:,0:-1])
    REINFORCE_logs.append(best_step_reward)
    
    if best_step_reward>final_reward:
        #print("final reward before udpate:",final_reward)
        final_reward = best_step_reward
        final_solution = best_step.numpy()
        #print("final reward after udpate:",final_reward)
        #print('updated final_solution=', final_solution)
    
    #print(compute_reward(best_obs,alpha))
    if n%eval_intv==0:
        print("train_step no.=",train_step_num)
        print('best_solution of this generation=', best_step.numpy())
        print('best step reward=',best_step_reward.round(3),f(best_step.numpy()))
        print('avg step reward=', round(avg_step_reward,3))
        #print('act_logits:', actions_distribution._logits.numpy()[0] ) #second action mean
        print('best_step_index:',best_step_index)
        print('collect_traj:', observations[0,:,0].numpy())
        
        #Test trajectory generated by mode of action_distribution in agent policy
        test_buffer.clear()
        test_driver.run(eval_env.reset())  #generate batches of trajectories with agent.collect_policy, and save them in replay_buffer
        experience = test_buffer.gather_all()  #get all the stored items in replay_buffer
        test_trajectory = experience.observation.numpy()[0,:,0]
        print('test_traj', test_trajectory)
        
               
print('final_solution=',final_solution,
      'final_reward=',final_reward,
      )
REINFORCE_logs = [max(REINFORCE_logs[0:i]) for i in range(1,generation_num+1)] #rolling max


#Compare the trajectory generated by the trained policy,
#and the trajectory generated by the gradient descent method.

#Get the GD trajectory
def forward_fn(x_val):
    return -(x_val**2-1)**2 - 0.3*(x_val-1)**2+5

def f_grad(x_val):
    return -4*(x_val**2-1)*x_val - 0.6*(x_val-1)

#compute the function values
x_arr = np.linspace(-1.5,1.5,100)
f_vals = []
for xi in x_arr:
    f_vals.append(forward_fn(xi))

#compute the gradient descent trajectory
initial_x = -1.5
#step_size = 0.5
grad_trajectory = [[initial_x,forward_fn(initial_x)]]
step_num = sub_episode_length
current_x = initial_x
for stp in range(step_num):
    grad = np.sign(f_grad(current_x))*step_size
    current_x += grad
    grad_trajectory.append([current_x,forward_fn(current_x)])
    

########################################## - Figure 1
initial_state = [initial_x]
env_obj = Env()
env_obj.reset()
env_obj._state = initial_x
REINFORCE_trajectory = [[initial_x, forward_fn(initial_x)]]
timestep = ts.TimeStep(step_type=np.array([StepType.MID]), 
                                reward=np.float32([-forward_fn(initial_x)**2]),  
                                discount=np.array(disc_factor,dtype=np.float32), 
                                observation=np.array([initial_state], dtype=np.float32)
                      )
for i in range(step_num-1):
    act = REINFORCE_agent.collect_policy.action(timestep)[0].numpy()[0]
    timestep = env_obj.step(act)
    #convert the format of timestep so that agent.policy.action accepts it.
    timestep = ts.TimeStep(step_type=np.array([timestep.step_type]), 
                           reward=np.float32([timestep.reward]), 
                           discount=np.array(disc_factor,dtype=np.float32), 
                           observation=np.array([timestep.observation], dtype=np.float32)
                          )
    REINFORCE_trajectory.append([timestep.observation[0][0],forward_fn(timestep.observation[0][0])])

#plot sac trajectory
plt.figure(figsize=(8,6))
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'


i = 0
plt.arrow(x=REINFORCE_trajectory[i][0],
          y=REINFORCE_trajectory[i][1],
          dx=REINFORCE_trajectory[i+1][0]-REINFORCE_trajectory[i][0],
          dy=REINFORCE_trajectory[i+1][1]-REINFORCE_trajectory[i][1],
          color='r',width=0.02,label='REINFORCE-OPT')

for i in range(0, step_num-2):
    plt.arrow(x=REINFORCE_trajectory[i][0],
              y=REINFORCE_trajectory[i][1],
              dx=REINFORCE_trajectory[i+1][0]-REINFORCE_trajectory[i][0],
              dy=REINFORCE_trajectory[i+1][1]-REINFORCE_trajectory[i][1],
              color='r',width=0.02)

plt.plot(x_arr,f_vals,linestyle='--',label='$\mathcal{L}(x)=-(x^2-1)^2-0.3(x-1)^2+5$')
plt.tick_params(labelsize=20)
plt.xlabel('$x$',size=25)
plt.ylabel('$\mathcal{L}(x)$',size=25)
plt.legend(loc='lower right',fontsize=20)   
plt.savefig("escape1D-REINFROCE-traj.eps")
plt.show()


#plot gd trajectory
plt.figure(figsize=(8,6))
i = 0
plt.arrow(x=grad_trajectory[i][0],
          y=grad_trajectory[i][1],
          dx=grad_trajectory[i+1][0]-grad_trajectory[i][0],
          dy=grad_trajectory[i+1][1]-grad_trajectory[i][1],
          color='b',linestyle='--',width=0.02, label='Gradient Ascent')

for i in range(0, step_num-1):   
    plt.arrow(x=grad_trajectory[i][0],
              y=grad_trajectory[i][1],
              dx=grad_trajectory[i+1][0]-grad_trajectory[i][0],
              dy=grad_trajectory[i+1][1]-grad_trajectory[i][1],
              color='b',linestyle='--',width=0.015)

plt.plot(x_arr,f_vals,linestyle='--',label='$\mathcal{L}(x)=-(x^2-1)^2-0.3(x-1)^2+5$')
plt.tick_params(labelsize=20)
plt.xlabel('$x$',size=25)
plt.ylabel('$\mathcal{L}(x)$',size=25)
plt.legend(loc='lower right',fontsize=20)   
plt.savefig("escape1D-grad-traj.eps")
plt.show()


################################ - Figure 2a
REINFORCE_fitness_traj = []
grad_fitness_traj = []
for i in range(step_num):
    REINFORCE_fitness_traj.append(REINFORCE_trajectory[i][1])
    
for i in range(step_num):
    grad_fitness_traj.append(grad_trajectory[i][1])
    
plt.figure(figsize=(8,6))
plt.plot(range(1,step_num+1),REINFORCE_fitness_traj,color='red',marker='o',linestyle='--',label='REINFORCE-OPT')
plt.plot(range(1,step_num+1),grad_fitness_traj,color='blue',marker='x',linestyle='--',label='Gradient Ascent')
plt.xlabel('$t$',size=30)
plt.gca().set_xscale('log')
plt.xlim([0,40]) 
plt.ylabel('$\mathcal{L}(x_t)$',size=30)
plt.tick_params(labelsize=20)
plt.legend(loc='upper left',fontsize=20)
#plt.title('$\mathcal{L}(x_t)$ Trajectory - The 1D Case',size=25)
plt.savefig("escape-fitness-traj-1D.eps")
plt.show()
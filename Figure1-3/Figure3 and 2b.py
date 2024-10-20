#NOTE: The objective function in Figure 3 in the paper is denoted as \mathcal{L}(x), but in this code the objective is denoted as f(x).

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
#from custom_normal_projection_network import NormalProjectionNetwork
import os,gc
import pygad
from pyswarms.single.global_best import GlobalBestPSO
import time


#To limit TensorFlow to CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cpus = tf.config.experimental.list_physical_devices('CPU') 
tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
#enable multiprocessing for parallel computing
tf_agents.system.multiprocessing.enable_interactive_mode()
gc.collect()

###############
#Define the objective f to be maximized
N = 2   #This the dimension number
state_dim = N

############### -- The objective function. In Figure 3a, f(x) is denoted as \mathcal{L}(x).
def f(x):
    return -(2-np.sum(np.cos(10*x)) + 0.05*np.sum(100*x**2))+10
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
######################### - Plot the Objective Function in Figure 3
x1_min = -1.0
x2_min = -1.0
x1_max = 1.0
x2_max = 1.0
x_num = 100

X1 = np.linspace(x1_min,x1_max,x_num)
X2 = np.linspace(x2_min,x2_max,x_num)
X1, X2 = np.meshgrid(X1, X2)
Y = np.zeros((x_num,x_num))
for i in range(x_num):
    for j in range(x_num):
        Y[i,j] = f( x=np.array([X1[i,j],X2[i,j]]) )

ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
# Plot the 3D surface
#ax.title.set_text(r"$f(x)=-ln((x-m_1)^2+0.00001)-ln((x-m_2)^2+0.01)$")
ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm, edgecolor='royalblue')
ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max), zlim=(np.min(Y), np.max(Y)-1.5))
ax.set_xlabel('$x_1$', fontsize=25)
ax.set_ylabel('$x_2$', fontsize=25)
ax.tick_params(labelsize=20)
#ax.text(-1.2, -1, 16.6, "$\mathcal{L}(x_1,x_2)=8-\cos(10x_1)-\cos(10x_2)-5x_1^2-5x_2^2$",
#        color='black', size=20)
plt.savefig("escape-2D-objective.eps", bbox_inches='tight', transparent=True)
plt.show()


################
#Set initial x-value
r = np.random.RandomState(0)
x0_reinforce = np.array([0.5,-1.0])
sub_episode_length = 30 #number of time_steps in a sub-episode. 
episode_length = sub_episode_length*6  #an trajectory starts from the initial timestep and has no other initial timesteps
                                      #each trajectory will be split to multiple episodes
env_num = 10  #Number of parallel environments, each environment is used to generate an episode
print('x0', x0_reinforce)

################
#Set hyper-parameters for REINFORCE-OPT
generation_num = 16000  #number of theta updates for REINFORCE-IP, also serves as the number
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
lr = lr_schedule(initial_lr=0.00002, C=50000)   
opt = tf.keras.optimizers.SGD(learning_rate=lr)
#opt = tf.keras.optimizers.Adam( )

train_step_num = 0

act_min = -1
act_max = 1
step_size = 0.05
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
        self._state = np.array(x0_reinforce,dtype=np.float32)
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
        self._state = np.array(x0_reinforce,dtype=np.float32)  #initial state
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
        normalized_act = (action-act_max)/np.sqrt((action-act_max)**2+0.0000001) 
        self._state = self._state + normalized_act*step_size    
        self._step_counter +=1
        
        ################# --- Compute R_{t+1}=R(S_t,A_t)
        R = compute_reward(self._state)
        
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
                                         fc_layer_params=(16,16,16), #Hidden layers
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

######## Train REINFORCE_agent's actor_network multiple times.
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
        #print('episode of rewards', rewards.round(3))
        #print('act_std:', actions_distribution.stddev()[0,0]  )
        #print('act_mean:', actions_distribution.mean()[0,0] ) #second action mean
        print('best_step_index:',best_step_index)
        print(observations[0])
        print(' ')
        
    if n%plot_intv==0:
        test_buffer.clear()
        test_driver.run(eval_env.reset())  #generate batches of trajectories with agent.collect_policy, and save them in replay_buffer
        experience = test_buffer.gather_all()  #get all the stored items in replay_buffer
        rl_trajectory = experience.observation.numpy()[0]
        
        #Plot
        fig, ax = plt.subplots(figsize=(12,12))
        CS = ax.contour(X1, X2, Y)
        
        i = 0
        plt.arrow(x=rl_trajectory[i][0],
          y=rl_trajectory[i][1],
          dx=rl_trajectory[i+1][0]-rl_trajectory[i][0],
          dy=rl_trajectory[i+1][1]-rl_trajectory[i][1],
          color='r',linestyle='--',width=0.01, label='REINFORCE-OPT')

        for i in range(0, len(rl_trajectory)-1):   
            plt.arrow(x=rl_trajectory[i][0],
              y=rl_trajectory[i][1],
              dx=rl_trajectory[i+1][0]-rl_trajectory[i][0],
              dy=rl_trajectory[i+1][1]-rl_trajectory[i][1],
              color='r',linestyle='--',width=0.01)
        
        #plt.plot(x_arr,f_vals,linestyle='--',label='$f(x)=(x^2-1)^2+0.3(x-1)^2$')
        plt.tick_params(size=16)
        plt.xlabel('x',size=20)
        plt.ylabel('f(x)',size=20)
        plt.legend(loc='upper right',fontsize=20)   
        plt.show()
        
print('final_solution=',final_solution,
      'final_reward=',final_reward,
      )
REINFORCE_logs = [max(REINFORCE_logs[0:i]) for i in range(1,generation_num+1)] #rolling max


############################################# - Second Part of Figure 3
############# - Create a trajectory with REINFORCE_agent.policy (which select action according to the mode of action distribution)
test_buffer.clear()
test_driver.run(eval_env.reset())  #generate batches of trajectories with agent.collect_policy, and save them in replay_buffer
experience = test_buffer.gather_all()  #get all the stored items in replay_buffer
rl_trajectory = experience.observation.numpy()[0]

############# - Create the Gradient Ascent Trajectory
ga_trajectory = [x0_reinforce]
step_num = 30
current_x = tf.Variable([0.5,-1])

for i in range(step_num):
    with tf.GradientTape() as tape:
        y = -(2-tf.reduce_sum(tf.math.cos(10*current_x)) + 0.05*tf.reduce_sum(100*current_x**2))+10
    gradient = tape.gradient(y, current_x)
    norm_gradient = gradient.numpy()/np.sqrt(np.sum(gradient.numpy()**2))
    current_x.assign(current_x.numpy() + norm_gradient*step_size)
    ga_trajectory.append(current_x.numpy())

############# - Plot Trajectories
fig, ax = plt.subplots(figsize=(8,8))
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'

#Contour Plot for the objective function
CS = ax.contour(X1, X2, Y)
ax.clabel(CS, inline=True, fontsize=20,fmt='%1.1f',
         colors=('r', 'red', 'blue', (1, 1, 0), '#afeeee', '0.5')
         )

#Plot the REINFORCE-OPT trajectory
i = 0
plt.arrow(x=rl_trajectory[i][0],
          y=rl_trajectory[i][1],
          dx=rl_trajectory[i+1][0]-rl_trajectory[i][0],
          dy=rl_trajectory[i+1][1]-rl_trajectory[i][1],
          color='r',linestyle='--',width=0.008, label='REINFORCE-OPT')

for i in range(0, len(rl_trajectory)-1):   
    plt.arrow(x=rl_trajectory[i][0],
              y=rl_trajectory[i][1],
              dx=rl_trajectory[i+1][0]-rl_trajectory[i][0],
              dy=rl_trajectory[i+1][1]-rl_trajectory[i][1],
              color='r',linestyle='--',width=0.008)

#Plot the gradient descent trajectory
i = 0
plt.arrow(x=ga_trajectory[i][0],
          y=ga_trajectory[i][1],
          dx=ga_trajectory[i+1][0]-ga_trajectory[i][0],
          dy=ga_trajectory[i+1][1]-ga_trajectory[i][1],
          color='b',linestyle='--',width=0.006, label='Gradient Ascent')

for i in range(0, len(ga_trajectory)-1):   
    plt.arrow(x=ga_trajectory[i][0],
              y=ga_trajectory[i][1],
              dx=ga_trajectory[i+1][0]-ga_trajectory[i][0],
              dy=ga_trajectory[i+1][1]-ga_trajectory[i][1],
              color='b',linestyle='--',width=0.006)

plt.plot(0,0,marker='o',color='black',markersize=12) 
plt.text(0.001, 0.05, s='Global Max',size=20)
plt.tick_params(size=8)
plt.xlabel('$x_1$',size=25)
plt.ylabel('$x_2$',size=25)
plt.legend(loc='upper right',fontsize=20)   
#plt.title(r"Trajectory in the $\mathbf{x}$-space",size=15)
plt.savefig("escape-2D-traj1.eps", bbox_inches='tight', transparent=True)
plt.show()



########################################################### - Figure 2b
rl_fitness_traj = []
ga_fitness_traj = []
for i in range(step_num):
    rl_fitness_traj.append(f(rl_trajectory[i]))
    
for i in range(step_num):
    ga_fitness_traj.append(f(ga_trajectory[i]))
    
plt.figure(figsize=(8,6))
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'

plt.plot(range(1,step_num+1),rl_fitness_traj,color='red',marker='o',linestyle='--',label='REINFORCE-OPT')
plt.plot(range(1,step_num+1),ga_fitness_traj,color='blue',marker='x',linestyle='--',label='Gradient Ascent')
plt.xlabel('$t$',size=28)
plt.gca().set_xscale('log')
plt.xlim([0,40]) 
plt.ylabel('$\mathcal{L}(x_t)$',size=28)
plt.tick_params(labelsize=20)
plt.legend(loc='upper left',fontsize=20)
#plt.title('$\mathcal{L}(x_t)$ Trajectory - The 2D Case',size=25)
plt.savefig("fitness-traj-2D.eps")
plt.show()

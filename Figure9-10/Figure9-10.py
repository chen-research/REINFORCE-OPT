#import driver
from tf_agents.drivers import py_driver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from sklearn.metrics import r2_score

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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as ax
import numpy as np
import tensorflow as tf
import tf_agents
#from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from custom_actor_distribution_network import ActorDistributionNetwork
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from custom_normal_projection_network import NormalProjectionNetwork
import os,gc

#To limit TensorFlow to a specific set of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cpus = tf.config.experimental.list_physical_devices('CPU') 
tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
#enable multiprocessing for parallel computing
tf_agents.system.multiprocessing.enable_interactive_mode()
gc.collect()

################
#Set hyper-parameters
disc_factor = 1.0
sub_episode_length = 10 #number of time_steps in a sub-episode. 
episode_length = sub_episode_length*6  #an trajectory starts from the initial timestep and has no other initial timesteps
                                      #each trajectory will be split to multiple episodes
env_num = 10  #Number of parallel environments, each environment is used to generate an episode
alpha = 0.1 #regularization coefficient
param_alpha = 0.0001 #regularization coefficient for actor_network
sub_episode_num = int(env_num*(episode_length/sub_episode_length)) #number of sub-episodes used for a single update of actor_net params
print("number of sub_episodes used for a single param update:", sub_episode_num)

#Learning Schedule = initial_lr * (C/(step+C))
class lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, C):
        self.initial_learning_rate = initial_lr
        self.C = C
    def __call__(self, step):
        return self.initial_learning_rate*self.C/(self.C+step)
lr = lr_schedule(initial_lr=0.0001, C=100000)   
opt = tf.keras.optimizers.legacy.SGD(learning_rate=lr)
train_step_num = 0

##################Generate the noised ye (ye_n)
N = 64
state_dim = N
h = 1/(N-1)
t = np.linspace(0,1,N)
nn = 1 #number of noised y's

#Create the foward model
def forward_func(x):
    """
    x:np.1darray.
    return: np.1darray
    """
    N = len(x)
    h = 1/(N-1)
    y = np.zeros((N+1),dtype=np.float32)
    x1 = np.append([0.0],x)
    for j in range(2,N+1):
        z = 0
        for i in range(1,j):
            z = z + 0.5*h*( x1[j-i]*x1[i+1]+x1[j-i+1]*x1[i] )
        y[j] = z
    return y[1:]

#Exact solution
xe = np.zeros(N)
for i in range(0, N):
    #xe[i] = 100 * (t[i]**2) * (1-t[i])**5
    xe[i] = 10 * t[i] * (1-t[i])**2

ye = forward_func(xe)  # exact data

#Create noised ye
noise = np.random.normal(loc=0.0,scale=0.05*ye,size=(nn,len(ye)))
ye_n = ye + noise
ye_n.shape

k = np.random.choice(range(nn))
plt.figure()
plt.plot(ye,color='red',label='true ye')
plt.plot(ye_n[k],color='blue',label='ye_n:{0}'.format(i))
plt.legend(loc='upper left')
plt.ylabel('ye')
plt.title("ye_n[{0}]".format(k))
plt.show()

def compute_reward(x, coeff):
    regularization = np.abs(x[0]) + np.abs(x[-1])
    R = np.sum(  ( forward_func(x)-ye_n )**2  )/nn
    R = np.float32(1/(R+coeff*regularization+0.001)) #R_{t+1} = |F(S_{t+1})-y0|^2
    return R
    

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
                            shape=(state_dim,), dtype=np.float32, minimum=-0.1, maximum=0.1, name='action') #a_t is an 2darray
    
        #Specify the format requirement for observation (It is a 2d-array for this case), 
        #i.e. the observable part of S_t, and it is stored in self._state
        self._observation_spec = array_spec.BoundedArraySpec(
                                 shape=(state_dim,), dtype=np.float32, name='observation') #default max and min is None
        #self._state = np.array([0.0]*state_dim,dtype=np.float32)
        rand = np.random.choice([0,1])
        self._state = np.float32(rand*xe*0.75-(1-rand)*xe*0.75)
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
        #self._state = np.array(np.random.normal(loc=0.0, scale=0.1, size=state_dim),
        #                       dtype=np.float32)  #initial state
        #self._state = -np.array(np.ones(state_dim)*0.1,dtype=np.float32)
        rand = np.random.choice([0,1])
        self._state = np.float32(rand*xe*0.75-(1-rand)*xe*0.75)
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
    
    
    def fwmodel(self,input_val):
        """
        same as forward_fcn.
        x:np.1darray.
        return: np.1darray
        """
        dim = len(input_val)
        h = 1/(dim-1)
        y = np.zeros((dim+1),dtype=np.float32)
        x1 = np.append([0.0],input_val)
        for j in range(2,dim+1):
            z = 0
            for i in range(1,j):
                z = z + 0.5*h*( x1[j-i]*x1[i+1]+x1[j-i+1]*x1[i] )
            y[j] = z
        return y[1:]
        
    
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
        self._state = self._state + action      
        self._step_counter +=1
        
        ################# --- Compute R_{t+1}=R(S_t,A_t)
        R = compute_reward(self._state,coeff=alpha)
        
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
eval_env = tf_py_environment.TFPyEnvironment(Env(), check_dims=False) 

# train_env.batch_size: The batch size expected for the actions and observations.  
print('train_env.batch_size = parallel environment number = ', env_num)


#actor_distribution_network outputs a distribution
#it is a neural net which outputs the parameter (mean and sd, named as loc and scale) for a normal distribution
#The NormalProjectionNetwork produces a covariance that is independent from the state, set by state_dependent_std=False in the source code.
actor_net = ActorDistributionNetwork(   
                                    train_env.observation_spec(),
                                    train_env.action_spec(),
                                    fc_layer_params=(128,128,128), #Hidden layers
                                    seed=0, #seed used for Keras kernal initializers for NormalProjectionNetwork.
                                         #discrete_projection_net=_categorical_projection_net
                                    continuous_projection_net=(NormalProjectionNetwork) #in the source code: state_dependent_std=False, which implies the covariance matrix is irrelevant to state    
                                    )

#Create the  REINFORCE_agent
train_step_counter = tf.Variable(0)

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


#@test {"skip": true}
def compute_avg_return(environment, policy, num_episodes=10, max_step=100):
    '''
    This function perform 'policy' in 'environment' for a few episodes and 
    compute the average of sum-of-total-rewards.
    '''
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        step_count = 0

        while ((not time_step.is_last()) & (step_count<max_step)) :
            action_step = policy.action(time_step)  #output an action based on time_step.observation (S_t)
            time_step = environment.step(action_step.action)  #take the action and output the next state and reward
            episode_return += time_step.reward  #update the total rewards
            step_count += 1
        total_return += episode_return

    avg_return = total_return / num_episodes
    #return avg_return.numpy()[0]
    return avg_return
    

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


def evaluate_policy(sub_episode_num, 
                    sub_episode_length, 
                    x_true, 
                    y_true, 
                    alpha,
                    episode_num=200,
                    ):
    #This function generate episodes by REINFORCE.collect_policy, and compute mean state in the last step
    #sub_episode_num is the number of episodes collect_driver.run() produces.
    #traj_num = 10000
    loop_num = int(episode_num/sub_episode_num)
    residual_num = episode_num%sub_episode_num
    total_obs = np.zeros((episode_num,state_dim),dtype=np.float32)

    for n in range(0,loop_num):
        #Generate 60 Trajectories
        replay_buffer.clear()
        collect_driver.run()  #a batch of trajectories will be saved in replay_buffer
        experience = replay_buffer.gather_all() #get the batch of trajectories, shape=(batch_size, episode_length)
        observations = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)=(60,10,64)
        total_obs[n*sub_episode_num:(n+1)*sub_episode_num] = observations.numpy()[:,-1,:]

    replay_buffer.clear()
    collect_driver.run()
    observations = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)=(60,10,64)
    total_obs[-residual_num:] = observations[0:residual_num,-1,:]
    
    #Cluster the estimates to two groups, and take the mean of each group as xe estimates
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(total_obs)
    gp1_samples = total_obs[np.where(kmeans.labels_==1)[0]]
    gp2_samples = total_obs[np.where(kmeans.labels_==0)[0]]
    xe_est1 = np.mean(gp1_samples,axis=0)
    xe_est2 = np.mean(gp2_samples,axis=0)

    ye_est1 = forward_func(xe_est1)
    ye_est2 = forward_func(xe_est2)

    num1 = np.sum(kmeans.labels_==1)
    num2 = np.sum(kmeans.labels_==0)
    
    
    print('gp1:',
          'xe R2=', r2_score(x_true,xe_est1).round(3),
          'neg_xe R2=', r2_score(x_true*(-1.0),xe_est1).round(3),
          'ye R2=', r2_score(y_true,ye_est1).round(3))
    
    print('gp2:',
          'xe R2=', r2_score(x_true,xe_est2).round(3),
          'neg_xe R2=', r2_score(x_true*(-1.0),xe_est2).round(3),
          'ye R2=', r2_score(y_true,ye_est2).round(3))

    plt.figure()
    plt.plot(x_true,color='red',label='true xe')
    plt.plot(-x_true,color='red')
    plt.plot(xe_est1,color='blue',label='estimated xe1')
    plt.plot(xe_est2,color='blue',label='estimated xe2')
    plt.legend(loc='upper left')
    plt.show()
            
    plt.figure()
    plt.plot(y_true,color='red',label='true ye')
    plt.plot(ye_est1,color='blue',label='estimated ye1')
    plt.plot(ye_est2,color='blue',label='estimated ye2')

    plt.legend(loc='upper left')
    plt.ylabel('ye')
    plt.show()
        
    performance = compute_reward(xe_est1,coeff=alpha)+compute_reward(xe_est2,coeff=alpha)
    print('train_step_no. = {0}: Avg_traj_R = {1}'.format(train_step_num, performance))
    
    return performance


######## Train REINFORCE_agent's actor_network multiple times.

update_num = 7500
eval_intv = 100 #number of updates required before each policy evaluation
eval_results = [] #for logging evaluation results
#num_eval_episodes = 10
max_reward = -1000.0
tolerence = 20 #If the eval_results does not improve within tolerence*eval_intv, then stop training.

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
    
    
    #Evaluate policy performance and log result
    if train_step_num % eval_intv == 0:
        print("train_step no.=",train_step_num)
        print('act_std:', actions_distribution.stddev()[0,0]  )
        eval_performance = evaluate_policy(sub_episode_num,
                                   sub_episode_length, 
                                   x_true=xe,
                                   y_true=ye,
                                   alpha=alpha,
                                   episode_num=1000)
        eval_results.append(eval_performance)
        
        #Stopping Rule
        if len(eval_results)>=tolerence:
            if eval_performance >20.0:
                break
            range_max = np.max(eval_results[-tolerence:])
            if range_max<=eval_results[-tolerence]:
                print('Stopping Step Num:', train_step_num)
                break
        

#Plot training history
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.bf'] = 'cm:italic:bold'

steps = range(0, train_step_num, eval_intv)
plt.figure(figsize=(12,6))
plt.plot(steps, eval_results)
plt.title('Performance of the Evolved Policy',size=15)
plt.ylabel('Sum of Rewards in a Generated Trajectory',size=15)
plt.xlabel("Number of Times " + r'$\mathbf{\theta}$'+ " Gets Updated",size=15)
plt.tick_params(labelsize=15)
plt.savefig("example1-2log.eps",bbox_inches='tight')


############Generate 10000 Trajectories 
traj_num = 10000
loop_num = int(traj_num/60)
residual_num = traj_num%60
estimates = np.zeros((traj_num,state_dim),dtype=np.float32)

for n in range(0,loop_num):
    #Generate 60 Trajectories
    replay_buffer.clear()
    collect_driver.run()  #a batch of trajectories will be saved in replay_buffer
    experience = replay_buffer.gather_all() #get the batch of trajectories, shape=(batch_size, episode_length)
    observations = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)=(60,10,64)
    estimates[n*60:(n+1)*60] = observations.numpy()[:,-1,:]

replay_buffer.clear()
collect_driver.run()
observations = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)=(60,10,64)
estimates[-residual_num:] = observations[0:residual_num,-1,:]
print(np.min(np.sum(np.abs(estimates),axis=1)))


#Cluster the estimates to two groups, and take the mean of each group as xe estimates
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(estimates)
gp1_samples = estimates[np.where(kmeans.labels_==1)[0]]
gp2_samples = estimates[np.where(kmeans.labels_==0)[0]]
#xe_est1 = kmeans.cluster_centers_[0]
xe_est1 = np.mean(gp1_samples,axis=0)
xe_est2 = np.mean(gp2_samples,axis=0)
xe_sd1 = np.std(gp1_samples,axis=0)
xe_sd2 = np.std(gp2_samples,axis=0)

ye_est1 = forward_func(xe_est1)
ye_est2 = forward_func(xe_est2)

num1 = np.sum(kmeans.labels_==1)
num2 = np.sum(kmeans.labels_==0)

#Bootstrapping Confidence Interval (CI) for Group1 Samples
B = 100000  #number of bootstrapping times 
bootstrap_mean_list = []
index = range(num1)
for i in range(B):
    bootstrap_index = np.random.choice(a=index,size=5,replace=True)
    bootstrap_sample = gp1_samples[bootstrap_index]
    bootstrap_mean_list.append(np.mean(bootstrap_sample,axis=0))

gp1CI_lb = np.percentile(a=bootstrap_mean_list,q=0.5,axis=0) #CI lower bound for grp1
gp1CI_up = np.percentile(a=bootstrap_mean_list,q=99.5,axis=0) #CI upper bound for grp1

#gp1CI_lb = np.min(bootstrap_mean_list,axis=0) #CI lower bound for grp1
#gp1CI_up = np.max(bootstrap_mean_list,axis=0) #CI upper bound for grp1

#Bootstrapping Confidence Interval (CI) for Group2 Samples
B = 100000  #number of bootstrapping times 
bootstrap_mean_list = []
index = range(num2)
for i in range(B):
    bootstrap_index = np.random.choice(a=index,size=5,replace=True)
    bootstrap_sample = gp2_samples[bootstrap_index]
    bootstrap_mean_list.append(np.mean(bootstrap_sample,axis=0))

gp2CI_lb = np.percentile(a=bootstrap_mean_list,q=0.5,axis=0) #CI lower bound for grp1
gp2CI_up = np.percentile(a=bootstrap_mean_list,q=99.5,axis=0) #CI upper bound for grp1

#gp2CI_lb = np.min(bootstrap_mean_list,axis=0) #CI lower bound for grp2
#gp2CI_up = np.max(bootstrap_mean_list,axis=0) #CI upper bound for grp2
gp2CI_up-gp2CI_lb

#Plot
print('xe est1')
xe_R2 = r2_score(xe,xe_est1).round(3)
neg_xe_R2 = r2_score(xe*(-1.0),xe_est1).round(3)
ye_R2 = r2_score(ye,ye_est1).round(3)
print('xe R2=', xe_R2, 'neg_xe R2=', neg_xe_R2, 'ye R2=',ye_R2)

print('xe est2')
xe_R2 = r2_score(xe,xe_est2).round(3)
neg_xe_R2 = r2_score(xe*(-1.0),xe_est2).round(3)
ye_R2 = r2_score(ye,ye_est2).round(3)
print('xe R2=', xe_R2, 'neg_xe R2=', neg_xe_R2, 'ye R2=', ye_R2)

f, ax = plt.subplots(2,1,figsize=(12,8))
ax[0].plot(t,xe,color='purple',label=r"$\mathbf{x}_e$")
ax[0].plot(t,-xe,color='red',label=r"$-\mathbf{x}_e$")
#ax[0].plot(xe_est1,color='green',marker='o',linestyle='--',markersize=4.5,label=r"$\bar{\mathbf{x}}_1$")
#ax[0].plot(xe_est2,color='blue',marker='x',linestyle='--',markersize=5.5,label=r"$\bar{\mathbf{x}}_2$")
ax[0].plot(t,gp1CI_lb,color='green',linestyle='--',label='99%-CI of grp1')
ax[0].plot(t,gp1CI_up,color='green',linestyle='--',label='99%-CI of grp1')
ax[0].plot(t,gp2CI_lb,color='blue',linestyle='--',label='99%-CI of grp2')
ax[0].plot(t,gp2CI_up,color='blue',linestyle='--',label='99%-CI of grp2')
ax[0].fill_between(x=t, y1=gp1CI_lb, y2=gp1CI_up, color='green', alpha=0.1)
ax[0].fill_between(x=t, y1=gp2CI_lb, y2=gp2CI_up, color='blue', alpha=0.1)



ax[0].legend(loc='upper right',fontsize=13.5)
ax[0].tick_params(labelsize=15)
ax[0].set_xlim(-0.02,1.33)
#plt.savefig("example1-xe.eps")
#plt.show()

ax[1].plot(t,ye,color='red',label='true ye')
ax[1].plot(t,ye_est1,color='blue',marker='o',linestyle='--', markersize=4.5, label=r"$f(\bar{\bf{x}}_1)$")
ax[1].plot(t,ye_est2,color='green',marker='x',linestyle='--', markersize=6.5, label=r'$f(\bar{\bf{x}}_2)$')
ax[1].legend(loc='lower right',fontsize=15)
ax[1].tick_params(labelsize=15)
ax[1].set_xlim(-0.02,1.33)
plt.savefig("example1-2.svg",transparent=True,bbox_inches='tight')
plt.show()

#Convert svg to eps
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

drawing = svg2rlg("example1-2.svg")
renderPDF.drawToFile(drawing, "example1-2.eps")
#renderPM.drawToFile(drawing, "file.png", fmt="PNG")
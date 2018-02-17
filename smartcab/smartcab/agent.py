import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.ep = epsilon

    def reset(self, destination=None, testing=False, trial=1):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing: 
            self.alpha, self.epsilon, self.ep = 0.0, 0.0, 0.0 
        else: 
            # 0. default
            #self.epsilon -= 0.05
            # 1. basic agent, linear decrease in epsilon
            #if trial > 1: self.epsilon -= 0.05
            # 2. exponential
            #if trial > 1: 
            #    self.epsilon = math.exp(-(trial-1)*0.1)
            # 3. power (1/t^2)
            #if trial > 1:
            #    self.epsilon = 1.0/(trial**0.3)
            # 4. quadratic - graph decay slower at first
            if trial > 1: 
                a = -1.0/(300**2)
                self.epsilon = 1 + a * (trial**2)
        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent 
        '''
        if inputs['oncoming'] == 'right' or inputs['oncoming'] == 'forward':
            on = 'Care'
        else: 
            on = 'NotCare'
        
        if inputs['left'] == 'forward': 
            left = 'Care'
        else: 
            left = 'NotCare'
        
        # state is waypoint, light, oncoming, left
        #state = (waypoint,inputs['light'],inputs['oncoming'],inputs['left']) # None # (original value)
        state = (waypoint, inputs['light'],on,left)
        
        # new 2018-02-15
        if inputs['light'] == 'green': left = 'NotCare'
        
        state = (waypoint, inputs['light'],on,left)
        '''
        state = (waypoint,inputs['light'],inputs['oncoming'],inputs['left'])
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        maxQ = None
        best = []
        for act in self.valid_actions:
            if maxQ is None: 
                maxQ = self.Q[state][act]
                best.append(act)
            else:
                if self.Q[state][act] > maxQ:
                    maxQ = self.Q[state][act]
                    best = []
                    best.append(act)
                elif self.Q[state][act] == maxQ:
                    best.append(act)
        if len(best) > 1:
            best = random.choice(best)
        elif len(best) == 1:
            best = best[0]      
        return(maxQ,best)


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if self.learning:
            if state in self.Q.keys():
                pass
            else:
                self.Q[state] = {}
                for act in self.valid_actions:
                    self.Q[state][act] = 0.0
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        
        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
       
        if not self.learning:
            action = random.choice(['left','right','forward',None]) # uniformly choose one sample out of the list
        else:
            if random.uniform(0.0,1.0) < self.epsilon:
                action = random.choice(['left','right','forward',None]) # uniformly choose one sample out of the list
            else:
                action = self.get_maxQ(state)[1] # make use of get_maxQ to find action which maximize Q values
        
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        # Q = Q(1-alpha) + alpha (r+maxQ-value), use function get_maxQ
        if self.learning:
            self.Q[state][action] = self.Q[state][action]*(1-self.alpha) + \
                                                            self.alpha*reward
                                #self.alpha*(reward+self.get_maxQ(state)[0])
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent,learning=True)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent,enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env,update_delay=0.01,log_metrics=True,optimized=True,display=False)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    
    # 0. default
    #sim.run(n_test=10)
    # 1. basic linear
    #sim.run(n_test=50,tolerance=1-49*(0.05))
    # 2. exponential
    #sim.run(n_test=50,tolerance=math.exp(-48*0.1))
    # 3. Power (1/t^2)
    #sim.run(n_test=50,tolerance=1.0/(49**0.3))
    # 4. cubic - graph decay slower at first
    a = -1.0/(300**2)
    sim.run(n_test=50,tolerance=(1+a*(299**2)))
    print(len(agent.Q.keys()))
    return(agent.Q)

if __name__ == '__main__':
    run()

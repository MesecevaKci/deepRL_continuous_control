# Continuous Control
## Project 2 of Udacity Deep Reinforcement Learning Nanodegree

The goal of this project is to train an reinforcement learning agent - a double-jointed arm - to move to target locations in the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)


```
Unity brain name: ReacherBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 33
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
```

As it can be seen above, the observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Solving the Environment

The project can be solved in two different versions of the environment.

Option 1: Solving the First Version

- This version contains a single agent.
The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

Option 2: Solving the Second Version

- The second version contains 20 identical agents, each with its own copy of the environment. 
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically, after each episode, the rewards that each agent received (without discounting) are added up, to get a score for each agent. This yields 20 (potentially different) scores. The final score is the average of these 20 scores. 

The environment is considered solved, when the average over 100 episodes of those average scores is at least +30. 

### Dependencies

The project uses 
- Python 3.6 
- PyTorch v0.4 
- and the Unity Reacher environment. 

The Python environment can be set up following the [Udacity instructions](https://github.com/udacity/deep-reinforcement-learning)
in the Dependencies section. Depending on the operating system, the Unity environment can be downloaded from:

Download the environment from one of the links below. You need only select the environment that matches your operating system:

   - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

### Solution

The problem is solved using the DDPG algorithm for the case of a sinle agent. The code is stored in the notebook [Continuous_Control.ipynb](Continuous_Control.ipynb) and the description of the solution is described in the [report.md](report.md) document.

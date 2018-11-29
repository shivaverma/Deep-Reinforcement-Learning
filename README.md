[//]: # (Image References)

[image1]: p1_navigation/images/banana.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, I am training an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

- A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

- The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

- The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

[//]: # (Image References)

[image2]: p2_continuous_control/images/joint_arm.gif "Trained Agent"

# Project 2: Continuous Control

## Introduction

For this project, I am training a double-jointed arm to move to the target locations.

![Trained Agent][image2]

- A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
- Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
- The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.
- Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, there are two separate versions of the Unity environment:

* The first version contains a single agent.
* The second version contains 20 identical agents, each with its own copy of the environment.

[//]: # (Image References)

[image3]: p3_collab-competition/images/tennis.gif "Trained Agent"

# Project 3: Multi Agent Control

## Introduction:

In this environment, I am training two agents to control rackets to bounce a ball over a net.

![Trained Agent][image3]

- If an agent hits the ball over the net, it receives a reward of +0.1.
- If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.
- Thus, the goal of each agent is to keep the ball in play.
- The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
- The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode.
- The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# Project 4: Quadcopter Controller

## Introduction:

- In this project, I trained a Quadcopter to fly in real world like environment.
- I have created my own reward function in this project.
- There are 4 numbers in the action space in this environment, which corresponds to rotor speed of all the 4 rotors.


from ratinabox.Agent import Agent
from ratinabox.contribs.TaskEnvironment import (TaskEnvironment, 
                                                Goal, Reward,
                                                get_goal_vector, 
                                                SpatialGoal,
                                                SpatialGoalEnvironment,
                                                test_environment_loop)

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
plt.ion()

#    _  _     ____              _   _       _ 
#  _| || |_  / ___| _ __   __ _| |_(_) __ _| |
# |_  ..  _| \___ \| '_ \ / _` | __| |/ _` | |
# |_      _|  ___) | |_) | (_| | |_| | (_| | |
#   |_||_|   |____/| .__/ \__,_|\__|_|\__,_|_|
#                  |_|                        
#  _____                           _      
# | ____|_  ____ _ _ __ ___  _ __ | | ___ 
# |  _| \ \/ / _` | '_ ` _ \| '_ \| |/ _ \
# | |___ >  < (_| | | | | | | |_) | |  __/
# |_____/_/\_\__,_|_| |_| |_| .__/|_|\___|
#                           |_|           

def create_spatial_environment():
    # SECTION: ENVIRONMENT
    # Specify the number of corners for the polygon (8 for an octagon)
    num_corners = 8
    # Create the coordinates for the octagon
    octagon_coords = oc = [[0.5 + 0.5*np.cos(2*np.pi*i/num_corners + np.pi/8), 
                            0.5 + 0.5*np.sin(2*np.pi*i/num_corners + np.pi/8)] 
                            for i in range(num_corners)]

    env = SpatialGoalEnvironment(params = {'boundary': octagon_coords})
    env.plot_environment()

    # SECTION: Goals
    # Make some goals
    midpoints = [np.mean([oc[i], oc[(i+1)%num_corners]], axis=0) 
                 for i in range(num_corners)]
    goals = [SpatialGoal(env, pos=np.array([x, y]))
             for x,y in midpoints]

    # SECTION: Agents
    # Make some goals
    env.goal_cache.reset_goals = goals # the pool of goals to sample from
    env.goal_cache.reset_n_goals = 5 # the number of goals to sample
    # Make some agents
    [env.add_agents(Agent(env)) for _ in range(3)]

    return env

env = create_spatial_environment()
test_environment_loop(env, episodes=10)

#    _  _      ____          _                                      _ 
#  _| || |_   / ___|   _ ___| |_ ___  _ __ ___     __ _  ___   __ _| |
# |_  ..  _| | |  | | | / __| __/ _ \| '_ ` _ \   / _` |/ _ \ / _` | |
# |_      _| | |__| |_| \__ \ || (_) | | | | | | | (_| | (_) | (_| | |
#   |_||_|    \____\__,_|___/\__\___/|_| |_| |_|  \__, |\___/ \__,_|_|
#                                                 |___/               

class SectorGoal(Goal):
    """
    This class defines a goal for the n-agonal task

    In essence, each wall side is the base of a triagle that extends
    to the center of the octagon. The goal is to step into the sector
    defined by the base of the current wall (where an LED is located)
    or for greater reward, step into the neighoring wall's sector
    """
    def __init__(self, env, ith_goal, 
                 num_corners=8, 
                 minor_reward_dir="left",
                 major_reward = Reward(1, 0.01, expire_clock=1),
                 minor_reward = Reward(1, 0.01, expire_clock=1),
                 center_point=[0.5, 0.5], **kws): 
        """
        """
        self.env = env
        super().__init__(env, **kws) # initialize the goal

        octagon_coords = oc = [[0.5 + 0.5*np.cos(2*np.pi*i/num_corners + np.pi/8), 
                                0.5 + 0.5*np.sin(2*np.pi*i/num_corners + np.pi/8)] 
                                for i in range(num_corners)]
        # midpoints = [np.mean([oc[i], oc[(i+1)%num_corners]], axis=0) 
        #              for i in range(num_corners)]
        self.line_segments = [[oc[i], oc[(i+1)%num_corners]] for i in range(num_corners)]

        #  🎯 GOAL ZONE
        # goal sector is defined by the ith line segment and the center point
        self.goal_sector = [*self.line_segments[ith_goal], center_point]
        # minor goal sector is defined by the (i +/- 1)th line segment and the center point
        if minor_reward_dir  == "left":
            self.minor_goal_sector = [*self.line_segments[(ith_goal-1)%num_corners], center_point]
        elif minor_reward_dir == "right":
            self.minor_goal_sector = [*self.line_segments[(ith_goal+1)%num_corners], center_point]
        else:
            raise ValueError("minor_reward_dir must be 'left' or 'right'")


    def check(self):
        """ checks for if the goal is satisfied by the environment's agent """
        from matplotlib.path import Path
        goal_sector = Path(self.goal_sector)
        minor_goal_sector = Path(self.minor_goal_sector)
        
        rewards = {}
        for _, agent in self.env.Ags.items():
            if goal_sector.contains_point(agent.pos):
                rewards[agent] = self.reward
            elif minor_goal_sector.contains_point(agent.pos):
                rewards[agent] = self.minor_reward
        
        return rewards

    def render(self):
        """ renders the goal """
        from matplotlib.path import Path
        import matplotlib.patches as patches
        goal_sector = Path(self.goal_sector)
        minor_goal_sector = Path(self.minor_goal_sector)
         # Create patches for each sector
        patch_goal_sector       = patches.PathPatch(goal_sector, facecolor='grey', lw=1)
        patch_minor_goal_sector = patches.PathPatch(minor_goal_sector, facecolor='red', lw=1)
        # Add patches to the axes
        plt.gca().add_patch(patch_goal_sector)
        plt.gca().add_patch(patch_minor_goal_sector)

# -------
# EXAMPLE
# -------
env.plot_environment()
g1 = SectorGoal(env, ith_goal=0, minor_reward_dir="left")
g2 = SectorGoal(env, ith_goal=4, minor_reward_dir="left")
g1.render()
g2.render()
animal = [0.2, 0.26]
plt.scatter(*animal, c='k', s=100)
plt.show()

# Check if the goal is satisfied
env.Ags["agent_0"].pos = animal
print("Check if goal 1?", g1.check())
print("Check if goal 2?", g2.check())

        
# TODOS: OTHER THINGS TO BE AWARE OF
# 1. spatial goal environment has a special render() for plotting spatial goals
#   - at the moment, any goal type one wants to see in the plot needs a render()
#   - in the future, users propbably can add a render() directory to the Goal class
#   - right now, it's the environemnt's job


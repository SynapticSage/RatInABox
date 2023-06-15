import gymnasium
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper
from supersuit.utils.wrapper_chooser import WrapperChooser
from copy import copy


# class taskenv_sb3(BaseParallelWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#     def _check_valid_for_black_death(self):
#         for agent in self.agents:
#             space = self.observation_space(agent)
#             assert isinstance(
#                 space, gymnasium.spaces.Box
#             ), f"observation sapces for black death must be Box spaces, is {space}"
#
#     def reset(self, seed=None, options=None):
#         obss, infos = self.env.reset(seed=seed, options=options)
#
#         self.agents = self.env.agents[:]
#         self._check_valid_for_black_death()
#         black_obs = {
#             agent: np.zeros_like(self.observation_space(agent).low)
#             for agent in self.agents
#             if agent not in obss
#         }
#         return {**obss, **black_obs}, infos
#
#     def step(self, actions):
#         active_actions = {agent: actions[agent] for agent in self.env.agents}
#         obss, rews, terms, truncs, infos = self.env.step(active_actions)
#         black_obs = {
#             agent: np.zeros_like(self.observation_space(agent).low)
#             for agent in self.agents
#             if agent not in obss
#         }
#         black_rews = {agent: 0.0 for agent in self.agents if agent not in obss}
#         black_infos = {agent: {} for agent in self.agents if agent not in obss}
#         terminations = np.fromiter(terms.values(), dtype=bool)
#         truncations = np.fromiter(truncs.values(), dtype=bool)
#         env_is_done = (terminations & truncations).all()
#         total_obs = {**black_obs, **obss}
#         total_rews = {**black_rews, **rews}
#         total_infos = {**black_infos, **infos}
#         total_dones = {agent: env_is_done for agent in self.agents}
#         if env_is_done:
#             self.agents.clear()
#         return total_obs, total_rews, total_dones, total_dones, total_infos


import stable_baselines3
import supersuit as ss

def create_env():
    """
    Create a SpatialGoalEnvironment with two agents
    """
    goalcachekws = dict(agentmode="interact", goalorder="nonsequential",
                        reset_n_goals=5, verbose=False)
    rewardcachekws = dict(default_reward_level=-0.01)
    # Create a test environment
    env = SpatialGoalEnvironment(params={'dimensionality':'2D'},
                                 render_every=1, teleport_on_reset=False,
                                 goalcachekws=goalcachekws,
                                 rewardcachekws=rewardcachekws,
                                 verbose=False)
    goals = [SpatialGoal(env, pos=np.array([x, y]))
             for (x,y) in product((0.05, 0.5, 0.95), (0.05, 0.5, 0.95))]
    env.goal_cache.reset_goals = goals
    # Create rats who are part of the environment and accept action
    Ag = Agent(env);  env.add_agents(Ag) 
    env.possible_agents = copy(env.agents)
    # Ag2 = Agent(env); env.add_agents(Ag2)
    return env 

def env(**kwargs):
    env_ = create_env()
    env_ = ss.pettingzoo_env_to_vec_env_v1(env_)
    env_ = ss.concat_vec_envs_v1(env_, 1, base_class="stable_baselines3")
    # env_.action_spaces = gymnasium.spaces.Box(env_.action_space.low, 
    #                                           env_.action_space.high, 
    #                                           shape=env_.action_space.shape)
    # env_.observation_spaces = gymnasium.spaces.Box(env_.observation_space.low, 
    #                                                env_.observation_space.high, 
    #                                                shape=env_.observation_space.shape)
    return env_

E = env()
model = stable_baselines3.PPO('MlpPolicy', E, verbose=1)
model.learn(total_timesteps=10000)


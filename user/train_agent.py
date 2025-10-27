'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below. 

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import torch 
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER 
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment.environment import WarehouseBrawl, Player, GameObject
from enum import Enum
from functools import partial
import matplotlib

from environment.agent import *
from typing import Optional, Type, List, Tuple

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 256,  # Reduced for faster training
                'net_arch': [dict(pi=[128, 128], vf=[128, 128])],  # Smaller network
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,
            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=1,
                                      n_steps=2048,  # Much smaller for faster training
                                      batch_size=64,
                                      ent_coef=0.01,
                                      learning_rate=3e-4,
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
       
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140

def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold.
    """
    player: Player = env.objects["player"]
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0
    return reward * env.dt

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:
    """Penalty for holding too many keys simultaneously"""
    player: Player = env.objects["player"]
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt
    return 0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0
    
def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 2.0
        elif env.objects["player"].weapon == "Spear":
            return 1.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

def aggressive_positioning_reward(env: WarehouseBrawl) -> float:
    """Reward for maintaining optimal attack distance from opponent"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    distance = abs(player.body.position.x - opponent.body.position.x)
    optimal_distance = 2.0  # Sweet spot for attacks
    
    # Reward for being in optimal attack range
    if 1.5 <= distance <= 3.0:
        return 0.1 * env.dt
    # Small penalty for being too close or too far
    elif distance < 0.5 or distance > 6.0:
        return -0.05 * env.dt
    return 0.0

def successful_attack_reward(env: WarehouseBrawl) -> float:
    """Reward for landing attacks and starting combos"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    reward = 0.0
    
    # Reward for hitting opponent
    if opponent.damage_taken_this_frame > 0:
        reward += 0.5
    
    # Bonus for high-damage attacks
    if opponent.damage_taken_this_frame > 5:
        reward += 1.0
        
    # Reward for being in attack state (check by class name)
    if "Attack" in player.state.__class__.__name__:
        reward += 0.1 * env.dt
        
    return reward

def weapon_advantage_reward(env: WarehouseBrawl) -> float:
    """Reward for having and maintaining weapon advantage"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    weapon_value = {"Punch": 0, "Hammer": 2, "Spear": 1}
    player_weapon = weapon_value.get(player.weapon, 0)
    opp_weapon = weapon_value.get(opponent.weapon, 0)
    
    advantage = player_weapon - opp_weapon
    
    # Reward for having weapon advantage
    if advantage > 0:
        return 0.05 * env.dt * advantage
    # Penalty for being at weapon disadvantage
    elif advantage < 0:
        return 0.02 * env.dt * advantage
    return 0.0

def stage_control_reward(env: WarehouseBrawl) -> float:
    """Reward for controlling center stage and good positioning"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    stage_center = 0.0
    player_pos = player.body.position.x
    opp_pos = opponent.body.position.x
    
    # Reward for controlling center stage
    if abs(player_pos - stage_center) < abs(opp_pos - stage_center):
        return 0.05 * env.dt
    # Penalty for being cornered
    elif abs(player_pos) > 4.0:
        return -0.1 * env.dt
    return 0.0

def defensive_positioning_reward(env: WarehouseBrawl) -> float:
    """Reward for good defensive positioning and spacing"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    distance = abs(player.body.position.x - opponent.body.position.x)
    opponent_state_name = opponent.state.__class__.__name__
    
    # Reward for maintaining safe distance when opponent is attacking
    if "Attack" in opponent_state_name and distance > 2.0:
        return 0.08 * env.dt
    # Penalty for being too close to attacking opponent
    elif "Attack" in opponent_state_name and distance < 1.0:
        return -0.15 * env.dt
    return 0.0

def successful_dodge_reward(env: WarehouseBrawl) -> float:
    """Reward for successfully dodging attacks"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    reward = 0.0
    
    opponent_state_name = opponent.state.__class__.__name__
    player_state_name = player.state.__class__.__name__
    
    # Reward for dodging when opponent attacks
    if ("Attack" in opponent_state_name and 
        any(state in player_state_name for state in ['Dodge', 'Dash', 'Roll'])):
        reward += 0.3
        
    # Reward for avoiding damage through movement
    if (player.damage_taken_this_frame == 0 and 
        opponent.damage_taken_this_frame == 0 and
        "Attack" in opponent_state_name):
        reward += 0.1 * env.dt
        
    return reward

def movement_efficiency_reward(env: WarehouseBrawl) -> float:
    """Reward for efficient movement and reduced unnecessary inputs"""
    player: Player = env.objects["player"]
    
    # Penalize excessive direction changes (indicates inefficient movement)
    if hasattr(player, 'prev_x'):
        direction_change = abs(player.body.position.x - player.prev_x)
        if direction_change < 0.1:  # Minimal movement
            return -0.02 * env.dt
    return 0.0

def recovery_reward(env: WarehouseBrawl) -> float:
    """Reward for recovering from disadvantage"""
    player: Player = env.objects["player"]
    
    # Simple recovery reward based on damage reduction
    if hasattr(player, 'prev_damage'):
        if player.damage < getattr(player, 'prev_damage', player.damage):
            return 0.1
    return 0.0

def ledge_risk_penalty(env: WarehouseBrawl) -> float:
    """Penalty for being near edges"""
    player: Player = env.objects["player"]
    
    ledge_distance = min(abs(player.body.position.x - 5.33), 
                        abs(player.body.position.x + 5.33))
    
    # Penalty for being near ledge
    if ledge_distance < 1.0:
        return -0.1 * env.dt
    return 0.0

def combo_efficiency_reward(env: WarehouseBrawl) -> float:
    """Reward for efficient combo execution"""
    # Track consecutive hits (simplified implementation)
    if hasattr(env, 'consecutive_hits'):
        if env.consecutive_hits > 1:
            return 0.2 * env.consecutive_hits
    return 0.0

def on_kill_confirm_reward(env: WarehouseBrawl, agent: str) -> float:
    """Reward for confirming kills from high damage"""
    if agent == 'player' and env.objects["opponent"].damage > 100:
        return 2.0
    return 0.0

def init_damage_tracking(env: WarehouseBrawl):
    """Initialize damage tracking for recovery rewards"""
    player: Player = env.objects["player"]
    if not hasattr(player, 'prev_damage'):
        player.prev_damage = player.damage

# --------------------------------------------------------------------------------
# ----------------------------- REWARD MANAGER -----------------------------------
# --------------------------------------------------------------------------------

def gen_reward_manager():
    reward_functions = {
        # Combat rewards
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=2.0),
        'aggressive_positioning_reward': RewTerm(func=aggressive_positioning_reward, weight=0.3),
        'successful_attack_reward': RewTerm(func=successful_attack_reward, weight=1.5),
        
        # Strategic rewards
        'weapon_advantage_reward': RewTerm(func=weapon_advantage_reward, weight=0.8),
        'stage_control_reward': RewTerm(func=stage_control_reward, weight=0.2),
        
        # Defense rewards
        'defensive_positioning_reward': RewTerm(func=defensive_positioning_reward, weight=0.4),
        'successful_dodge_reward': RewTerm(func=successful_dodge_reward, weight=1.2),
        
        # Technical skill rewards
        'movement_efficiency_reward': RewTerm(func=movement_efficiency_reward, weight=0.1),
        'recovery_reward': RewTerm(func=recovery_reward, weight=2.0),
        'excessive_input_penalty': RewTerm(func=holding_more_than_3_keys, weight=-0.02),
        
        # Risk management
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.6),
        'ledge_risk_penalty': RewTerm(func=ledge_risk_penalty, weight=-0.8),
    }
    
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=100)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=25)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=15)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=12)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=8))
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------

if __name__ == '__main__':
    # Use RecurrentPPO for better temporal understanding
    my_agent = RecurrentPPOAgent()

    # Enhanced reward manager
    reward_manager = gen_reward_manager()
    
    # Self-play handler
    selfplay_handler = SelfPlayRandom(
        partial(type(my_agent))
    )

    # Improved save settings
    save_handler = SaveHandler(
        agent=my_agent,
        save_freq=25_000,  # Save more frequently for shorter training
        max_saved=10,      # Keep fewer models
        save_path='checkpoints',
        run_name='quick_training',
        mode=SaveHandlerMode.FORCE
    )

    # More balanced opponent mix
    opponent_specification = {
        'self_play': (4, selfplay_handler),
        'constant_agent': (2, partial(ConstantAgent)),
        'based_agent': (2, partial(BasedAgent)),
        'clockwork_agent': (1, partial(ClockworkAgent)),
    }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    # Train with reasonable timesteps
    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=100_000,
        train_logging=TrainLogging.TO_FILE
    )
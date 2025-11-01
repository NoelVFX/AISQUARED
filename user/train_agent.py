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

def ledge_penalty_reward(env: WarehouseBrawl) -> float:
    """Smarter ledge detection that allows platform navigation"""
    player: Player = env.objects["player"]
    
    player_x, player_y = player.body.position.x, player.body.position.y
    
    # Stage boundaries (adjust based on your stage)
    left_ledge = -5.0
    right_ledge = 5.0
    
    # Only penalize if actually falling off (low Y) and near edge
    if player_y < -2.0:  # Actually falling
        if player_x < left_ledge + 0.5 or player_x > right_ledge - 0.5:
            return -0.5 * env.dt
    
    # Less penalty for being near edge but on platform or high up
    if player_y > 1.0:  # On platform or in air
        if player_x < left_ledge + 0.3 or player_x > right_ledge - 0.3:
            return -0.1 * env.dt  # Smaller penalty
    
    return 0.0

def suicide_penalty_reward(env: WarehouseBrawl) -> float:
    """Massive penalty for falling off"""
    player: Player = env.objects["player"]
    
    # Check if player is falling off (adjust threshold based on your stage)
    if player.body.position.y < -10.0:  # Fell off bottom
        return -10.0
    return 0.0

def center_stage_bonus_reward(env: WarehouseBrawl) -> float:
    """Reward for staying near center"""
    player: Player = env.objects["player"]
    
    distance_from_center = abs(player.body.position.x)
    
    # Reward for being in center area
    if distance_from_center < 2.0:
        return 0.1 * env.dt
    return 0.0

def aggression_reward(env: WarehouseBrawl) -> float:
    """Reward for being close to opponent and in attack position"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    distance = abs(player.body.position.x - opponent.body.position.x)
    
    # Reward for being in optimal attack range
    if 1.0 <= distance <= 3.0:
        return 0.2 * env.dt
    # Small penalty for being too far
    elif distance > 5.0:
        return -0.1 * env.dt
    return 0.0

def attack_frequency_reward(env: WarehouseBrawl) -> float:
    """Reward for frequently attempting attacks"""
    player: Player = env.objects["player"]
    
    # Check if player is in attack state
    state_name = player.state.__class__.__name__
    
    if "Attack" in state_name:
        return 0.15 * env.dt
    return 0.0

def smart_closing_distance_reward(env: WarehouseBrawl) -> float:
    """Reward for moving toward opponent only when safe"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    if not hasattr(env, 'prev_distance'):
        env.prev_distance = abs(player.body.position.x - opponent.body.position.x)
        return 0.0
    
    current_distance = abs(player.body.position.x - opponent.body.position.x)
    distance_change = env.prev_distance - current_distance  # Positive = getting closer
    
    env.prev_distance = current_distance
    
    # Only reward closing distance when not near edges
    player_x, player_y = player.body.position.x, player.body.position.y
    stage_half_width = 5.0
    
    # Check if safe to approach (not near edges)
    is_safe = abs(player_x) < stage_half_width - 1.0
    
    if distance_change > 0.1 and is_safe:  # Getting closer and safe
        return 0.1
    elif distance_change < -0.1 and is_safe:  # Moving away and safe
        return -0.05
    elif not is_safe and distance_change > 0.1:  # Getting closer but near edge
        return 0.02  # Small reward to encourage careful approach
    elif not is_safe and distance_change < -0.1:  # Moving away from edge - good!
        return 0.05  # Reward moving away from edges
    
    return 0.0

def combo_aggression_reward(env: WarehouseBrawl) -> float:
    """Reward for continuous aggressive actions"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    if not hasattr(env, 'aggression_counter'):
        env.aggression_counter = 0
        env.last_aggression_step = 0
    
    current_step = getattr(env, 'step_count', 0)
    
    # Check for aggressive actions
    state_name = player.state.__class__.__name__
    distance = abs(player.body.position.x - opponent.body.position.x)
    
    is_aggressive = (
        "Attack" in state_name or 
        distance < 3.0 or
        opponent.damage_taken_this_frame > 0
    )
    
    if is_aggressive:
        env.aggression_counter += 1
        env.last_aggression_step = current_step
    else:
        # Reset if too much time passes without aggression
        if current_step - env.last_aggression_step > 60:  # 2 seconds at 30fps
            env.aggression_counter = 0
    
    # Reward sustained aggression
    if env.aggression_counter > 10:
        return 0.05 * min(env.aggression_counter / 10, 3.0)  # Cap at 3x bonus
    return 0.0

def pressure_reward(env: WarehouseBrawl) -> float:
    """Reward for keeping pressure on opponent (being close when opponent is vulnerable)"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    distance = abs(player.body.position.x - opponent.body.position.x)
    opp_state_name = opponent.state.__class__.__name__
    
    # Opponent is vulnerable if in hitstun, knockdown, or recovery
    is_opponent_vulnerable = any(state in opp_state_name for state in ['Hit', 'Knockdown', 'Stun', 'Recovery'])
    
    if is_opponent_vulnerable and distance < 3.0:
        return 0.3 * env.dt  # Big reward for pressuring vulnerable opponent
    elif is_opponent_vulnerable and distance < 5.0:
        return 0.1 * env.dt  # Smaller reward for being somewhat close
    
    return 0.0

def offensive_positioning_reward(env: WarehouseBrawl) -> float:
    """Reward for maintaining offensive positioning (higher ground, center control, etc.)"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    reward = 0.0
    
    # Reward for having height advantage
    if player.body.position.y > opponent.body.position.y + 0.5:
        reward += 0.05 * env.dt
    
    # Reward for having center stage advantage
    player_dist_from_center = abs(player.body.position.x)
    opp_dist_from_center = abs(opponent.body.position.x)
    
    if player_dist_from_center < opp_dist_from_center:
        reward += 0.03 * env.dt
    
    # Reward for being closer to opponent (aggressive positioning)
    distance = abs(player.body.position.x - opponent.body.position.x)
    if distance < 3.0:
        reward += 0.04 * env.dt
    
    return reward

def safe_relentless_reward(env: WarehouseBrawl) -> float:
    """Reward for continuous movement toward opponent only when safe"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    if not hasattr(env, 'prev_player_x'):
        env.prev_player_x = player.body.position.x
        return 0.0
    
    player_to_opponent = opponent.body.position.x - player.body.position.x
    movement = player.body.position.x - env.prev_player_x
    
    env.prev_player_x = player.body.position.x
    
    # Check safety conditions
    player_x, player_y = player.body.position.x, player.body.position.y
    stage_half_width = 5.0
    is_safe = abs(player_x) < stage_half_width - 1.0
    
    # Only reward movement toward opponent when safe
    if abs(player_to_opponent) > 1.0 and is_safe:  # Only if not already very close and safe
        if (player_to_opponent > 0 and movement > 0.01) or (player_to_opponent < 0 and movement < -0.01):
            return 0.08 * env.dt
        elif (player_to_opponent > 0 and movement < -0.01) or (player_to_opponent < 0 and movement > 0.01):
            return -0.04 * env.dt
    
    return 0.0

def smart_platform_navigation_reward(env: WarehouseBrawl) -> float:
    """Reward for using platforms intelligently to reach opponent"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    reward = 0.0
    
    # Check if opponent is on the other side of a gap
    player_x, player_y = player.body.position.x, player.body.position.y
    opp_x, opp_y = opponent.body.position.x, opponent.body.position.y
    
    # If opponent is on other side and player uses platform to cross
    if abs(opp_x - player_x) > 4.0:  # Opponent is far away on other side
        if player_y > 2.0:  # Player is on a platform
            # Reward for using platform to cross gap
            reward += 0.1 * env.dt
            
            # Extra reward if moving toward opponent while on platform
            if (opp_x > player_x and player.body.velocity.x > 0) or (opp_x < player_x and player.body.velocity.x < 0):
                reward += 0.05 * env.dt
    
    return reward

def safe_platform_landing_reward(env: WarehouseBrawl) -> float:
    """Reward for safely landing on platforms and transitioning between them"""
    player: Player = env.objects["player"]
    
    if not hasattr(env, 'prev_player_y'):
        env.prev_player_y = player.body.position.y
        return 0.0
    
    current_y = player.body.position.y
    prev_y = env.prev_player_y
    env.prev_player_y = current_y
    
    # Reward for successful platform landing (coming down onto platform)
    if prev_y > current_y and 2.0 < current_y < 4.0:  # Landing on platform height
        if abs(player.body.velocity.y) < 2.0:  # Controlled landing
            return 0.2
    
    return 0.0

def platform_to_platform_reward(env: WarehouseBrawl) -> float:
    """Reward for successfully moving between platforms"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    if not hasattr(env, 'platform_transition_start'):
        env.platform_transition_start = None
    
    player_x, player_y = player.body.position.x, player.body.position.y
    opp_x, opp_y = opponent.body.position.x, opponent.body.position.y
    
    # Check if starting platform transition
    if player_y > 2.0 and env.platform_transition_start is None:
        env.platform_transition_start = (player_x, player_y)
    
    # Check if successfully completed platform transition
    if env.platform_transition_start is not None and player_y < 1.0:  # Back on ground
        start_x, start_y = env.platform_transition_start
        distance_traveled = abs(player_x - start_x)
        
        # Reward for covering significant distance via platforms
        if distance_traveled > 3.0:
            reward = 0.3
            # Bonus if transition got closer to opponent
            start_dist_to_opp = abs(start_x - opp_x)
            current_dist_to_opp = abs(player_x - opp_x)
            if current_dist_to_opp < start_dist_to_opp:
                reward += 0.2
        else:
            reward = 0.1
            
        env.platform_transition_start = None
        return reward
    
    return 0.0

def avoid_stuck_platform_penalty(env: WarehouseBrawl) -> float:
    """Penalty for getting stuck on platforms unnecessarily"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    player_x, player_y = player.body.position.x, player.body.position.y
    opp_x, opp_y = opponent.body.position.x, opponent.body.position.y
    
    # Penalty for staying on platform when opponent is easily reachable on ground
    if player_y > 2.0 and opp_y < 1.0:  # On platform when opponent is on ground
        horizontal_dist = abs(player_x - opp_x)
        if horizontal_dist < 3.0:  # Close enough to engage without platforms
            return -0.1 * env.dt
    
    return 0.0

def symmetric_movement_reward(env: WarehouseBrawl) -> float:
    """Reward for moving in both directions equally (prevents directional bias)"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    if not hasattr(env, 'left_movement_count'):
        env.left_movement_count = 0
        env.right_movement_count = 0
        env.last_movement_direction = 0
    
    # Track movement direction
    current_velocity_x = player.body.velocity.x
    
    if current_velocity_x > 0.1:  # Moving right
        env.right_movement_count += 1
        env.last_movement_direction = 1
    elif current_velocity_x < -0.1:  # Moving left
        env.left_movement_count += 1
        env.last_movement_direction = -1
    
    # Reward balanced movement (only check after some time)
    total_movements = env.left_movement_count + env.right_movement_count
    if total_movements > 50:
        balance_ratio = min(env.left_movement_count, env.right_movement_count) / max(env.left_movement_count, env.right_movement_count)
        if balance_ratio > 0.7:  # Good balance
            return 0.02 * env.dt
        elif balance_ratio < 0.3:  # Poor balance
            return -0.01 * env.dt
    
    return 0.0

def bidirectional_approach_reward(env: WarehouseBrawl) -> float:
    """Reward for approaching opponent from both left and right sides"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    if not hasattr(env, 'left_approaches'):
        env.left_approaches = 0
        env.right_approaches = 0
    
    player_x = player.body.position.x
    opp_x = opponent.body.position.x
    player_vel_x = player.body.velocity.x
    
    # Check if approaching opponent
    distance = abs(player_x - opp_x)
    if distance < 4.0 and abs(player_vel_x) > 0.5:
        if player_x < opp_x and player_vel_x > 0:  # Approaching from left
            env.left_approaches += 1
            return 0.05
        elif player_x > opp_x and player_vel_x < 0:  # Approaching from right
            env.right_approaches += 1
            return 0.05
    
    return 0.0

def spawn_agnostic_aggression_reward(env: WarehouseBrawl) -> float:
    """Reward aggression regardless of spawn position"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    if not hasattr(env, 'initial_spawn_difference'):
        # Record initial positions to understand spawn configuration
        env.initial_spawn_difference = opponent.body.position.x - player.body.position.x
    
    current_distance = abs(player.body.position.x - opponent.body.position.x)
    
    # Reward for closing distance regardless of direction
    if not hasattr(env, 'prev_distance'):
        env.prev_distance = current_distance
        return 0.0
    
    distance_change = env.prev_distance - current_distance
    env.prev_distance = current_distance
    
    if distance_change > 0.1:  # Getting closer
        return 0.08
    elif distance_change < -0.2:  # Moving away significantly
        return -0.04
    
    return 0.0

def predict_respawn_position(env: WarehouseBrawl) -> float:
    """Predict where opponent will respawn based on map layout"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    player_x = player.body.position.x
    
    # Respawn logic based on map layout:
    # - Ground1: Left platform at y=2.85, x=-7.0 to -2.0
    # - Ground2: Right platform at y=0.85, x=2.0 to 7.0
    # - Moving Stage: Between (1,0) and (-1,2)
    
    # If player is on right side, opponent respawns on left platform
    if player_x > 0:
        # Respawn on left platform (Ground1) - center at x=-4.5
        return -4.5
    else:
        # Respawn on right platform (Ground2) - center at x=4.5
        return 4.5

def smart_respawn_anticipation_reward(env: WarehouseBrawl) -> float:
    """Reward for moving toward opponent's respawn point using actual map coordinates"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Check if opponent is KO'd
    opp_state_name = opponent.state.__class__.__name__
    is_opponent_KO = any(state in opp_state_name for state in ['Knockdown', 'KO', 'Dead', 'Respawning'])
    
    if is_opponent_KO:
        player_x, player_y = player.body.position.x, player.body.position.y
        target_respawn_x = predict_respawn_position(env)
        
        # Calculate optimal path to respawn point
        distance_to_respawn = abs(player_x - target_respawn_x)
        
        if not hasattr(env, 'prev_respawn_distance'):
            env.prev_respawn_distance = distance_to_respawn
            return 0.0
        
        distance_change = env.prev_respawn_distance - distance_to_respawn
        env.prev_respawn_distance = distance_to_respawn
        
        # Enhanced reward based on map awareness
        reward = 0.0
        
        if distance_change > 0.05:  # Moving toward respawn point
            reward += 0.1
            
            # Bonus for taking efficient path (considering platforms)
            if _is_taking_efficient_path(env, player_x, player_y, target_respawn_x):
                reward += 0.05
                
        elif distance_change < -0.05:  # Moving away from respawn point
            reward -= 0.05
        
        return reward
    
    # Reset when opponent respawns
    elif hasattr(env, 'prev_respawn_distance'):
        del env.prev_respawn_distance
    
    return 0.0

def _is_taking_efficient_path(env: WarehouseBrawl, player_x: float, player_y: float, target_x: float) -> bool:
    """Check if player is taking an efficient path considering platform layout"""
    # Platform coordinates from your description:
    # Ground1: y=2.85, x=-7.0 to -2.0
    # Ground2: y=0.85, x=2.0 to 7.0
    # Moving Stage: between (1,0) and (-1,2)
    
    # Efficient path: use platforms when they help, avoid when unnecessary
    requires_platform_crossing = (player_x < -2.0 and target_x > 2.0) or (player_x > 2.0 and target_x < -2.0)
    
    if requires_platform_crossing:
        # Should use moving platform or jump between platforms
        if player_y > 1.0:  # On platform height
            return True
    else:
        # No platform needed, should stay on ground
        if player_y < 1.5:  # On ground level
            return True
    
    return False

def platform_aware_movement_reward(env: WarehouseBrawl) -> float:
    """Reward for intelligent platform usage based on actual map layout"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    player_x, player_y = player.body.position.x, player.body.position.y
    opp_x, opp_y = opponent.body.position.x, opponent.body.position.y
    
    reward = 0.0
    
    # Platform coordinates
    left_platform_range = (-7.0, -2.0)   # Ground1
    right_platform_range = (2.0, 7.0)     # Ground2
    moving_platform_center_range = (-1.0, 1.0)  # Moving stage area
    
    # Check if on appropriate platform for current situation
    horizontal_distance = abs(player_x - opp_x)
    
    # Use platforms when opponent is on opposite side
    if (player_x < 0 and opp_x > 2.0) or (player_x > 0 and opp_x < -2.0):
        # Opponent is on opposite side, should use platforms
        if (left_platform_range[0] <= player_x <= left_platform_range[1] and player_y > 2.0) or \
           (right_platform_range[0] <= player_x <= right_platform_range[1] and player_y > 0.85) or \
           (moving_platform_center_range[0] <= player_x <= moving_platform_center_range[1] and player_y > 0.5):
            reward += 0.08 * env.dt  # Good platform usage
    else:
        # Opponent is on same side, should use ground
        if player_y < 1.5:  # On ground level
            reward += 0.05 * env.dt  # Good ground positioning
    
    return reward

def map_aware_edge_avoidance(env: WarehouseBrawl) -> float:
    """Edge avoidance using actual map boundaries"""
    player: Player = env.objects["player"]
    
    player_x, player_y = player.body.position.x, player.body.position.y
    
    # Map boundaries from your description
    left_boundary = -7.45   # Screen width 14.9, so half is 7.45
    right_boundary = 7.45
    bottom_boundary = -4.97  # Screen height 9.94, so half is 4.97
    
    penalty = 0.0
    
    # Penalty for approaching screen edges
    distance_to_left_edge = abs(player_x - left_boundary)
    distance_to_right_edge = abs(player_x - right_boundary)
    distance_to_bottom = abs(player_y - bottom_boundary)
    
    # Strong penalty near screen edges (death zones)
    if distance_to_left_edge < 1.0 or distance_to_right_edge < 1.0:
        penalty -= 0.3 * env.dt
    elif distance_to_left_edge < 2.0 or distance_to_right_edge < 2.0:
        penalty -= 0.15 * env.dt
    
    # Penalty for falling off (below ground level)
    if player_y < bottom_boundary + 1.0:
        penalty -= 0.2 * env.dt
    
    return penalty

def aggressive_approach_reward(env: WarehouseBrawl) -> float:
    """Reward for actively moving toward opponent"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    if not hasattr(env, 'prev_distance_to_opponent'):
        env.prev_distance_to_opponent = abs(player.body.position.x - opponent.body.position.x)
        return 0.0
    
    current_distance = abs(player.body.position.x - opponent.body.position.x)
    distance_change = env.prev_distance_to_opponent - current_distance  # Positive = getting closer
    
    env.prev_distance_to_opponent = current_distance
    
    # Strong reward for closing distance
    if distance_change > 0.1:  # Getting closer
        return 0.15
    elif distance_change < -0.2:  # Moving away significantly
        return -0.1
    
    return 0.0

def platform_safety_reward(env: WarehouseBrawl) -> float:
    """Reward for staying on platforms and avoiding edges"""
    player: Player = env.objects["player"]
    
    player_x, player_y = player.body.position.x, player.body.position.y
    
    # Map boundaries
    left_boundary = -7.45
    right_boundary = 7.45
    bottom_boundary = -4.97
    
    reward = 0.0
    
    # Reward for being on safe ground (not near edges)
    safe_zone_left = -5.0
    safe_zone_right = 5.0
    
    if safe_zone_left <= player_x <= safe_zone_right:
        reward += 0.05 * env.dt  # Constant reward for staying in safe zone
    
    # Penalty for being too close to edges
    edge_threshold = 1.0
    if player_x < safe_zone_left + edge_threshold or player_x > safe_zone_right - edge_threshold:
        reward -= 0.1 * env.dt
    
    # Big penalty for falling off
    if player_y < -3.0:
        reward -= 0.5
    
    return reward

def jump_toward_opponent_reward(env: WarehouseBrawl) -> float:
    """Reward for jumping toward opponent intelligently"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    player_x, player_y = player.body.position.x, player.body.position.y
    opp_x, opp_y = opponent.body.position.x, opponent.body.position.y
    
    distance_x = abs(player_x - opp_x)
    
    # Check if player is jumping toward opponent
    if player_y > 1.0:  # Player is in air
        # Determine direction to opponent
        direction_to_opponent = 1 if opp_x > player_x else -1
        player_velocity_x = player.body.velocity.x
        
        # Reward for moving toward opponent while jumping
        if (direction_to_opponent > 0 and player_velocity_x > 0.5) or (direction_to_opponent < 0 and player_velocity_x < -0.5):
            return 0.2 * env.dt
    
    return 0.0

def continuous_engagement_reward(env: WarehouseBrawl) -> float:
    """Reward for staying engaged with opponent"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    distance = abs(player.body.position.x - opponent.body.position.x)
    
    # Reward for maintaining engagement distance
    if 2.0 <= distance <= 5.0:
        return 0.1 * env.dt
    # Penalty for being too far
    elif distance > 8.0:
        return -0.15 * env.dt
    
    return 0.0

def facing_opponent_reward(env: WarehouseBrawl) -> float:
    """Reward for positioning that faces the enemy"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    player_x = player.body.position.x
    opp_x = opponent.body.position.x
    
    # Detect facing direction using velocity and position
    # If moving, use velocity direction; if stationary, use position relative to opponent
    player_velocity_x = player.body.velocity.x
    
    if abs(player_velocity_x) > 0.1:  # Moving - use velocity direction
        player_facing_right = player_velocity_x > 0
    else:  # Stationary - face toward opponent by default
        player_facing_right = player_x < opp_x
    
    # Reward for facing toward opponent
    if (player_facing_right and player_x < opp_x) or (not player_facing_right and player_x > opp_x):
        return 0.1 * env.dt
    
    return 0.0

def defensive_positioning_reward(env: WarehouseBrawl) -> float:
    """Reward for optimal defensive positioning (distance + facing)"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    distance = abs(player.body.position.x - opponent.body.position.x)
    player_x = player.body.position.x
    opp_x = opponent.body.position.x
    
    # Detect facing direction
    player_velocity_x = player.body.velocity.x
    
    if abs(player_velocity_x) > 0.1:
        player_facing_right = player_velocity_x > 0
    else:
        player_facing_right = player_x < opp_x
    
    facing_correct = (player_facing_right and player_x < opp_x) or (not player_facing_right and player_x > opp_x)
    
    # Optimal defensive positioning: safe distance + facing opponent
    if 2.5 <= distance <= 4.0 and facing_correct:
        return 0.15 * env.dt
    elif distance < 2.0 and facing_correct:
        return 0.05 * env.dt  # Smaller reward if too close but facing correctly
    
    return 0.0

def safe_positioning_reward(env: WarehouseBrawl) -> float:
    """Reward for maintaining safe distance while keeping stage control"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    distance = abs(player.body.position.x - opponent.body.position.x)
    player_x = player.body.position.x
    
    # Safe distance range
    if 2.5 <= distance <= 4.0:
        # Bonus for center stage control
        if abs(player_x) < 2.0:
            return 0.12 * env.dt
        else:
            return 0.08 * env.dt
    # Penalty for being too close
    elif distance < 1.5:
        return -0.1 * env.dt
    
    return 0.0

def weapon_acquisition_reward(env: WarehouseBrawl) -> float:
    """Reward for acquiring and keeping weapons"""
    player: Player = env.objects["player"]
    
    weapon_value = {"Punch": 0, "Hammer": 2, "Spear": 1}
    current_weapon = weapon_value.get(player.weapon, 0)
    
    if not hasattr(env, 'previous_weapon'):
        env.previous_weapon = current_weapon
    
    reward = 0.0
    
    # Reward for weapon upgrade
    if current_weapon > env.previous_weapon:
        reward += 1.5
    
    # Small constant reward for having a weapon
    if current_weapon > 0:
        reward += 0.03 * env.dt
    
    env.previous_weapon = current_weapon
    return reward

def defensive_spacing_reward(env: WarehouseBrawl) -> float:
    """Reward for maintaining optimal defensive spacing from opponent"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    distance = abs(player.body.position.x - opponent.body.position.x)
    opponent_state_name = opponent.state.__class__.__name__
    
    # Optimal defensive spacing ranges
    safe_distance = 3.0
    danger_distance = 1.5
    
    # Big reward for maintaining safe distance when opponent is attacking
    if "Attack" in opponent_state_name and distance > safe_distance:
        return 0.2 * env.dt
    
    # Reward for general safe spacing
    if safe_distance - 0.5 <= distance <= safe_distance + 1.0:
        return 0.1 * env.dt
    
    # Penalty for being in danger zone when opponent can attack
    if distance < danger_distance:
        return -0.15 * env.dt
        
    return 0.0

def strategic_positioning_reward(env: WarehouseBrawl) -> float:
    """Reward for good strategic positioning (NOT running away)"""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    distance = abs(player.body.position.x - opponent.body.position.x)
    player_x = player.body.position.x
    
    # Stage boundaries
    stage_half_width = 5.0
    safe_zone_boundary = 3.5  # Don't go too close to edges
    
    # PENALIZE being near edges (anti-runaway)
    if abs(player_x) > safe_zone_boundary:
        return -0.2 * env.dt
    
    # Reward for maintaining engagement distance while staying in safe zone
    if 1.5 <= distance <= 3.5 and abs(player_x) < safe_zone_boundary:
        return 0.1 * env.dt
    
    # Penalty for being too far (running away)
    if distance > 6.0:
        return -0.15 * env.dt
        
    return 0.0

def center_control_reward(env: WarehouseBrawl) -> float:
    """Reward for controlling center stage (prevents edge running)"""
    player: Player = env.objects["player"]
    
    player_x = player.body.position.x
    
    # Strong reward for center stage control
    if abs(player_x) < 2.0:
        return 0.15 * env.dt
    
    # Penalty for edge hugging
    if abs(player_x) > 4.0:
        return -0.1 * env.dt
    
    return 0.0

def anti_suicide_reward(env: WarehouseBrawl) -> float:
    """Strong penalty for any behavior that leads to self-destruction"""
    player: Player = env.objects["player"]
    
    player_x = player.body.position.x
    player_y = player.body.position.y
    
    # Stage boundaries (adjust based on your stage)
    stage_half_width = 5.0
    bottom_death_zone = -3.0
    
    penalty = 0.0
    
    # Strong penalty for approaching edges while low
    if player_y < 1.0:  # On ground or low
        if abs(player_x) > stage_half_width - 0.5:  # Very close to edge
            penalty -= 0.3 * env.dt
        elif abs(player_x) > stage_half_width - 1.5:  # Close to edge
            penalty -= 0.15 * env.dt
    
    # Massive penalty for actually falling
    if player_y < bottom_death_zone:
        penalty -= 1.0
    
    return penalty

# --------------------------------------------------------------------------------
# ----------------------------- REWARD MANAGER -----------------------------------
# --------------------------------------------------------------------------------

def gen_reward_manager():
    reward_functions = {
     # DEFENSIVE POSITIONING REWARDS - HIGH PRIORITY
    'defensive_positioning_reward': RewTerm(func=defensive_positioning_reward, weight=3.5),
    'facing_opponent_reward': RewTerm(func=facing_opponent_reward, weight=3.0),
    'strategic_positioning_reward': RewTerm(func=strategic_positioning_reward, weight=3.0),
    'center_control_reward': RewTerm(func=center_control_reward, weight=2.5),
    'defensive_spacing_reward': RewTerm(func=defensive_spacing_reward, weight=2.0),
    'anti_suicide_reward': RewTerm(func=anti_suicide_reward, weight=2.5),

    # WEAPON & STRATEGIC REWARDS - HIGH PRIORITY  
    'weapon_advantage_reward': RewTerm(func=weapon_advantage_reward, weight=3.0),
    'weapon_acquisition_reward': RewTerm(func=weapon_acquisition_reward, weight=2.5),
    'stage_control_reward': RewTerm(func=stage_control_reward, weight=2.0),
    'center_stage_bonus_reward': RewTerm(func=center_stage_bonus_reward, weight=1.5),

    # REDUCED AGGRESSION REWARDS - MEDIUM/LOW PRIORITY
    'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=2.0),
    'successful_attack_reward': RewTerm(func=successful_attack_reward, weight=1.0),
    'aggressive_approach_reward': RewTerm(func=aggressive_approach_reward, weight=0.5),
    'continuous_engagement_reward': RewTerm(func=continuous_engagement_reward, weight=0.3),

    # DEFENSIVE MOVEMENT REWARDS
    'platform_safety_reward': RewTerm(func=platform_safety_reward, weight=2.0),
    'smart_platform_navigation_reward': RewTerm(func=smart_platform_navigation_reward, weight=1.0),
    'safe_platform_landing_reward': RewTerm(func=safe_platform_landing_reward, weight=1.0),
    'symmetric_movement_reward': RewTerm(func=symmetric_movement_reward, weight=0.8),

    # ANTI-SUICIDE REWARDS (keep high for safety)
    'suicide_penalty_reward': RewTerm(func=suicide_penalty_reward, weight=-10.0),
    'ledge_penalty_reward': RewTerm(func=ledge_penalty_reward, weight=-3.0),

    # MINIMAL AGGRESSION REWARDS
    'jump_toward_opponent_reward': RewTerm(func=jump_toward_opponent_reward, weight=0.2),
    'aggressive_positioning_reward': RewTerm(func=aggressive_positioning_reward, weight=0.1),
    'bidirectional_approach_reward': RewTerm(func=bidirectional_approach_reward, weight=0.3),
    'spawn_agnostic_aggression_reward': RewTerm(func=spawn_agnostic_aggression_reward, weight=0.2),
    }
    
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=100)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=25)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=15)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=12)),
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------

if __name__ == '__main__':
    # Use PPO for aggressive behavior
    my_agent = SB3Agent(sb3_class=PPO)

    # Enhanced reward manager with aggressive approach rewards
    reward_manager = gen_reward_manager()
    
    # Self-play handler
    selfplay_handler = SelfPlayRandom(partial(type(my_agent)))
    
    # Create directory if it doesn't exist
    save_dir = 'checkpoints/symmetric_training'  # Changed folder name
    os.makedirs(save_dir, exist_ok=True)
    
    # Improved save settings
    save_handler = SaveHandler(
        agent=my_agent,
        save_freq=25_000,  # Save more frequently
        max_saved=10,
        save_path=save_dir,
        run_name='symmetric_model',
        mode=SaveHandlerMode.FORCE
    )

    # More aggressive opponent mix that forces engagement
    opponent_specification = {
        'self_play': (3, selfplay_handler),
        'based_agent': (1, partial(BasedAgent)),  # Reduced weight
        'aggressive_closer': (3, partial(ClockworkAgent, action_sheet=[
            (15, ['d', 'space']),  # Right + jump
            (10, ['d', 'j']),      # Right + attack
            (15, ['a', 'space']),  # Left + jump  
            (10, ['a', 'j']),      # Left + attack
            (5, ['space', 'j']),   # Jump attack
        ])),
        'platform_aggressor': (2, partial(ClockworkAgent, action_sheet=[
            (10, ['d', 'space']),  # Right jump
            (5, ['d']),           # Right move
            (10, ['space', 'j']), # Jump attack
            (10, ['a', 'space']), # Left jump
            (5, ['a']),           # Left move
        ])),
        'constant_attacker': (1, partial(ClockworkAgent, action_sheet=[
            (5, ['j']),           # Constant attacks
            (5, ['d']),           # Movement
            (5, ['a']),           # Movement
            (5, ['space']),       # Jumping
        ])),
    }
    
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    # Train with focus on aggressive behavior
    print("🎯 Starting AGGRESSIVE training...")
    print("   - Emphasizing approach behavior")
    print("   - Rewarding platform safety") 
    print("   - Encouraging jump attacks")
    
    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=500_000,  # Extended training
        train_logging=TrainLogging.TO_FILE
    )

    # Save the final model
    final_model_path = os.path.join(save_dir, "rl-model.zip")
    my_agent.save(final_model_path)
    print(f"✅ Defensive model saved as: {final_model_path}")
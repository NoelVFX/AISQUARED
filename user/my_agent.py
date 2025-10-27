import os
import gdown
from typing import Optional
import numpy as np
from environment.agent import Agent
from stable_baselines3 import PPO

class SubmittedAgent(Agent):
    '''
    Hybrid Agent: Uses RL for most decisions but rule-based for critical situations
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)
        self.emergency_mode = False

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = PPO(
                "MlpPolicy", 
                self.env, 
                verbose=0,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,
            )
            del self.env
        else:
            self.model = PPO.load(self.file_path)

    def _emergency_actions(self, obs):
        """Rule-based actions for critical situations"""
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        player_damage = self.obs_helper.get_section(obs, 'player_damage')
        
        action = self.act_helper.zeros()
        
        # Emergency recovery when at high damage
        if player_damage > 80:
            # Run away and play defensively
            if pos[0] > opp_pos[0]:
                action = self.act_helper.press_keys(['d'])  # Move right
            else:
                action = self.act_helper.press_keys(['a'])  # Move left
                
            # Add shield for defense
            action = self.act_helper.press_keys(['g'], action)
            
        # Edge guarding - prevent opponent from recovering
        elif opp_pos[1] < -2.0:  # Opponent is off-stage
            # Move to edge and attack
            if abs(pos[0] - opp_pos[0]) < 3.0:
                action = self.act_helper.press_keys(['j'])  # Attack
                
        return action

    def predict(self, obs):
        # Check if we need emergency actions
        player_damage = self.obs_helper.get_section(obs, 'player_damage')
        opp_damage = self.obs_helper.get_section(obs, 'opponent_damage')
        
        # Use emergency rules in critical situations
        if player_damage > 80 or (opp_damage > 90 and player_damage < 50):
            self.emergency_mode = True
            return self._emergency_actions(obs)
        else:
            self.emergency_mode = False
            action, _ = self.model.predict(obs, deterministic=True)
            return action

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
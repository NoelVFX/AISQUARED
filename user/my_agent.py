import os
import gdown
from typing import Optional
import numpy as np
from environment.agent import Agent
from stable_baselines3 import PPO
from stable_baselines3.common.utils import constant_fn

class SubmittedAgent(Agent):
    def __init__(self, file_path: str = None):
        super().__init__(file_path)
        self.model = None
        self.step_count = 0
        self._initialized = False
        self.last_action = None
        print(f"🎯 SubmittedAgent created with file: {file_path}")

    def _initialize(self) -> None:
        if self._initialized:
            return
            
        print(f"🔄 Initializing SubmittedAgent...")
        
        # Try to load model if path provided
        if self.file_path and os.path.exists(self.file_path):
            print(f"📁 Found model file: {self.file_path}")
            try:
                # Try multiple loading methods
                try:
                    self.model = PPO.load(self.file_path)
                    print("✅ Model loaded with default method")
                except Exception as e:
                    print(f"⚠️ Default load failed: {e}")
                    # Try with custom objects
                    custom_objects = {
                        "learning_rate": 0.0,
                        "lr_schedule": lambda _: 0.0,
                        "clip_range": lambda _: 0.0,
                    }
                    self.model = PPO.load(self.file_path, custom_objects=custom_objects)
                    print("✅ Model loaded with custom objects")
                    
                if self.model:
                    print(f"📊 Model info:")
                    print(f"   - Observation space: {self.model.observation_space}")
                    print(f"   - Action space: {self.model.action_space}")
                    print(f"   - Policy: {self.model.policy}")
                
            except Exception as e:
                print(f"❌ All loading methods failed: {e}")
                self.model = None
        else:
            print(f"❌ Model file not found: {self.file_path}")
            self.model = None
            
        self._initialized = True
        print("✅ SubmittedAgent initialization complete")

    def _emergency_movement(self):
        """More aggressive movement that actually attacks"""
        self.step_count += 1
        
        action = self.act_helper.zeros()
        
        # More dynamic movement pattern
        cycle = self.step_count % 300
        
        if cycle < 75:
            # Move right and attack frequently
            action = self.act_helper.press_keys(['d'], action)
            if cycle % 15 == 0:  # Attack every 15 frames
                action = self.act_helper.press_keys(['j'], action)
        elif cycle < 150:
            # Move left and use heavy attack
            action = self.act_helper.press_keys(['a'], action)
            if cycle % 20 == 0:
                action = self.act_helper.press_keys(['k'], action)  # Heavy attack
        elif cycle < 225:
            # Jump and attack
            action = self.act_helper.press_keys(['w'], action)
            if cycle % 25 == 0:
                action = self.act_helper.press_keys(['j'], action)
        else:
            # Shield and wait
            action = self.act_helper.press_keys(['g'], action)
            
        # Always add some movement to prevent standing still
        if cycle % 50 == 0:
            action = self.act_helper.press_keys(['w'], action)  # Jump occasionally
            
        if self.step_count % 100 == 0:
            print(f"🏃 Emergency movement (step {self.step_count}, cycle: {cycle})")
            print(f"   Current action: {action}")
            
        return action

    def _debug_observation(self, obs):
        """Debug information about the observation"""
        if obs is not None and hasattr(obs, 'shape'):
            print(f"👀 Observation shape: {obs.shape}")
            print(f"👀 Observation type: {type(obs)}")
            if hasattr(obs, 'dtype'):
                print(f"👀 Observation dtype: {obs.dtype}")
            
            # Print some key observation values if available
            if isinstance(obs, np.ndarray) and obs.size > 10:
                print(f"👀 First 10 obs values: {obs[:10]}")
        
    def predict(self, obs):
     if not self._initialized:
        self._initialize()
    
     self.step_count += 1
    
    # If no model, use emergency movement
     if self.model is None:
        return self._emergency_movement()
        
     try:
        # Get model prediction - this outputs probabilities for each action dimension
        action_probs, _states = self.model.predict(obs, deterministic=True)
        
        # 🚨 CRITICAL FIX: Handle multi-dimensional action space
        # For MultiDiscrete/MultiBinary, each action dimension is independent
        # Use threshold for each dimension separately
        binary_action = (action_probs > 0.15).astype(np.float32)
        
        # Debug output
        if self.step_count % 100 == 0:
            print(f"🤖 Model prediction (step {self.step_count})")
            print(f"   Raw probabilities: {action_probs}")
            print(f"   Binary action: {binary_action}")
            print(f"   Probabilities sum: {np.sum(action_probs):.2f}")
            
            # Show which keys are being pressed
            key_mapping = ['a', 'd', 'w', 's', 'j', 'k', 'l', 'g', 'h', 'space']
            active_keys = []
            for i, pressed in enumerate(binary_action):
                if pressed > 0 and i < len(key_mapping):
                    active_keys.append(key_mapping[i])
            print(f"   Active keys: {active_keys}")
        
        return binary_action
        
     except Exception as e:
        print(f"❌ Prediction error: {e}")
        return self._emergency_movement()

    def _convert_model_action_to_keys(self, model_action):
        """
        Convert the model's numerical action output to key presses
        This is crucial - the model outputs action indices, not direct key presses
        """
        if model_action is None:
            return self._emergency_movement()
            
        # If action is a single number (discrete action space)
        if isinstance(model_action, (int, np.integer)) or (hasattr(model_action, 'shape') and model_action.shape == ()):
            action_idx = int(model_action)
            return self._action_from_index(action_idx)
        
        # If action is a probability distribution (multiple actions)
        elif hasattr(model_action, '__len__') and len(model_action) > 1:
            # Take the action with highest probability
            action_idx = np.argmax(model_action)
            return self._action_from_index(action_idx)
        
        else:
            print(f"🤔 Unknown action format: {model_action}")
            return self._emergency_movement()
    
    def _action_from_index(self, action_idx):
        """Map action index to actual key combinations"""
        action = self.act_helper.zeros()
        
        # Define action mappings - adjust these based on your action space
        action_mappings = {
            0: [],  # No action
            1: ['a'],  # Left
            2: ['d'],  # Right
            3: ['w'],  # Jump
            4: ['s'],  # Crouch
            5: ['j'],  # Light attack
            6: ['k'],  # Heavy attack
            7: ['l'],  # Special
            8: ['g'],  # Shield
            9: ['a', 'j'],  # Left + attack
            10: ['d', 'j'],  # Right + attack
            11: ['w', 'j'],  # Jump + attack
            12: ['a', 'k'],  # Left + heavy
            13: ['d', 'k'],  # Right + heavy
            14: ['w', 'k'],  # Jump + heavy
        }
        
        if action_idx in action_mappings:
            keys = action_mappings[action_idx]
            if keys:
                action = self.act_helper.press_keys(keys, action)
                if self.step_count % 100 == 0:
                    print(f"🎮 Executing action {action_idx}: {keys}")
        
        return action

    def save(self, file_path: str) -> None:
        if self.model is not None:
            self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        if self.model is None:
            self.model = PPO("MlpPolicy", env, verbose=1)
        else:
            self.model.set_env(env)
            
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
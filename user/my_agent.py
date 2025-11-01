import os
import gdown
from typing import Optional
import numpy as np
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

class SubmittedAgent(Agent):
    def __init__(self, file_path: str = None):
        super().__init__(file_path)
        self.model = None
        self.step_count = 0
        self._initialized = False
        self.last_action = None
        print(f"🎯 SubmittedAgent created with file: {file_path}")

        # To run a TTNN model, you must maintain a pointer to the device and can be done by 
        # uncommmenting the line below to use the device pointer
        # self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))

    def _initialize(self) -> None:
     if self._initialized:
        return
        
     print(f"🔄 Initializing SubmittedAgent...")
     self.debug_valid_keys()
    
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
                
                # DEBUG: Show action space details
                if hasattr(self.model.action_space, 'n'):
                    print(f"   - Action space size: {self.model.action_space.n}")
                elif hasattr(self.model.action_space, 'shape'):
                    print(f"   - Action space shape: {self.model.action_space.shape}")
                
        except Exception as e:
            print(f"❌ All loading methods failed: {e}")
            self.model = None
     else:
        print(f"❌ Model file not found: {self.file_path}")
        self.model = None
        
     self._initialized = True
     print("✅ SubmittedAgent initialization complete")

    def debug_valid_keys(self):
     print("🔑 VALID KEYS DEBUG:")
     try:
        # Test each key to see which ones work
        valid_keys = []
        test_keys = ['w', 'a', 's', 'd', 'space', 'h', 'l', 'j', 'k', 'g', 'f']
        
        for key in test_keys:
            try:
                test_action = self.act_helper.zeros()
                self.act_helper.press_keys([key], test_action)
                valid_keys.append(key)
                print(f"   ✅ '{key}' is valid")
            except KeyError:
                print(f"   ❌ '{key}' is NOT valid")
        
        print(f"📋 FINAL VALID KEYS: {valid_keys}")
        return valid_keys
     except Exception as e:
        print(f"❌ Debug keys failed: {e}")
        return []

    def _emergency_movement(self):
     self.step_count += 1
    
     action = self.act_helper.zeros()
    
    # More dynamic movement pattern using only valid keys
     cycle = self.step_count % 120  # Shorter cycle for more responsiveness
    
     if cycle < 30:
        # Move right and attack frequently
        action = self.act_helper.press_keys(['d'], action)
        if cycle % 8 == 0:  # More frequent attacks
            action = self.act_helper.press_keys(['j'], action)
     elif cycle < 60:
        # Move left and use heavy attack
        action = self.act_helper.press_keys(['a'], action)
        if cycle % 10 == 0:
            action = self.act_helper.press_keys(['k'], action)  # Heavy attack
     elif cycle < 90:
        # Use space for jump with attack
        action = self.act_helper.press_keys(['space'], action)
        if cycle % 12 == 0:
            action = self.act_helper.press_keys(['j'], action)
     else:
        # Use 'w' and attacks (since 'w' is available)
        action = self.act_helper.press_keys(['w'], action)
        if cycle % 15 == 0:
            action = self.act_helper.press_keys(['l'], action)  # Special attack
    
    # Debug output
     if self.step_count % 100 == 0:
        print(f"🆘 EMERGENCY: Using fallback movement (cycle {cycle})")
        
     return action
    
    def _action_from_index(self, action_idx):
     action = self.act_helper.zeros()
    
    # Define action mappings - ONLY USE VALID KEYS: ['w', 'a', 's', 'd', 'space', 'h', 'l', 'j', 'k', 'g']
     action_mappings = {
        # Basic movements
        0: [],           # No action
        1: ['a'],        # Left
        2: ['d'],        # Right
        3: ['space'],    # Jump 
        4: ['s'],        # Crouch
        5: ['w'],        # Use 'w' since it's available in your keys
        
        # Attacks
        6: ['j'],        # Light attack
        7: ['k'],        # Heavy attack
        8: ['l'],        # Special
        9: ['h'],        # Use 'h' since it's available
        10: ['g'],       # Shield
        
        # Combined actions (limit to 2 keys max)
        11: ['a', 'j'],       # Left + attack
        12: ['d', 'j'],       # Right + attack
        13: ['space', 'j'],   # Jump + attack
        14: ['a', 'space'],   # Left + jump
        15: ['d', 'space'],   # Right + jump
        16: ['w', 'j'],       # 'w' + attack
        17: ['a', 'k'],       # Left + heavy
        18: ['d', 'k'],       # Right + heavy
     }
     
     if action_idx in action_mappings:
        keys = action_mappings[action_idx]
        if keys:
            try:
                action = self.act_helper.press_keys(keys)
                if self.step_count % 100 == 0:
                    print(f"🎮 Executing action {action_idx}: {keys}")
            except KeyError as e:
                print(f"⚠️ Invalid keys {keys} in action {action_idx}: {e}")
                return self._emergency_movement()
     else:
        print(f"⚠️ Unknown action index: {action_idx}, using emergency movement")
        return self._emergency_movement()
    
     return action

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path
    
    def _emergency_recovery(self, obs):
        self.step_count += 1
    
        action = self.act_helper.zeros()
    
        if obs is not None and len(obs) >= 2:
            player_x = obs[0]
            player_y = obs[1]
            
            # Check if falling and below platform level
            if player_y < 0.5:  # Below platform level
                # Determine which platform to recover to
                if player_x < 0:  # On left side, recover to left platform
                    action = self.act_helper.press_keys(['d', 'space'], action)  # Move right + jump (using space)
                    print("🆘 EMERGENCY: Recovering to left platform")
                else:  # On right side, recover to right platform
                    action = self.act_helper.press_keys(['a', 'space'], action)  # Move left + jump (using space)
                    print("🆘 EMERGENCY: Recovering to right platform")
            else:
                # Use normal emergency movement
                return self._emergency_movement()
            
        return action

    def predict(self, obs):
     if not self._initialized:
        self._initialize()

     self.step_count += 1
     
     if obs is not None and len(obs) >= 2:
        player_y = obs[1]
        # If falling below critical level, use emergency recovery
        if player_y < -2.0:
            print("🚨 CRITICAL: Agent falling, activating emergency recovery!")
            return self._emergency_recovery(obs)
    
     if self.model is None:
        return self._emergency_movement()
        
     try:
        action, _states = self.model.predict(obs, deterministic=True)
        
        # FIX: Handle different action formats safely
        if isinstance(action, (int, np.integer)):
            # Discrete action space - single number
            return self._action_from_index(int(action))
        elif hasattr(action, 'shape') and action.shape == ():
            # Single number in array format
            return self._action_from_index(int(action))
        elif isinstance(action, (list, np.ndarray)):
            # Multi-discrete or continuous actions
            if len(action) == 1:
                # Single action in array
                return self._action_from_index(int(action[0]))
            else:
                # Multiple actions - use the new safe conversion
                return self._convert_to_key_presses_safe(action)
        else:
            print(f"🤔 Unknown action format: {type(action)} - {action}")
            return self._emergency_movement()
        
     except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return self._emergency_movement()

    def _convert_to_key_presses_safe(self, action_array):
     action = self.act_helper.zeros()

     key_mappings = [
        ['a'],       # 0: Left
        ['d'],       # 1: Right  
        ['space'],   # 2: Jump 
        ['s'],       # 3: Crouch
        ['j'],       # 4: Light attack
        ['k'],       # 5: Heavy attack
        ['l'],       # 6: Special
        ['g'],       # 7: Shield
        ['h'],       # 8: Other action
        ['w'],       # 9: Use 'w' instead of 'f' since it's available
     ]
    
    # Handle different action array types
     if isinstance(action_array, np.ndarray):
        if action_array.dtype == np.int64 or action_array.dtype == np.int32:
            # Discrete actions - use the action index
            action_idx = int(action_array[0]) if len(action_array) == 1 else 0
            if action_idx < len(key_mappings):
                keys = key_mappings[action_idx]
                try:
                    action = self.act_helper.press_keys(keys)
                    if self.step_count % 100 == 0:
                        print(f"🎮 Discrete action {action_idx}: {keys}")
                except KeyError as e:
                    print(f"⚠️ Invalid key in mapping: {e}, using emergency")
                    return self._emergency_movement()
            else:
                print(f"⚠️ Action index {action_idx} out of range, using emergency")
                return self._emergency_movement()
        else:
            # Continuous/probability actions
            activated_actions = 0
            for i in range(min(len(action_array), len(key_mappings))):
                if action_array[i] > 0.1:  # Lower threshold
                    keys = key_mappings[i]
                    try:
                        action = self.act_helper.press_keys(keys, action)
                        activated_actions += 1
                        if self.step_count % 100 == 0 and activated_actions <= 3:
                            print(f"🎮 Action {i}: {keys} (value: {action_array[i]:.2f})")
                    except KeyError as e:
                        print(f"⚠️ Invalid key {keys}: {e}, skipping this action")
                        continue
            
            # If no actions activated, use emergency
            if activated_actions == 0:
                return self._emergency_movement()
    
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
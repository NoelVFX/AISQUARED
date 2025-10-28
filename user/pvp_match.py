from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent
from user.my_agent import SubmittedAgent
import pygame
pygame.init()
import os

# Check if the model file exists
model_path = "C:/Users/Anson/AISQUARED-1/checkpoints/symmetric_training/rl-model.zip"
print(f"Model exists: {os.path.exists(model_path)}")

# If file doesn't exist, try to find it in other locations
if not os.path.exists(model_path):
    print("❌ Model file not found at specified path. Searching for alternatives...")
    # Try relative path
    alt_path = "checkpoints/advanced_training_v2/rl-model.zip"
    if os.path.exists(alt_path):
        model_path = alt_path
        print(f"✅ Found model at: {alt_path}")
    else:
        print("❌ Model not found in alternative locations")

my_agent = UserInputAgent()

# Input your file path here in SubmittedAgent if you are loading a model:
opponent = SubmittedAgent(file_path=model_path)

match_time = 99999

# Run a single real-time match
run_real_time_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * 999990000,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)
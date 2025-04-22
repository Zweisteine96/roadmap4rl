import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


# load the model
if False:
    # simply to test the gymnasium installation
    env = gymnasium.make('Pendulum-v1', render_mode="rgb_array", g=9.81)
    print(f"made a simple pendulum env")
r"""
More details: https://gymnasium.farama.org/main/environments/mujoco/ant/#arguments
NOTE:
    1. tweaking the env simulation params
    - #reset_noise_scale: To avoid overfitting the policy, 
    this should be set to a value appropriate to the size of the robot, 
    we want the value to be as large as possible without 
    the initial distribution of states being invalid.
    - #frame_skip: dt = frame_skip * model.opt.timestep (integrator time in the MJCF model file),
    Go1 has an integrator timestep as 0.002, select frame_skip as 25, set the value of dt to 0.05s.
    - #max_episode_steps: determines the number of steps per episode before truncation.

    2. tweaking the env termination params
    - #healthy_z_range:  to terminate the environment when the robot falls over, 
    or jumps really high, 
    here we have to choose a value that is logical for the height of the robot.
    - #terminate_when_unhealthy: set as False to disable termination altogether, 
    which is not desirable in the case of Go1.

    3. tweaking the env reward params
    - #forward_reward_weight
    - #ctrl_cost_weight:
    - #contact_cost_weight
    - #healthy_reward
    - #main_body: indicates trunk or torso
"""
print(f"initializing the environment...")
env = gym.make(
    'Ant-v5',
    xml_file='../mujoco_menagerie/unitree_go1/scene.xml',
    render_mode='human',
    forward_reward_weight=1,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=1,
    main_body=1,
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,
    max_episode_steps=1000,
)
print(f"env initialized!")

print(f"start training...")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
print(f"training done!")

print(f"replaying the policy...")
""" vec_env = model.get_env()
obs, _ = vec_env.reset()
for i in range(1000):
    print(f"iteration: {i}")
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = vec_env.step(action)
    vec_env.render() """
obs, _ = env.reset()
for i in range(100000):
    print(f"iteration: {i}")
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    """ if terminated or truncated:
        break """

env.close()
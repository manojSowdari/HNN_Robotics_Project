# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3.8 - AzureML
#     language: python
#     name: python38-azureml
# ---

# +
import pybullet as p
import pybullet_data
import numpy as np
import time

# Use DIRECT mode to avoid graphics
physicsClient = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)

plane = p.loadURDF("plane.urdf")

# If you don't have "two_link_arm.urdf" yet, consider using a built-in robot for testing
# For example, you can replace this with "kuka_iiwa/model.urdf" from pybullet_data
robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0.1])

num_steps = 1000
dt = 1.0 / 240.0
data = []

for i in range(num_steps):
    p.stepSimulation()
    # time.sleep(dt) # Not strictly necessary in DIRECT mode
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    # For demonstration, let's just collect the first joint's data
    q = joint_states[0][0]  # angle
    dq = joint_states[0][1] # velocity
    data.append([q, dq])

data = np.array(data)
np.save("robot_arm_data.npy", data)
p.disconnect()

print("Simulation complete. Data saved to 'robot_arm_data.npy'")
# -



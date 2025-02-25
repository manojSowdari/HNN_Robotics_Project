How to Run the Project
Simulation and Data Collection:

Run the simulation script (e.g., single-joint-system.py) to simulate the robotic system in headless mode and generate trajectory data.
The script uses PyBullet in DIRECT mode and saves the recorded states to control_loop_output.npy.
Control Loop:

The control loop applies a naive torque search to drive a selected joint toward a desired angle.
You can modify the q_des (desired joint angle) and the torque candidate range to experiment with different control strategies.
Visualizing the Data:

Use provided or custom Python scripts/notebooks to load the .npy file, analyze, and plot the joint angle and velocity over time with Matplotlib.
Training an HNN (Future Work):

Although this project focuses on simulation and control, the next steps could include training a Hamiltonian Neural Network on the collected simulation data to learn the system’s dynamics.
Pushing Code to GitHub
We set up Git on the Azure compute instance, configured safe directory settings, and pushed our code using a Personal Access Token (PAT). The basic commands we used were:

bash
Copy code
git init
git add .
git commit -m "Initial commit from Azure ML compute instance"
git remote add origin https://github.com/yourusername/HNN_Robotics_Project.git
git push -u origin master
For renaming branches or force pushing, please refer to the Git documentation.

Future Work
Advanced Control:
Integrate the HNN with the control loop for model-based control.
Improved Simulation:
Experiment with different URDFs, tuning joint parameters, and using symplectic integrators.
Experiment Tracking:
Use Azure ML experiment tracking to log metrics and compare different control strategies.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
PyBullet for the simulation engine.
Azure Machine Learning for providing the cloud environment.
GitHub for version control and collaboration

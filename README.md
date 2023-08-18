# Machine Learning Based Control of Living Cells
This GitHub repository showcases my technical project that I completed during my Master's in Engineering Mathematics in Univeristy of Bristol. This project focused on exploring the application of Proximal Policy Optimization (PPO) in the field of Cybergenetics control, more specifically the aim was to showcase the potential of Model-Free Reinforcement Learning (MFRL) techniques to control complex systems in the domain of Cybergenetics.


## Contents
The `Environment` folder houses the files related to the custom Gym environment and the training of a Proximal Policy Optimization (PPO) agent using that environment. The `GTS_Environment.py` file is a custom OpenAI Gym environment designed specifically for the Genetic Toggle SWitch (GTS). This environment encapsulates the dynamics and interactions within the GTS and provides a structured interface for reinforcement learning agents to interact with and learn from. In the `Run_GTS_Environment.py` file, a PPO agent is trained on the custom gym environment using the stable-baselines3 library's PPO algorithm. This script demonstrates the application of reinforcement learning in the context of controlling the Genetic Toggle Switch.

## Goals of the study
The primary objective of this study was to evaluate the performance of the PPO algorithm in the context of Cybergenetics. PPO, a reinforcement learning technique, was employed to control a complex and unstable Genetic Toggle Switch (GTS) for extended durations. The project's focus on the control of dynamic and unpredictable systems is crucial in assessing the feasibility of using MFRL techniques in real-world applications. To assess the feasibility of PPO, the agent was evaluated against an untrained PPO, a Proportional-Integral-Derivative (PID) and relay controller. Integral Squared Error (ISE) was employed as a performance metric to assess the control capabilities of the controllers.

* Evaluate the performance of PPO in controlling a deterministic GTS.
* Compare PPO's performance with PID and relay controllers.
* Analyze the implications of MFRL in Cybergenetics.
* Identify potential areas for improvement in MFRL techniques.

## Conclusion Highlights
Through a comprehensive evaluation, the project compared the performance of the PPO agent against other controllers, including an untrained PPO agent, a PID controller, and a relay controller. While the PID and relay controllers exhibited better performance in terms of Integral of Squared Error (ISE), the trained PPO agent showcased notable improvement from its untrained state. This suggests that while MFRL shows promise, further refinement and exploration are essential before practical implementation in Cybergenetics.


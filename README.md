# Machine Learning Based Control of Living Cells
This GitHub repository showcases my technical project that I completed during my Master's in Engineering Mathematics in Univeristy of Bristol. This project focused on exploring the application of Proximal Policy Optimization (PPO) in the field of Cybergenetics control, more specifically the aim was to showcase the potential of Model-Free Reinforcement Learning (MFRL) techniques to control complex systems in the domain of Cybergenetics.


## Contents
The `Environment` folder houses the files related to the custom Gym environment and the training of a Proximal Policy Optimization (PPO) agent using that environment. The `GTS_Environment.py` file is a custom OpenAI Gym environment designed specifically for the Genetic Toggle SWitch (GTS). This environment encapsulates the dynamics and interactions within the GTS and provides a structured interface for reinforcement learning agents to interact with and learn from. In the `Run_GTS_Environment.py` file, a PPO agent is trained on the custom gym environment using the stable-baselines3 library's PPO algorithm. The PPO agent learns from interactions with the `GTS_Environment` and adapts its policy to optimize its performance in the GTS scenario. The training process demonstrates the potential of reinforcement learning techniques in tackling complex and dynamic control challenges.

The `Controllers` folder contains implementations of different controllers used to manage and control the Genetic Toggle Switch (GTS). The `PID.py` file holds the implementation of a Proportional-Integral-Derivative (PID) controller. This classic control technique is widely used for its ability to maintain stability and regulate systems by adjusting control signals based on proportional, integral, and derivative terms. In this implementation, manual tuning techniques were applied to configure the PID parameters to effectively control the GTS. The `Relay.py` file contains the implementation of a relay controller. A relay controller operates by switching between two states based on predefined thresholds. This control approach is often suited for systems that require on-off switching action rather than continuous adjustments. 

The `Plots` folder contains three files that plots the dynamics of the GTS and the performance of the four controllers. The `GTS_Simulation.py` file performs a simulation of the deterministic GTS model. By running this script, the behavior of the GTS under controlled conditions can be observed. This simulation is instrumental in verifying the correctness of the model and ensuring that it operates as expected. The 'ISE.py' file generates plots illustrating the Integral of Squared Error (ISE) for the different controllers utilised in this project. It compares the performance of the untrained PPO, PPO, PID, and Relay controllers based on their ISE values. Finally the 'Nullclines.py' file produces nullcline plots for the deterministic GTS model. The code indentifies the stable and unstable equilibrium points within the system. These plots aid in comprehending the behavior and stability characteristics of the GTS.

## Goals of the study
The primary objective of this study was to evaluate the performance of the PPO algorithm in the context of Cybergenetics. PPO, a reinforcement learning technique, was employed to control a complex and unstable Genetic Toggle Switch (GTS) for extended durations. The project's focus on the control of dynamic and unpredictable systems is crucial in assessing the feasibility of using MFRL techniques in real-world applications. To assess the feasibility of PPO, the agent was evaluated against an untrained PPO, a Proportional-Integral-Derivative (PID) and relay controller. Integral Squared Error (ISE) was employed as a performance metric to assess the control capabilities of the controllers.

* Evaluate the performance of PPO in controlling a deterministic GTS.
* Compare PPO's performance with PID and relay controllers.
* Analyze the implications of MFRL in Cybergenetics.
* Identify potential areas for improvement in MFRL techniques.


## Genetic Toggle Switch
(GTS) is a bi-stable GRN and is widely employed in complicated synthetic circuitry when bi-stability, memory, or binary signal processing is desired, as it is a fundamental topology in core natural gene regulation networks and one of the foundational results of synthetic biology. Therefore, the GTS was chosen as the subject of the control tests conducted in this project. 

$$
\frac{dmRNA_{TetR}}{dt}=k_{T}^{m0}+\frac{k_{T}^{m}}{1+(\frac{LacI}{\theta _{LacI}}\times \frac{1}{1+(\frac{IPTG}{\theta _{IPTG}})^{\eta _{IPTG}}})^{\eta _{LacI}}}-g_{T}^{m}\times mRNA_{TetR}
$$

$$
\frac{dmRNA_{LacI}}{dt}=k_{L}^{m0}+\frac{k_{L}^{m}}{1+(\frac{TetR}{\theta _{TetR}}\times \frac{1}{1+(\frac{aTC}{\theta _{aTC}})^{\eta _{aTC}}})^{\eta _{TetR}}}-g_{L}^{m}\times mRNA_{LacI}
$$

$$
\frac{d\mathit{LacI}}{dt} = k_{L}^{P}\times mRNA_{LacI} - g_{L}^{P}\times LacI
$$

$$
\frac{d\mathit{TetR}}{dt} = k_{T}^{P}\times mRNA_{TetR} - g_{T}^{P}\times TetR
$$

## Conclusion Highlights
Through a comprehensive evaluation, the project compared the performance of the PPO agent against other controllers, including an untrained PPO agent, a PID controller, and a relay controller. While the PID and relay controllers exhibited better performance in terms of Integral of Squared Error (ISE), the trained PPO agent showcased notable improvement from its untrained state. This suggests that while MFRL shows promise, further refinement and exploration are essential before practical implementation in Cybergenetics.


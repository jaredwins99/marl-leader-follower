# Multi-Agent RL: Probing the Neural Codes of Reward Learning 
**Part of the Columbia Data Science Institute Scholars program and in collaboration with the Systems Intelligence Lab**

*STATUS: Pre-LMM*

**Note**: This work contributed to a larger research article, now available as a preprint: https://pmc.ncbi.nlm.nih.gov/articles/PMC12407837/.

---

## Multi-Agent Independent Learning

In parallel to an empirical neuroscience experiment with mice, we trained two RL agents on a similar cooperation task to study the learning process and discover potential emergent properties. 
Our goal was to forward-model our RL  agents while being blinded to the particulars of the experimental data, and specifically seek asymmetries that result from the learning process as opposed to modeled asymmetries in our agents. 

<img width="480" height="480" alt="animation" src="https://github.com/user-attachments/assets/ae10cd7b-9747-45c1-9743-dba4165efc2d" />

---

## Independent Q-Learning: Gauss-Seidel Iteration

Q-Learning is a model-free, tabular method for reinforcement learning. 
Agents learn through experiences, as opposed to knowing the environmental dynamics. 
State-specific expected sums of rewards are updated in an online way according to a bootstrapped target. 
To manage multiple agents, the Gauss-Seidel algorithm iterates between updating the Q-values for different individual agents.

<br>

<img width="880" height="450" alt="unnamed" src="https://github.com/user-attachments/assets/a0bacf69-b1c8-462f-8bf1-7cd553225805" />

<br>

What we see here is that each agent models both its own reward function and the other's, yielding four models total: all four are isomorphic.

---

## Incremental Learning

Even for simple tasks, RL agents can struggle to learn cause and effect. 
Key-and-door problems in particular are difficult to master. 
To overcome this challenge, we broke learning down into separate regimes, helping the agents learn faster and more effectively. 
In this case, the agents can be taught to first activate the reward, next find fixed rewards, then find random rewards–stepwise and not all at once. 

---

## Results

Paper's conclusion: MARL revealed that behavioral asymmetries and social roles (initiator, leader) can emerge spontaneously among initially identical agents from environmental demands, likely driven by small stochastic differences in learning.
My own read: There is role differentiation, but not to the extent of mice. The agent "roles" are statistical asymmetries in a 2-agent task; mouse roles involve neuromodulation, richer behavioral repertoire, and well, being a mouse.

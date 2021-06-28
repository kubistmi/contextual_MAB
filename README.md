# Contextual multi-armed bandit
This repo contains a review of the contextual multi-armed bandits.  
Includes proposed framework for extendible building blocks that form the contextual bandit problem.  
You can find the overview of the contextual bandits, dataset, and the framework in the [presentation](presentation.ipynb)

## Structure
- src/ contains the modules of the framework
  - agent, environment, oracle, policy, data provider
- [train_manual](train_manual.ipynb) demonstrates how to combine these modules to form a contextual bandit
- [train_grid](train_grid.py) allows you to run the model for a specified parameter grid
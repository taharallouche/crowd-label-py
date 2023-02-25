# Truth-tracking via Approval Voting: Size Matters!
## Tahar Allouche, Jérôme Lang, Florian Yger
### AAAI-2022

**Paper Abstract ([paper available here](https://arxiv.org/pdf/2112.04387.pdf)):**\
Epistemic social choice aims at unveiling a hidden ground
truth given votes, which are interpreted as noisy signals about
it. We consider here a simple setting where votes consist
of approval ballots: each voter approves a set of alternatives which they believe can possibly be the ground truth.
Based on the intuitive idea that more reliable votes contain
fewer alternatives, we define several noise models that are approval voting variants of the Mallows model. The likelihood maximizing alternative is then characterized as the winner of
a weighted approval rule, where the weight of a ballot decreases with its cardinality. We have conducted an experiment
on three image annotation datasets; they conclude that rules
based on our noise model outperform standard approval voting; the best performance is obtained by a variant of the Condorcet noise model.


**Repository:**\
This repository contains the python code and datasets used in the Experiments section in the paper. 

**Datasets:**\
The data that we used was originally collected in:
>Shah, Nihar, Dengyong Zhou, and Yuval Peres. "Approval voting and incentives in crowdsourcing." International conference on machine learning. PMLR, 2015.

The above paper designs mechanisms to incentivize crowdsourcing workers to answer truthfully in approval voting settings.

**Code:**\
Here we succintly present the main function in the [python file](experiments.py):
- `compare_methods(n_batch, data)`: This function compare the accuracy of different aggregation methods across the chosen dataset for different number of voters.
It samples *n_batch* batches for each number of voters, and average the precision over them.

To run the experiments, run the command:\
`python3 src/experiments.py`\
You will be asked to select the *dataset* (animals, textures or languages) and the *number of batches*.

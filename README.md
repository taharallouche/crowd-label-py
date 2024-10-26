# :mage_man: hakeem :mage_man:

This python package serves two purposes:
1. Ensures the reporucibility of my [AAAI-2022 paper](https://ojs.aaai.org/index.php/AAAI/article/view/20403)'s experiments, introducing the **vote-size-matters** crowdsourcing data labelling method. 📚🧪📊
2. Enable you to apply the data labelling method to your own datasets.🛠️🗃️


## The vote-size-matters collective labelling method
If you possess an unlabeled dataset comprising 📷 images, 🔊 sounds, 🎥 videos, or ✉️ texts, and you have collected some crowdsourced annotations with the aim of aggregating them optimally to deduce the correct label for each instance, then `hakeem` is the solution you're seeking! 🚀 

The package implements the size-matters truth tracking principle, 💡 which has consistently shown superior performance compared to other voter-agnostic aggregation rules :chart_with_upwards_trend:. One notable advantage of this method is its reliance on a simple intuition, making the results it produces entirely explainable! :dart:🌟

In fact, the method's key principles include:
1. Granting hesitant voters the flexibility to select more than one possible label. 🤔🔄
2. Relying on mathematically proven [payment schemes](https://proceedings.mlr.press/v37/shaha15.html) to ensure sincerity of voters.📊✅
3. Assigning greater weight to voters who choose fewer labels. After all, a voter familiar with the correct label would likely choose that option, whereas a voter who selects too many labels probably doesn't know the correct answer.⚖️

Various weighting schemes are provided to the user, with each one being optimal under different assumptions. The choice of the right scheme is yours to make!

## First guide: How to reproduce the paper's results
After cloning the repo, you should:
1. install [uv](https://docs.astral.sh/uv/).
2. run the following command in the terminal:
 ```bash session 
 uv run make reproduce
 ```
  This will run the python script for comparing the aggregation rules and saving the results. You will be asked to enter your preferences, like choosing the dataset, the number of voters and the number of simulations.


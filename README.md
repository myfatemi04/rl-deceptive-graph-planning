# Reinforcement Learning with Graph Neural Networks Enables Zero-Shot Deceptive Path Planning Over Arbitrary Graphs

August 2023

This repository holds the code for our paper, "Reinforcement Learning with Graph Neural Networks Enables Zero-Shot Deceptive Path Planning Over Arbitrary Graphs", which we are submitting to AAMAS 2024.

## Installation

Run `pip install -r requirements.txt` to install the required packages.

## Training a model

Run the file `train_for_deceptiveness.py` to train a model. This will output a model to the `checkpoints` folder, which you can use in experiment.

## Testing the model

We have populated some models in the `models/sage_ambiguity_2` and `models/sage_exaggeration_4` folders. You can render animations of their performance or compare different levels of deceptiveness statically by running `continuous_sim.py`.

For example:
```
python3 continuous_sim.py --deception-type exaggeration --action animate --seed=51
```

Renders an animation of an exaggeration-tuned model on a random graph with seed 51 to the file `animation.mp4`.

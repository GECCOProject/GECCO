# GECCO
## About
GECCO is a _lightweight_ image classifier based on single MLP and graph convolutional layers. We find that our model can achieve up to 16x better latency than other state-of-the-art models. The paper for our model can be found at https://arxiv.org/abs/2402.00564
## Reproducibility
To reproduce the results in the GECCO paper, the practitioner should change the featurelength (hidden size) of the model in the model.py file. For MNIST, MSTAR, use featurelength of 32, MSTAR, use a featurelength of 48, and CXR, use a featurelength of 56. 
Additionally, update the directories in the code to match the datasets in your own workspace.

# Analysis of Open Vocabulary Detection (OVD) and Few-Shot Detection (FSD) for Remote Sensing

TO-DO

* Embedding training and detection use different box formats. Fix it to use always one or to choose the right one
* Add optional validation set for training
* Take each DOTA subset and generate prototypes with its images
* Make sure prototypes are well computed (extract features just like at model level)
* Evaluate DINO+RPN on each subset and plot the curve
* Prepare a framework to train embeddings (prototypes and bg_prototypes) to classify boxes and handle background examples (bad proposals)
* Test learned embeddings on the evaluation set
* Fine-tune OWL-ViT on the different sets and add its curve to the plot
* Implement CLIP+RPN (and RemoteCLIP) approach by selecting the backbone
* Try sous-échantillonnage
* Look into how to train/fine-tune RPN on a different dataset of base classes to be more suitable for remote sensing images

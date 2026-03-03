# SketchRNN

# Summary
This is a pytorch implementation of SketchRNN, a variational autoencoder based architecture used for modeling sequential stroke based drawing. The goal is to predict the next storke using the previous stroke conditioned on the encoder latent output. This experiment was trained on one class from google quickdraw dataset and achieve good output for both sketch reconstruction and generation. 
# Table of Contents
- [Repo Content](#repo-content)
- [Key Features](#key-features)
- [Result](#result)
- [Lesson Learnt and Troubleshooting](#lesson-learnt-and-troubleshooting)
- [Limitation](#limitation)
# Repo Content

# Features

# Result
## Dataset 
The cat dataset on google quickdraw was used to train the model. There are a total of ~120,000 sample was used. the quickdraw format was converted to sketchrnn format, and the pen movement (dx and dy) was normalized their joint standard deviation. Any sketch that are too long or too short are drop by considering them as outlier using IQR percentile. 

## train and validation loss

# Lesson Learnt and Troubleshooting
## Posterior Collapse
## Overiftting and Generalization
## custom lstm 

# Limitation

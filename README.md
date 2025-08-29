# CIFLE: cycle-informed fatigue-life estimation

## Project overview
- CIFLE consists of three modules that work in sequence  
  - **M1** synthesizes representative hysteresis loops from basic LCF inputs  
  - **M2** extracts physically interpretable fatigue parameters from each loop  
  - **M3** predicts the fatigue life fraction and its uncertainty from these parameters

This repository provides Python source code and configuration files.  
A public CSV with M2 features for M3 training is also provided.

## Model development
`/model`  
Contains scripts for modules M1, M2, and M3.

## Analysis
`/analysis`  
Scripts for model analysis including Shapley value analysis and uncertainty contours.

## Installation
Python 3.12 or newer is recommended.

## Citation
If you use this code or model in your research, please cite our work  
(citation details to be added upon publication).

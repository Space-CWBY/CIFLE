# CIFLE: cycle-informed fatigue-life estimation framework

## Project overview
- CIFLE consists of three modules that work in sequence

  - **M1** synthesizes representative hysteresis loops from basic LCF inputs  
  - **M2** extracts physically interpretable fatigue parameters from each loop  
  - **M3** predicts the fatigue life fraction and its uncertainty from these parameters

This repository provides Python source code and configuration files.  
A public CSV with M2 features for M3 training is also provided.

## Model development
```
/model
```

Contains scripts for modules M1, M2, and M3.

## Analysis
```
/analysis
```  

Scripts for model integration, deployment, and interpretability analysis.

- **`CIFLE.py`**: The main orchestration script. It integrates M1 (synthesis), M2 (feature extraction), and M3 (prediction) to perform the full fatigue life estimation pipeline.
- Scripts for model analysis including Shapley value analysis and uncertainty contours.

## Usage
The `CIFLE.py` script integrates all modules to estimate fatigue life bounds and generate probability maps. You can run it via the command line:

```bash
python analysis/CIFLE.py \
    --ann_model "path/to/m1_ann.h5" \
    --ann_stats "path/to/m1_mean_std.csv" \
    --bnn_model "path/to/m3_bnn.pt" \
    --bnn_train_data "path/to/calculation_results.csv" \
    --raw_data_dir "path/to/raw_loops_directory" \
    --outdir "results"

## Installation
Python 3.12 or newer is recommended.

## Citation
If you use this code or model in your research, please cite our work  
(citation details to be added upon publication).

# ProtoTEx: Explaining Model Decisions with Prototype Tensors 

This repository contains example implementations of Prototype tensors for providing case-based reasoning. It has been forked and modified from the original repository that comes with the paper here: https://utexas.app.box.com/v/das-acl-2022 .


## Quick Start

To run ProtoTEx on the propganda detection task, first we need the following:

```
mkdir Logs
mkdir Models
mkdir artifacts
pip install -r requirements.txt
```

To train the model, use the script `general.sh` that contains all the arg parameters to adjust the training process. 

Run `python main.py --help` to see all the available parameters.

After running the `general.sh` that would train the model, the model will be saved in the `Models` folder. To run the explaratory analysis, run the `inference_and_explanation.sh` script. This will generate all the prototypes and the test / train examples that are close to prototypes in the `artifacts` folder. Other analysis are located in `Notebooks/post_hoc_analysis.ipynb`
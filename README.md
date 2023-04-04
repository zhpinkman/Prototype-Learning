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

After running the `general.sh` that would train the model, the model will be saved in the `Models` folder. There are different scripts such as training and evaluation in the general.sh that you comment or uncomment to run the different parts of the pipeline. To run the explaratory analysis, run the `inference_and_explanation.sh` script. This will generate all the prototypes and the test / train examples that are close to prototypes in the `artifacts` folder. Other analysis are located in `Notebooks/post_hoc_analysis.ipynb`.

glue dataset that is currently in the gitignore file and extract it under `data` directory. You can download it from here: https://drive.google.com/file/d/1XIapOHwt_m5Z5O5VyCkkozltEyApIMLX/view?usp=share_link
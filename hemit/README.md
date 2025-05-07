This directory contains code to reproduce training of HEMIT-ORION, the HEMIT model trained on ORION using original HEMIT codebase https://github.com/BianChang/Pix2pix_DualBranch.

To make it compatible with with ORION dataset (more than 3 channels mIF images, etc.), we modified some scripts. You can find them in adapter_scripts and use diff.txt, to identify where to copy them in original codebase.

`correct_names.py` and `create_split.py` are there to preprocess the ORION dataset to make it compatible with HEMIT training. You need to change global variables to run it.

The weights of the model is available on the huggingface hub: https://huggingface.co/...

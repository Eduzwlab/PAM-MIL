# PAM-MIL
## Continuous PAM50 subtype scores reveal spatial molecular heterogeneity in prostate cancer histology

PAM50 molecular subtyping is clinically important but requires RNA sequencing, limiting routine use. PAM-MIL is a PAM50-informed multiple-instance regression model that predicts continuous PAM50 subtype scores directly from routine whole-slide images, capturing both dominant and admixed subtype composition. Subtype-specific attention maps localize morphologically and molecularly distinct regions within slides, validated by spatial transcriptomics, and reveal that admixed luminal A tumors carry elevated recurrence risk compared to pure luminal A tumors.

!["PAM-MIL"](./assets/PAM-MIL.png)

## Contents
- [Pre-requisites and Environment](#pre-requisites-and-environment)
- [Prepare Patch Features](#prepare-patch-features)
- [K-fold Cross Validation](#k-fold-cross-validation)
- [Heatmaps generation](#heatmaps-generation)
## Pre-requisites and Environment
### Pre-requisites
* Linux (Tested on Ubuntu 24.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 3090) 
* Python (3.10.19), OpenCV (4.9.0), Openslide-python (1.4.1) and Pytorch (2.1.2)

### Environment Configuration
1. Create a virtual environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).
   ```
   $ conda create -n pammil python=3.10.18
   $ conda activate pammil
   $ pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
   ```
      *Note:  `pip install` command is required for Pytorch installation.*


2. To try out the Python code and set up environment, please activate the `wsvit` environment first:

    ``` shell
    $ conda activate pammil
    $ cd pammil/
    ```
3. For ease of use, you can just set up the environment and run the following:
   ``` shell
   $ pip install -r requirements.txt
   ```

## Prepare Patch Features
To preprocess WSIs, we used [CLAM](https://github.com/mahmoodlab/CLAM/tree/master#wsi-segmentation-and-patching).
### Patching
```shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 512 --step_size 512 --preset tcga.csv --seg --patch
```
### Feature Extraction
```shell
python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY 
```
## K-fold Cross Validation
```shell

python main.py --project=$PROJECT_NAME --datasets=tcga --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --cv_fold=5 --title=pammil --seed=1
```

## Heatmaps generation
```shell
python generate_heatmaps.py --data_root $DATASET_PATH --slides_dir $SLIDES_PATH --model_root $MODEL_PATH --label_csv $LABEL_CSV_PATH --output_dir $OUTPUT_PATH --patch_level 0 --thumb_scale 0.125 0.125 --alpha 0.5 --style cividis --dpi 220

```


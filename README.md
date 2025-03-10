# Zero-Shot Industrial Anomaly Segmentation with Image-Aware Prompt Generation
Zero-Shot Anomaly Segmentation by IAP-AS, a model designed to detect anomalies in industrial images, precisely identify their locations, and restore abnormal images to normal using an LLM prompting and zero-shot segmentation techniques. 
![framework](./figures/fig.%202.jpg)

## Requirements & Setup
This codebase utilizes Anaconda for managing environmental dependencies. Please follow these steps to set up the environment:
1. **Download Anaconda:** [Click here](https://www.anaconda.com/download) to download Anaconda.
2. **Clone the Repository:**
Clone the repository using the following command.
   ```bash
   git clone https://github.com/../IAP-ZSAS.git
   ```
3. **Install Requirements:**
   - Navigate to the cloned repository:
     ```bash
     cd IAP-AS
     ```
   - Create a Conda environment from the provided `environment.yaml` file:
     ```bash
     conda env create -f environment.yaml
     ```
   - Activate the Conda environment:
     ```bash
     conda activate iap-as
     ```
This will set up the environment required to run the codebase.

## Datasets
Below are the details and download links for datasets used in our experiments:

[Dataset details]

All datasets should be stored in the `datasets` folder, with each dataset placed in a subfolder named after the dataset (e.g., `datasets/MVTec-AD/`, `datasets/MPDD/`, `datasets/BTAD/`, etc.).

1. **MVTec-AD** [(Download)](https://www.mvtec.com/downloads): The MVTec AD dataset comprises approximately 5,000 images across 15 classes, including texture-related categories such as fabric and wood.
2. **MPDD** [(Download)](https://github.com/stepanje/MPDD): MPDD is a dataset designed for visual defect detection in metal part manufacturing. It contains over 1,000 images.
3. **BTAD** [(Download)](http://avires.dimi.uniud.it/papers/btad/btad.zip): BTAD (beanTech Anomaly Detection) is a dataset of real-world industrial anomalies, consisting of 2,830 images of three industrial products that exhibit body and surface defects.
4. **KSDD1** [(Download)](https://www.vicos.si/resources/kolektorsdd/): The KSDD1 dataset includes 347 normal images and 52 abnormal images, specifically for detecting micro-defects on metal surfaces.
5. **MTD** [(Download)](https://github.com/abin24/Magnetic-tile-defect-datasets.): The MTD dataset contains images of magnetic tiles, featuring various types of defects. 
6. **DTD-Synthetic** [(Download)](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1): DTD-Synthetic is based on the DTD (Describable Texture Dataset) and includes synthesized texture images with anomalies. IIt consists of 47 diverse texture classes.
7. **DAGM** [(Download)](https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html): The DAGM2007 dataset comprises artificially generated images with characteristics similar to real-world problems. It is divided into 10 datasets: six for algorithm development and four for performance evaluation.

These datasets provide valuable resources for our experiments and each known for their high-resolution, texture-rich images that are well-suited for industrial anomaly segmentation.

## Zero-Shot Anomaly Segmentation (ZSAS) TEST
Replace `<dataset>` with one of the following options: `mvtec`, `ksdd`, `mtd`.

Replace `<model>` with one of the following options: `base`, `iap_zsas`.

```python
python test_zsas.py --dataset <dataset name> --model <model name> 
```
This command excel our proposed model for zero-shot anomaly segmentation(ZSAS) on the specified dataset using the selected model, with best configurations loaded, running 10 epochs each.

## Ablation Study TEST
Ablatin study on MVTec-AD texture dataset.
```python
python test_ablation.py --image True --prompt True --filter True 
```

#### Optional arguments
```
  --gpu                             gpu number
  --dataset                         dataset name
  --model                           model name
  --box_threshold                   GroundingSAM box threshold
  --text_threshold                  GroundingSAM text threshold
  --size_threshold                  Bounding-box size threshold
  --iou_threshold                   IoU threshold
  --random_img_num                  random image extraction number
  --eval_resolution                 Description of evaluation resolution
  --exp_idx                         Description of experiment index
  --version                         Description of evaluation version
```

## Special Thanks to
We extend our gratitude to the authors of the following libraries for generously sharing their source code and dataset:

[RAM](https://github.com/xinyu1205/recognize-anything),
[Llama3](https://github.com/meta-llama/llama3),
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO),
[SAM](https://github.com/facebookresearch/segment-anything),
[SAA+](https://github.com/caoyunkang/Segment-Any-Anomaly?tab=readme-ov-file)
Your contributions are greatly appreciated.

## Citation

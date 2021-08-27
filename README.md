EyeDiseaseSegmentation
==============================

**This project is my contribution to the final Thesis**

## Features
* A framework for exploring multiple retinal datasets with multiple SOTA segmnentation models
* All hyperparameters and training configuration are defined in config.py and automatically save when running
* Experiment tracking and data visualization with both Tensorboard and Weight and Bias
* Available pretrained model for inference 
* A **proposed architecture** which achieve SOTA performance on SE, HE lesions on IDRiD dataset, and reasonable performance on other datasets 

**Note: This instruction is for Linux users**
## Install
* Clone the repo
```
git clone https://github.com/duylebkHCM/EyeDiseaseSegmentation.git
```
* Install pipenv 
```
pip install pipenv
```
* Install the dependencies and activate env
```
pipenv install && pipenv shell
```

## Organization
* notebooks/: contains data exploration on some retinal dataset
* pipeline.py: main file to execute the whole experiment pipeline from training to inference, return metric and analyze the result
* pipeline_vessel.py: similar to pipeline.py but use for vessel segmentation task
* ensemble.py: use to get prediction base on the result of multiple model
* src/: this folder contains all the file use for experiment including archs folder contains all model architectures, config.py use to define configuration, optim.py defines many optimizer, losses.py defines loss functions, scheduler.py define learning scheduler, aucpr.py calculate main metric, train.py is used for model training, stat_results performs some statistic analysis and tta.py use for inference on test set using **test-time-augmentation**

## My proposed architecture
This propose use Unet++ as a baseline model, and then add some improvements like Multi-Head Self Attention blocks in the last stages of the encoder, use Multi-Head Cross Attention on the skip pathway to filter out noisy and irrelevant information. This proposed show a better result than the baseline and achieve reasonable performance  with reponse to other SOTA method.

<p align="center">
<img src ="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/pp.png" alt="pp" width=600 />
</p>

## Do experiments
### Datasets
The experiments are performed on 3 public dataset, one for retina lesion which is IDRiD  and other two for retina vessel which are DRIVE and CHASEDB1.
Some of the example in these datasets:</br>
<p align="center">
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/IDRID.png" alt="IDRID" width=800 />
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/drive.png" alt="drive" width=600 />
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/chase.png" alt="chase" width=600 />
</p>

### Training
* First you need to download the dataset, for IDRiD download at [IDRiD Download](https://idrid.grand-challenge.org/Data_Download/), for DRIVE download at [DRIVE download](https://drive.grand-challenge.org/Download/), for CHASEDB1 download at [CHASEDB1 download](https://blogs.kingston.ac.uk/retinal/chasedb1/), then for DRIVE and CHASEDB1 split into train, val and test set. If you want to perform augmentattion execute augment_data.py in src/data/ folder. 
* Next, modify config.py file. Some configuration you should care about are train_img_path, train_mask_path; augmentation is level of augmentation; model_name and model_param can be referenced at __init__.py at src/archs/; set deep_supervision to True if you want to perform training with it.
* Finally run python pipeline.py or pipeline_vessel.py to start the experiment, you can also track the experiment with Tensorboard and Weight-and-Bias given the links on terminal.
* Outputs of the experiments is save in outputs folder including predicted mask, AUC figure, metrics analysis text files.
* Checkpoints will include best and last checkpoint, and the config.json file, all are saved in models folder

## Results
Some of the results are show below:

### IDRiD
<p align="center">
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/resultunetvspp.png" alt="resultunetvspp" width=800 />
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/ppvsother.png" alt="ppvsother" width=600 />
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/EX.png" alt="EX" width=600 />
</p>

### DRIVE
<p align="center">
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/driveunetvspp.png" alt ="driveunetvspp" width=600 />
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/driveppvsother.png" alt="driveppvsother" width=600 />
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/drivere.png" alt="drivere" width=600 />    
</p>

### CHASEDB1
<p align="center">
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/chaseunetvspp.png" alt ="chaseunetvspp" width=600 />
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/chaseppvsother.png" alt="chaseppvsother" width=600 />
<img src="https://github.com/duylebkHCM/EyeDiseaseSegmentation/blob/main/assets/chasere.png" alt="chasere" width=600 />    
</p>

## Citation
If you find this project is helpful for your work please cite here 
```
@thesis{le_bui_2021, title={A study on a diagnosis system for Eye diseases}, author={Le, Duy Anh and Bui, Anh Ba}, year={2021}}
```

## References
Le, D. A., &amp; Ba, A. B. (2021). A study on a diagnosis system for Eye diseases (thesis). 

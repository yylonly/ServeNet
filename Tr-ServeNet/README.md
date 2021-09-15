Transfer Learning for Web Services Classification
=============

Yilong Yang, Zhaotian Li, Jing Zhang and Yang Chen. "Transfer Learning for Web Services Classification." presented at IEEE 13th International Conferences on Web Services (ICWS'21), Sept. 2021. (Short Paper).
------- 

#### 
Dataset: 
App Dataset: Appstore.csv
Service Dataset: ServiceWithName.csv

* 1_Pre_Processing: Preprocess App and Web service data, calculate and obtain mathematical features.

* 2_Data_Prepare_for_DataGenerator: Further process the data, divide the training and test set, tokenize and padding, etc.

* 3_Pre-training_Model: Pre-train the source model on App dataset.

* 4_Transfer_Learning_and_Fine-tuning: Transfer the pre-trained model and fine-tuning on the target domain.

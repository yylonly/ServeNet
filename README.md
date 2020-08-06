
# ServeNet: A Deep Neural Network for Web Service Classification


Automated service classification plays a crucial role in service management such as service discovery, selection, and composition. In recent years, machine learning techniques have been used for service classification. However, they can only predict around 10 to 20 service categories due to the quality of feature engineering and the imbalance problem of service dataset. In this project, we present a deep neural network ServeNet with a novel dataset splitting algorithm to deal with these issues. ServeNet can automatically abstract low-level representation to high-level features, and then predict service classification based on the service datasets produced by the proposed splitting algorithm. To demonstrate the effectiveness of our approach, we conducted a comprehensive experimental study on 10,000 real-world services in 50 categories. The result shows that ServeNet can achieve higher accuracy than other machine learning methods.

### Please cite our paper as follows:

* Yilong Yang, Wei Ke, Weiru Wang, Yongxin Zhao. "Deep Learning for Web Services Classification". presented at the 11th International Conferences on Web Services (ICWS’19), Milan, Italy, July 2019.

* Yilong Yang, Nafees Qamar, Peng Liu, Katarina Grolinger, Weiru Wang, Zhi Li, Zhifang Liao. "ServeNet: A Deep Neural Network for Web Service Classification" . to be presented at the 12th International Conferences on Web Services (ICWS’20), Beijing, China, Oct. 2020.


### DataSet in Release:

* RAW data
* CSV
* hdf5 or h5
* Tensor in pickle


### Reuseable Models in Release:

* BILSTM.hdf5
* CLSTM.hdf5
* CNN.hdf5
* LSTM.hdf5
* RCNN.hdf5
* ServeNet.hdf5
* ServeNet-ICWS20.hdf5


### Development 

#### (Online) Google Colab

Directly open .ipynb in Google Colab and prepare data 

#### (Local) Start jupyter lab with docker

* git clone https://github.com/yylonly/ServeNet.git

#### CPU
* docker build . -t servenet:cpu -f Dockerfile-CPU
* docker run -itd --rm --name servenet-cpu -p 8888:8888 -v /yourpath:/data servenet:cpu
#### Find URL in log to open jupyter lab
* docker logs servenet-cpu 


#### GPU
* docker build . -t servenet:gpu -f Dockerfile-GPU
* docker run -itd --rm --runtime=nvidia --name servenet-gpu -p 8888:8888 -v /yourpath:/data servenet:gpu

#### Find URL in log to open jupyter lab
* docker logs servenet-gpu 


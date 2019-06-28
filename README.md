[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/yylonly/ServeNet/master)

# ServeNet: A Deep Neural Network for Web Service Classification


Automated service classification plays a crucial role in service management such as service discovery, selection, and composition. In recent years, machine learning techniques have been used for service classification. However, they can only predict around 10 to 20 service categories due to the quality of feature engineering and the imbalance problem of service dataset. In this project, we present a deep neural network ServeNet with a novel dataset splitting algorithm to deal with these issues. ServeNet can automatically abstract low-level representation to high-level features, and then predict service classification based on the service datasets produced by the proposed splitting algorithm. To demonstrate the effectiveness of our approach, we conducted a comprehensive experimental study on 10,000 real-world services in 50 categories. The result shows that ServeNet can achieve higher accuracy than other machine learning methods.

### Please cite our paper as follows:

Yilong Yang, Peng Liu, Lianchao Ding, Bingqing Shen, Weiru Wang. [ServeNet: A Deep Neural Network for Web Service Classification](https://www.researchgate.net/publication/325778290_ServeNet_A_Deep_Neural_Network_for_Web_Service_Classification). arXiv:1806.05437v1


### Start jupyter lab with docker

* git clone https://github.com/yylonly/ServeNet.git
* docker build . -t servenet
* docker run -itd --name servenet -p 8888:8888 --rm -v /path/ServeNet:/data ServeNet 
* docker logs ServeNet //find url to open jupyter lab

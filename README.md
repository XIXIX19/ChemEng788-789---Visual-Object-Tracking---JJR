# ChemEng788-789---Visual-Object-Tracking---JJR
For ChemEng788 Group Project

1) Download all the datasets from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html
2) Three folders 1)'Dataset'2)'Dataset/OTB100'3) 'Dataset/OTB50' are created
3) All the datasets are unzipped manually to 'OTB50'(the first 49 datasets) and 'OTB100'(all datasets)
4) The folder 'Dataset' is used to store these files
5) The 'Preprocessing01, 02, 03.py' are appiled to delete some unnecessary or wrong files
6) The 'Datapreparing.py' is applied to make each frame in each video to the input size for feature extraction
7) The 'Datageneration.py' is applied to store the inputs into the pkl files
8) The 'SiameseFC_train.py' is applied to train the model to extract features of the object
9) The 'log_i_j%' folders store the training result
11) The 'Tracking.py' is applied to track the object based on the training result
12) The 'Tracking_demo.py' is applied to visualize the tracking result

Reference:
Wang, L., Ouyang, W., Wang, X., & Lu, H. (2015). Visual tracking with fully convolutional networks. In Proceedings of the IEEE international conference on computer vision (pp. 3119-3127).
Code Reference
https://github.com/mozhuangb/SiameseFC-pytorch
https://github.com/huanglianghua/siamfc-pytorch
https://github.com/iandreariley/Keras-Siamese-FC
https://github.com/torrvision/siamfc-tf

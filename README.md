# CNN-Machine-Learning-Model-2024
This application involves the development of a CNN-based Machine Learning Model for image classification of barrier coatings microstructures in elastic stress contours, with the goal of accelerating thermo-mechanical fracture tests and reducing the runtime of finite element simulations. Built using the Kaggle Jupyter IDE and the NVIDIA T4 x2 Tensor Core GPU.

Overview:
![Image 7-3-24 at 9 19 PM](https://github.com/sagars2004/CNN-Machine-Learning-Model-2024/assets/145163371/b0d1ccef-ad4a-45c2-9b49-1161ffa33b9b)


Convolutional NN Features:
- Initial inputs of 150x150 images with 4 channels
- Batch size = 32, Weight decay = 1e-4, CrossEntropy loss, Adam optimizer
- Additional hyperparameter tuning:
    - 5-Fold cross validation
    - Batch normalization
    - PyTorch tensors for xData and yData
    - Learning rate scheduler

Accuracy Performance Metrics- Training Accuracy: 99.59%, Test Accuracy: 93.10%
![Image 6-23-24 at 2 02 PM (1)](https://github.com/sagars2004/CNN-Machine-Learning-Model-2024/assets/145163371/ae8da9c1-c23e-4e74-874f-fb1dc1579856)


Loss Performance Metrics- Training Loss: 0.022, Test Loss: 0.235
![Image 6-23-24 at 2 02 PM](https://github.com/sagars2004/CNN-Machine-Learning-Model-2024/assets/145163371/74c87686-8855-4af9-8a96-1cfa9490743a)

Other Classification Metrics- Average Precision: 0.887, Average Recall: 0.890, Average F1 Score: 0.898


ROC Curve and AUC-
![Image 6-22-24 at 12 54 PM](https://github.com/sagars2004/CNN-Machine-Learning-Model-2024/assets/145163371/cb2e1b40-b504-44fc-8172-879eb7895fb8)

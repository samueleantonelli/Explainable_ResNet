# ResNet
Explainability approaches applied to a ResNet50 and ResNet101 trained on CUB-200-2011

1. Copy the git directory:

   ```bash
   git clone https://github.com/samueleantonelli/Explainable_ResNet.git

2. Install requirements by running:
   ```bash
   cd CNN_classifier
   pip install -r requirements_cnn.txt
   
3. Download the CUB-200-2011 dataset
   ```bash
   wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
   mv CUB_200_2011.tgz?download=1 CUB_200_2011.tgz
   tar -xvzf CUB_200_2011.tgz #to open the dataset
   mv CUB_200_2011.tgz Explainable_ResNet/CNN_classifier/CUB_200_2011

4. Modify the paths as needed
   navigate to: **utils/Config.py** and modify the data_path and model_save_path as needed
   Create a new directory to save the models if necessary. 
   
6. Run the code:

    ```bash
    python train.py --net_choice ResNet --model_choice 50 #for training ResNet 50
    python train.py --net_choice ResNet --model_choice 101 #for training ResNet 101
On first attempt the code will extract the images before starting to train the ResNet, while on later trials the images will be ready to be used.


# Explainability
To start working on the explainability, focus on the **TorchVision_Interpret.ipynb**. This Jupyter notebook allows to pick one image out of the ones used in the ResNet training and to apply four different explaination approaches.
Note: before starting to look into interpretability, run the principal ResNet in order to create the model to explain (either ResNet50 or ResNet101).

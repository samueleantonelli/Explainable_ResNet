# Explainability for ResNet trained on CUB-200-2011
Explainability approaches applied to a ResNet50 and ResNet101 trained on CUB-200-2011.

## Steps to implement the code

1. Create a new environment with python >= 3.10

2. Copy the git directory: 

   ```bash
   git clone https://github.com/samueleantonelli/Explainable_ResNet.git

3. Install requirements by running:
   ```bash
   pip install -r requirements.txt
   
3. Download the CUB-200-2011 dataset
   ```bash
   wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
   mv CUB_200_2011.tgz?download=1 CUB_200_2011.tgz
   tar -xvzf CUB_200_2011.tgz
   mv CUB_200_2011.tgz CUB_200_2011/
   mv attributes.txt CUB_200_2011/

4. Modify the paths as needed
   navigate to: **utils/Config.py** and modify the data_path and model_save_path with the personal paths.
   Create a new directory to save the models if necessary, by default they are saved in model_save1. 
   
6. Run the code:
    ```bash
    python train.py --net_choice ResNet --model_choice 50 #for training ResNet 50
    python train.py --net_choice ResNet --model_choice 101 #for training ResNet 101
other specific arguments, such as epochs number, can be found in the file train.py
On first attempt the code will extract the images before starting to train the ResNet, while on later trials the images will be ready to be used without this step.

#### N.B.: the model for ResNet50 is already available in the folder model_save1/ResNet/ResNet50.pkl. The model for ResNet101 required too much space and couldn't be uploaded. It is necessary to manually train it to use it in the explanation part. 

## Explainability
To start working with the explainability approaches, move on the **Explainability.ipynb**, outside CNN_classifier. This Jupyter notebook allows to pick one image out of the ones used in the ResNet training and testing and to apply five different explaination approaches.

Please, before running the notebook modify the paths to the dataset in the second cell with your own.   


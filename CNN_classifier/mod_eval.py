import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.models_for_cub import ResNet
from utils.Config import Config
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from cub import cub200

#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define dataset transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

#load datasets and dataloaders
print("Loading datasets...")
train_dataset = cub200(root=Config.data_path, train=True, transform=train_transform)
test_dataset = cub200(root=Config.data_path, train=False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

#load the trained model
print("Loading trained model...")
model = ResNet(pre_trained=False, n_class=200, model_choice=50).to(device)

#load the saved state_dict
state_dict = torch.load("/mnt/cimec-storage6/users/samuele.antonelli/modifica_resnet_backup/modifica_resnet/CNN_classifier/model_save1/ResNet/ResNet/ResNet50.pkl", map_location=device)

#check if the keys contain 'module.', meaning it was trained using DataParallel
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # Remove 'module.' prefix if present

#load the modified state_dict into the model
model.load_state_dict(new_state_dict, strict=False)  #set strict=False to ignore missing keys if necessary

model.eval()

#define evaluation function
def evaluate_model(loader, dataset_name="Dataset"):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = total_correct / total_samples
    print(f"{dataset_name} Accuracy: {accuracy:.2%}")

    #classification Report
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(all_labels, all_preds))

    #confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(200), yticklabels=range(200))
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return accuracy

#evaluate on training and test datasets
print("Evaluating model on training data...")
train_accuracy = evaluate_model(train_loader, "Training Set")

print("Evaluating model on test data...")
test_accuracy = evaluate_model(test_loader, "Test Set")


print(f"Train accuracy = {train_accuracy*100:.2f}%, Test accuracy = {test_accuracy*100:.2f}%")

if train_accuracy > 90 and test_accuracy < 50:
    print("\nOverfitting condition met.")
    print("\nThe model is likely overfitting. Consider adding regularization or improving data augmentation.")
elif test_accuracy > 90 and train_accuracy < 60:
    print("\nUnderfitting condition met.")
    print("\nThe model is underfitting. Consider increasing training epochs or optimizing hyperparameters.")
else:
    print("\n Balanced condition met.")
    print("\nModel performance seems balanced. Check misclassifications for further insights.")

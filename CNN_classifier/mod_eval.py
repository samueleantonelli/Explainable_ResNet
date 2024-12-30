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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define dataset transformations
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

# Load datasets and dataloaders
print("Loading datasets...")
train_dataset = cub200(root=Config.data_path, train=True, transform=train_transform)
test_dataset = cub200(root=Config.data_path, train=False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the trained model
print("Loading trained model...")
model = ResNet(pre_trained=False, n_class=200, model_choice=50).to(device)
model.load_state_dict(torch.load("/mnt/cimec-storage6/users/samuele.antonelli/modifica_resnet/CNN_classifier/model_save/ResNet50.pkl"))  
model.eval()

# Define evaluation function
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

    # Classification Report
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(all_labels, all_preds))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(200), yticklabels=range(200))
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return accuracy

# Evaluate on training and test datasets
print("Evaluating model on training data...")
train_accuracy = evaluate_model(train_loader, "Training Set")

print("Evaluating model on test data...")
test_accuracy = evaluate_model(test_loader, "Test Set")

# Analyze model for potential issues
if train_accuracy > 90 and test_accuracy < 50:
    print("\nThe model is likely overfitting. Consider adding regularization or improving data augmentation.")
elif train_accuracy < 60:
    print("\nThe model is underfitting. Consider increasing training epochs or optimizing hyperparameters.")
else:
    print("\nModel performance seems balanced. Check misclassifications for further insights.")


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import time
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Output directory
output_dir = "model_outputs"
os.makedirs(output_dir, exist_ok=True)

# Dataset
dataset_dir = r"C:\Users\rahul\Desktop\final_project\AI data"

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),  # Convert image to tensor before applying RandomErasing
    transforms.RandomErasing(p=0.2),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Load dataset
full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
class_names = full_dataset.classes
train_size = int(0.6 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

# Model setup
model_dict = {
    "ResNet18": models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "MobileNetV2": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
    "EfficientNetB0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
}

# Modify output layers
for name, model in model_dict.items():
    if name == "ResNet18":
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    else:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model_dict[name] = model.to(device)

results = []

def train_and_validate(model, name, epochs=100):  # Updated epochs to 100
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    train_accs, val_accs, train_losses, val_f1s = [], [], [], []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        all_preds, all_labels, total_loss = [], [], 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

        train_accs.append(train_acc)
        train_losses.append(total_loss / len(train_loader))
        val_acc, val_f1 = validate_epoch(model, val_loader)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f"{name} - Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    training_time = time.time() - start_time

    final_metrics = evaluate_model(model, val_loader, name)
    results.append([
        name,
        final_metrics['accuracy'],
        final_metrics['precision'],
        final_metrics['recall'],
        final_metrics['f1'],
        training_time
    ])

    plot_metrics(name, train_accs, val_accs, val_f1s, train_losses)

def validate_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    return acc, f1

def evaluate_model(model, loader, model_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

    # Save classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=1)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f"{model_name}_classification_report.csv"))

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, model_name)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

def plot_metrics(model_name, train_accs, val_accs, val_f1s, train_losses):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(val_f1s, label="Val F1-Score", color='green')
    plt.title(f"{model_name} - F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_losses, label="Train Loss", color='red')
    plt.title(f"{model_name} - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_metrics_plot.png"))
    plt.close()

def predict_image_with_all_models(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    print(f"\nPredictions for image: {image_path}\n")
    for name, model in model_dict.items():
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probs = nn.functional.softmax(output, dim=1)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[0, pred_idx].item()
        print(f"{name}: {class_names[pred_idx]} ({confidence*100:.2f}%)")

# Train and evaluate all models with 100 epochs
for model_name, model in model_dict.items():
    train_and_validate(model, model_name, epochs=100)

# Save final comparison table
summary_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Training Time (s)"])
summary_df.to_csv(os.path.join(output_dir, "final_model_summary.csv"), index=False)
print("\nFinal Model Comparison:\n")
print(summary_df)

# Predict using all models
test_image_path = r"C:\Users\rahul\Desktop\final_project\AI data\Geranium\image_02646.jpg"
predict_image_with_all_models(test_image_path)
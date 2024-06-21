import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import random

def few_shot_dataset(dataset, num_classes=100, num_samples_per_class=5):
    """
    Create a few-shot dataset with a limited number of samples per class.
    """
    indices = []
    targets = np.array(dataset.targets)
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        class_sample_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
        indices.extend(class_sample_indices)
    return Subset(dataset, indices)

def set_seed(seed=42):
    """
    Set seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def main(num_epochs=30, batch_size=32, learning_rate=0.01, num_workers=2, num_samples_per_class=5):
    # Set seed for reproducibility
    set_seed()

    # Step 1: Data Preparation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # Create few-shot train dataset
    few_shot_train_dataset = few_shot_dataset(train_dataset, num_samples_per_class=num_samples_per_class)

    # Split train dataset into train and validation sets
    val_size = int(0.2 * len(few_shot_train_dataset))
    train_size = len(few_shot_train_dataset) - val_size
    train_subset, val_subset = random_split(few_shot_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Step 2: Model Selection
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust the final layer for CIFAR-100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Step 3: Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop with validation
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validate after each epoch
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Step 4: Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')

if __name__ == '__main__':
    main()

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 1024 
K_VALUES = list(range(1, 15)) 

transform = transforms.Compose([
    lambda img: transforms.functional.rotate(img, -90),
    lambda img: transforms.functional.hflip(img),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def extract_all_data(loader):
    x_all, y_all = [], []
    for images, labels in loader:
        x_all.append(images.view(images.shape[0], -1).numpy())
        y_all.append(labels.numpy())
    return np.vstack(x_all), np.concatenate(y_all)

X_train, y_train = extract_all_data(train_loader)
X_test, y_test = extract_all_data(test_loader)

test_accuracies = []

for k in K_VALUES:
    
    knn_model = KNeighborsClassifier(
        n_neighbors=k, 
        weights='distance', 
        n_jobs=-1           
    )
    
    knn_model.fit(X_train, y_train)
    
    test_preds = knn_model.predict(X_test)
    
    test_acc = accuracy_score(y_test, test_preds) * 100
    test_accuracies.append(test_acc)
    
    print(f"K = {k} | Test Acc: {test_acc:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(K_VALUES, test_accuracies, label='Test Accuracy', marker='s', color='orange')
plt.title('KNN Accuracy vs. Number of Neighbors (K)')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy (%)')
plt.xticks(K_VALUES)
plt.legend()
plt.grid(True)
plt.savefig('knn_accuracy_vs_k.png', dpi=300, bbox_inches='tight')
plt.show()
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib
import numpy as np

BATCH_SIZE = 1024
TREES_PER_STEP = 15
TOTAL_STEPS = 15
MODEL_SAVE_PATH = "emnist_random_forest.pkl"

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

rf_model = RandomForestClassifier(
    n_estimators=10,      
    warm_start=True,     
    n_jobs=-1,           
    random_state=42,
    max_depth=13         
)

train_accuracies = []
test_accuracies = []
tree_counts = []

for step in range(1, TOTAL_STEPS + 1):
    current_trees = step * TREES_PER_STEP
    rf_model.set_params(n_estimators=current_trees)
    
    rf_model.fit(X_train, y_train)
    
    train_preds = rf_model.predict(X_train)
    test_preds = rf_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds) * 100
    test_acc = accuracy_score(y_test, test_preds) * 100
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    tree_counts.append(current_trees)
    
    print(f"Step [{step}/{TOTAL_STEPS}] | Trees: {current_trees} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

joblib.dump(rf_model, MODEL_SAVE_PATH)

plt.figure(figsize=(10, 6))
plt.plot(tree_counts, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(tree_counts, test_accuracies, label='Test Accuracy', marker='s')
plt.title('Random Forest Accuracy vs. Number of Trees')
plt.xlabel('Number of Trees in Forest')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('rf_accuracy_vs_trees.png', dpi=300, bbox_inches='tight')
plt.show()
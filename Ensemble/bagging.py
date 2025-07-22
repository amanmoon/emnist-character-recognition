import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib
import numpy as np

BATCH_SIZE = 1024 
ESTIMATORS_PER_STEP = 5
TOTAL_STEPS = 10
MODEL_SAVE_PATH = "emnist_bagging.pkl"

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

base_tree = DecisionTreeClassifier(max_depth=20, random_state=42)

bagging_model = BaggingClassifier(
    estimator=base_tree,
    n_estimators=0,      
    warm_start=True,     
    n_jobs=-1,           
    random_state=42
)

train_accuracies = []
test_accuracies = []
estimator_counts = []

for step in range(1, TOTAL_STEPS + 1):
    current_estimators = step * ESTIMATORS_PER_STEP
    bagging_model.set_params(n_estimators=current_estimators)
    
    bagging_model.fit(X_train, y_train)
    
    train_preds = bagging_model.predict(X_train)
    test_preds = bagging_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds) * 100
    test_acc = accuracy_score(y_test, test_preds) * 100
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    estimator_counts.append(current_estimators)
    
    print(f"Step [{step}/{TOTAL_STEPS}] | Estimators: {current_estimators} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

joblib.dump(bagging_model, MODEL_SAVE_PATH)

plt.figure(figsize=(10, 6))
plt.plot(estimator_counts, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(estimator_counts, test_accuracies, label='Test Accuracy', marker='s')
plt.title('Bagging Accuracy vs. Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('bagging_accuracy_vs_estimators.png', dpi=300, bbox_inches='tight')
plt.show()
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import matplotlib.pyplot as plt
import joblib
import numpy as np

BATCH_SIZE = 1024 
EPOCHS = 15
CLASSES = np.arange(47) 
MODEL_SAVE_PATH = "emnist_voting_ensemble.pkl"
transform = transforms.Compose([
    lambda img: transforms.functional.rotate(img, -90),
    lambda img: transforms.functional.hflip(img),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def extract_all_data(loader):
    x_all, y_all = [], []
    for images, labels in loader:
        x_all.append(images.view(images.shape[0], -1).numpy())
        y_all.append(labels.numpy())
    return np.vstack(x_all), np.concatenate(y_all)

X_test, y_test = extract_all_data(test_loader)

clf1 = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, random_state=42, n_jobs=-1)

clf2 = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, random_state=42, n_jobs=-1)

clf3 = SGDClassifier(loss='perceptron', penalty='l2', alpha=1e-4, random_state=42, n_jobs=-1)

train_accuracies = []
test_accuracies = []

for epoch in range(EPOCHS):
    
    for images, labels in train_loader:
        X_batch = images.view(images.shape[0], -1).numpy()
        y_batch = labels.numpy()

        clf1.partial_fit(X_batch, y_batch, classes=CLASSES)
        clf2.partial_fit(X_batch, y_batch, classes=CLASSES)
        clf3.partial_fit(X_batch, y_batch, classes=CLASSES)
        
    print(f"Calculating votes for Epoch {epoch+1}...")
    
    preds1_test = clf1.predict(X_test)
    preds2_test = clf2.predict(X_test)
    preds3_test = clf3.predict(X_test)
    
    preds1_train = clf1.predict(X_batch)
    preds2_train = clf2.predict(X_batch)
    preds3_train = clf3.predict(X_batch)
    
    stacked_test_preds = np.array([preds1_test, preds2_test, preds3_test])
    stacked_train_preds = np.array([preds1_train, preds2_train, preds3_train])
    
    try:
        final_test_preds = mode(stacked_test_preds, axis=0, keepdims=True).mode[0]
        final_train_preds = mode(stacked_train_preds, axis=0, keepdims=True).mode[0]
    except TypeError:
        final_test_preds = mode(stacked_test_preds, axis=0)[0][0]
        final_train_preds = mode(stacked_train_preds, axis=0)[0][0]
    
    train_acc = accuracy_score(y_batch, final_train_preds) * 100
    test_acc = accuracy_score(y_test, final_test_preds) * 100
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

ensemble_dict = {
    'logistic_regression': clf1,
    'svm': clf2,
    'perceptron': clf3
}
joblib.dump(ensemble_dict, MODEL_SAVE_PATH)

plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy (Last Batch)', marker='o')
plt.plot(range(1, EPOCHS + 1), test_accuracies, label='Test Accuracy', marker='s')
plt.title('Voting Ensemble Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('voting_ensemble_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()
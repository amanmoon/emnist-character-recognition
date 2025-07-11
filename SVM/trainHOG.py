import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import matplotlib.pyplot as plt
import joblib
import numpy as np

BATCH_SIZE = 1024  
EPOCHS = 15
CLASSES = np.arange(47) 
MODEL_SAVE_PATH = "emnist_hog_svm.pkl"

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

def extract_hog_features(images_tensor):
    images_np = images_tensor.numpy()[:, 0, :, :] 
    
    hog_features_list = []
    for img in images_np:
        features = hog(
            img, 
            orientations=9, 
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2), 
            visualize=False,
            feature_vector=True
        )
        hog_features_list.append(features)
        
    return np.array(hog_features_list)

svm_model = SGDClassifier(
    loss='hinge', 
    penalty='elasticnet', 
    l1_ratio=0.15,
    alpha=1e-5,              
    learning_rate='adaptive', 
    eta0=0.01,               
    random_state=42,
    n_jobs=-1
)

train_accuracies = []
test_accuracies = []

for epoch in range(EPOCHS):
    
    epoch_train_preds = []
    epoch_train_labels = []
    
    for i, (images, labels) in enumerate(train_loader):
        X_batch_hog = extract_hog_features(images)
        y_batch = labels.numpy()
        
        svm_model.partial_fit(X_batch_hog, y_batch, classes=CLASSES)
        
        preds = svm_model.predict(X_batch_hog)
        epoch_train_preds.extend(preds)
        epoch_train_labels.extend(y_batch)
        
    train_acc = accuracy_score(epoch_train_labels, epoch_train_preds) * 100
    train_accuracies.append(train_acc)
    
    epoch_test_preds = []
    epoch_test_labels = []
    
    for images, labels in test_loader:
        X_batch_hog = extract_hog_features(images)
        y_batch = labels.numpy()
        
        preds = svm_model.predict(X_batch_hog)
        epoch_test_preds.extend(preds)
        epoch_test_labels.extend(y_batch)
        
    test_acc = accuracy_score(epoch_test_labels, epoch_test_preds) * 100
    test_accuracies.append(test_acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

joblib.dump(svm_model, MODEL_SAVE_PATH)

plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, EPOCHS + 1), test_accuracies, label='Test Accuracy', marker='s')
plt.title('HOG + SVM Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('hog_svm_accuracy_vs_epoch.png', dpi=300, bbox_inches='tight')
plt.show()
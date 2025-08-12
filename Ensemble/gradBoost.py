import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 1024 
BOOSTING_ROUNDS = 60
MODEL_SAVE_PATH = "emnist_xgboost.json" 

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

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

xgb_model = XGBClassifier(
    n_estimators=BOOSTING_ROUNDS,
    learning_rate=0.01,
    max_depth=8,             
    objective='multi:softmax', 
    num_class=47,
    tree_method='hist',      
    device=device,           
    random_state=42,
    eval_metric="merror"     
)

eval_set = [(X_train, y_train), (X_test, y_test)]

xgb_model.fit(
    X_train, 
    y_train, 
    eval_set=eval_set, 
    verbose=True 
)

xgb_model.save_model(MODEL_SAVE_PATH)

results = xgb_model.evals_result()

train_accuracies = [(1 - err) * 100 for err in results['validation_0']['merror']]
test_accuracies = [(1 - err) * 100 for err in results['validation_1']['merror']]

plt.figure(figsize=(10, 6))
plt.plot(range(1, BOOSTING_ROUNDS + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, BOOSTING_ROUNDS + 1), test_accuracies, label='Test Accuracy', marker='s')
plt.title('XGBoost Accuracy vs. Boosting Rounds')
plt.xlabel('Boosting Round (Trees)')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('xgboost_accuracy_vs_rounds.png', dpi=300, bbox_inches='tight')
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

BATCH_SIZE = 512
EPOCHS = 15
LEARNING_RATE = 0.0002
L1_LAMBDA = 1e-6 
L2_LAMBDA = 1e-5 
MODEL_SAVE_PATH = "emnist_logistic_regression.pth"

NUM_CLASSES = 47
INPUT_DIM = 28 * 28

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

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

class MultiClassLR(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiClassLR, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.linear(x)
        return out

model = MultiClassLR(INPUT_DIM, NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

train_accuracies = []
test_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        base_loss = criterion(outputs, labels)
        
        l1_reg = torch.tensor(0., device=device)
        l2_reg = torch.tensor(0., device=device)
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg += torch.norm(param, p=1)
                l2_reg += torch.norm(param, p=2) ** 2
                
        loss = base_loss + (L1_LAMBDA * l1_reg) + (L2_LAMBDA * l2_reg)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    train_acc = 100 * correct_train / total_train
    train_accuracies.append(train_acc)
    
    model.eval()
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    test_acc = 100 * correct_test / total_test
    test_accuracies.append(test_acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Total Loss: {loss.item():.4f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)

plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, EPOCHS + 1), test_accuracies, label='Test Accuracy', marker='s')
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_epoch.png', dpi=300, bbox_inches='tight')
plt.show()
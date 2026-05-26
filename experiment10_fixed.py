import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("第10次实验：CNN 训练过程分析与可视化")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========== 数据加载 ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("\n加载 MNIST 数据集...")
full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_set, val_set = random_split(full_train, [50000, 10000])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

print(f"训练集: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")

# ========== CNN 模型定义 ==========
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        self.conv1_out = self.relu1(self.conv1(x))
        x = self.pool1(self.conv1_out)
        
        self.conv2_out = self.relu2(self.conv2(x))
        x = self.pool2(self.conv2_out)
        
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# ========== 混淆矩阵函数 ==========
def compute_confusion_matrix(preds, labels, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    return cm

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    # 归一化
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.nan_to_num(cm_percent)
    
    # 绘制
    plt.imshow(cm_percent, cmap='Blues')
    plt.colorbar()
    plt.title('测试集混淆矩阵 (归一化)')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    
    # 添加数字标注
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i][j]}', ha='center', va='center', 
                    color='white' if cm_percent[i][j] > 0.5 else 'black')
    
    plt.xticks(range(10), range(10))
    plt.yticks(range(10), range(10))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ========== 任务2：优化器对比 ==========
def train_with_optimizer(optimizer_name, lr=0.001, momentum=0.9, epochs=5):
    print(f"\n优化器: {optimizer_name}, lr={lr}")
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD+Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = 100 * test_correct / test_total
    print(f"测试准确率: {test_acc:.2f}%")
    
    return {
        'name': optimizer_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_acc': test_acc
    }

print("\n" + "="*60)
print("任务2：优化器对比")
print("="*60)

optimizer_results = []
optimizer_results.append(train_with_optimizer('SGD', lr=0.01))
optimizer_results.append(train_with_optimizer('SGD+Momentum', lr=0.01))
optimizer_results.append(train_with_optimizer('Adam', lr=0.001))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
for r in optimizer_results:
    plt.plot(range(1, 6), r['val_accs'], 'o-', label=r['name'])
plt.title('不同优化器验证 Accuracy 对比')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
for r in optimizer_results:
    plt.plot(range(1, 6), r['val_losses'], 'o-', label=r['name'])
plt.title('不同优化器验证 Loss 对比')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('task2_optimizer_comparison.png')
plt.close()
print("\n已保存: task2_optimizer_comparison.png")

# ========== 任务3：学习率对比 ==========
print("\n" + "="*60)
print("任务3：学习率对比 (Adam优化器)")
print("="*60)

lr_results = []
for lr in [0.1, 0.01, 0.001]:
    lr_results.append(train_with_optimizer('Adam', lr=lr, epochs=5))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
for i, r in enumerate(lr_results):
    lr_val = [0.1, 0.01, 0.001][i]
    plt.plot(range(1, 6), r['val_accs'], 'o-', label=f"lr={lr_val}")
plt.title('不同学习率验证 Accuracy 对比')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
for i, r in enumerate(lr_results):
    lr_val = [0.1, 0.01, 0.001][i]
    plt.plot(range(1, 6), r['val_losses'], 'o-', label=f"lr={lr_val}")
plt.title('不同学习率验证 Loss 对比')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('task3_lr_comparison.png')
plt.close()
print("\n已保存: task3_lr_comparison.png")

# ========== 任务1：重新训练最终模型 ==========
print("\n" + "="*60)
print("任务1：重新训练最终模型")
print("="*60)

final_model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.001)

for epoch in range(5):
    final_model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = final_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)
    
    final_model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = final_model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)
    
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

final_model.eval()
test_correct, test_total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = final_model(images)
        preds = outputs.argmax(1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * test_correct / test_total
print(f"\n最终测试准确率: {test_acc:.2f}%")

# ========== 任务4：卷积核可视化 ==========
print("\n" + "="*60)
print("任务4：卷积核可视化")
print("="*60)

conv1_weights = final_model.conv1.weight.data.cpu().numpy()
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i < len(conv1_weights):
        kernel = conv1_weights[i, 0]
        ax.imshow(kernel, cmap='gray')
        ax.set_title(f'Kernel {i+1}')
        ax.axis('off')
plt.suptitle('第一层卷积核可视化')
plt.tight_layout()
plt.savefig('task4_conv_kernels.png')
plt.close()
print("已保存: task4_conv_kernels.png")

# ========== 任务5：Feature map 可视化 ==========
print("\n" + "="*60)
print("任务5：Feature map 可视化")
print("="*60)

test_img, test_label = test_set[0]
test_img_tensor = test_img.unsqueeze(0).to(device)

final_model.eval()
with torch.no_grad():
    conv1_out = final_model.conv1(test_img_tensor)
    conv1_out = final_model.relu1(conv1_out)
    feature_maps = conv1_out.squeeze().cpu().numpy()

plt.figure(figsize=(14, 8))
plt.subplot(3, 6, 1)
img_display = test_img * 0.5 + 0.5
plt.imshow(img_display.squeeze(), cmap='gray')
plt.title(f'原图 (标签: {test_label})')
plt.axis('off')

for i in range(min(16, len(feature_maps))):
    plt.subplot(3, 6, i+2)
    plt.imshow(feature_maps[i], cmap='gray')
    plt.title(f'Map {i+1}')
    plt.axis('off')
plt.suptitle('第一层卷积输出的 Feature Maps')
plt.tight_layout()
plt.savefig('task5_feature_maps.png')
plt.close()
print("已保存: task5_feature_maps.png")

# ========== 任务6：错误分类样本分析 ==========
print("\n" + "="*60)
print("任务6：错误分类样本分析")
print("="*60)

final_model.eval()
misclassified = []
with torch.no_grad():
    for i, (img, label) in enumerate(test_set):
        img_tensor = img.unsqueeze(0).to(device)
        output = final_model(img_tensor)
        pred = output.argmax(1).item()
        if pred != label:
            misclassified.append((i, img, label, pred))
        if len(misclassified) >= 20:
            break

print(f"找到 {len(misclassified)} 个错误分类样本")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    if i < len(misclassified):
        idx, img, true_label, pred_label = misclassified[i]
        img_display = img * 0.5 + 0.5
        ax.imshow(img_display.squeeze(), cmap='gray')
        ax.set_title(f'真:{true_label} 预:{pred_label}', color='red')
        ax.axis('off')
plt.suptitle('错误分类样本分析')
plt.tight_layout()
plt.savefig('task6_misclassified.png')
plt.close()
print("已保存: task6_misclassified.png")

# ========== 任务7：混淆矩阵 ==========
print("\n" + "="*60)
print("任务7：混淆矩阵")
print("="*60)

cm = compute_confusion_matrix(all_preds, all_labels)
plot_confusion_matrix(cm, 'task7_confusion_matrix.png')
print("已保存: task7_confusion_matrix.png")

# 找出最易混淆的类别对
cm_no_diag = cm.copy()
for i in range(10):
    cm_no_diag[i][i] = 0
most_confused = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
print(f"\n最容易混淆的类别对: {most_confused[0]} 和 {most_confused[1]}, 混淆数量: {cm[most_confused]}")

# ========== 打印总结 ==========
print("\n" + "="*60)
print("实验总结")
print("="*60)
print(f"""
任务2 优化器对比结果:
  - SGD: {optimizer_results[0]['test_acc']:.2f}%
  - SGD+Momentum: {optimizer_results[1]['test_acc']:.2f}%
  - Adam: {optimizer_results[2]['test_acc']:.2f}%

任务3 学习率对比结果:
  - lr=0.1: {lr_results[0]['test_acc']:.2f}%
  - lr=0.01: {lr_results[1]['test_acc']:.2f}%
  - lr=0.001: {lr_results[2]['test_acc']:.2f}%

任务1 最终模型测试准确率: {test_acc:.2f}%

生成的文件:
  - task2_optimizer_comparison.png
  - task3_lr_comparison.png
  - task4_conv_kernels.png
  - task5_feature_maps.png
  - task6_misclassified.png
  - task7_confusion_matrix.png
""")

print("\n第10次实验完成！")

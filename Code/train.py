import torch
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from lithology_classification_model import DSSMLS
from data_loader import LithologyDataset  
import os

# 配置文件或超参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
NUM_CLASSES = 10
EMBEDDING_DIM = 64
GAMMA = 1.0
KERNEL_SIZES = [3, 5]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据加载
def get_data_loaders():
    #  数据集类
    train_dataset = LithologyDataset(train=True)  # 获取训练数据集
    val_dataset = LithologyDataset(train=False)  # 获取验证数据集

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# 模型训练
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for i, (support_set, query_set, unlabeled_set, query_labels) in enumerate(
        train_loader
    ):
        support_set, query_set, unlabeled_set, query_labels = (
            support_set.to(DEVICE),
            query_set.to(DEVICE),
            unlabeled_set.to(DEVICE),
            query_labels.to(DEVICE),
        )

         
        distances, probabilities = model(support_set, query_set, unlabeled_set)

        
        loss = model.calculate_loss(query_set, query_labels, distances)

       
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新权重

        running_loss += loss.item()
        if i % 10 == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

    return running_loss / len(train_loader)


# 验证模型
def validate(model, val_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (support_set, query_set, unlabeled_set, query_labels) in enumerate(
            val_loader
        ):
            support_set, query_set, unlabeled_set, query_labels = (
                support_set.to(DEVICE),
                query_set.to(DEVICE),
                unlabeled_set.to(DEVICE),
                query_labels.to(DEVICE),
            )

            # 前向传播
            distances, probabilities = model(support_set, query_set, unlabeled_set)

            # 计算准确率
            accuracy = model.evaluate(query_set, query_labels, distances)
            correct += accuracy.item() * query_set.size(0)
            total += query_set.size(0)

    avg_accuracy = correct / total
    print(f"Validation Accuracy: {avg_accuracy:.4f}")
    return avg_accuracy


# 保存模型
def save_checkpoint(model, epoch, best_val_acc, save_dir="checkpoints"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(
        save_dir, f"dssmls_epoch_{epoch}_val_acc_{best_val_acc:.4f}.pth"
    )
    model.save_model(save_path)


# 主训练流程
def main():
    # 加载数据
    train_loader, val_loader = get_data_loaders()

    # 初始化模型
    model = DSSMLS(
        in_channels=1,
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        kernel_sizes=KERNEL_SIZES,
        gamma=GAMMA,
    ).to(DEVICE)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()  # 适用于分类任务

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_loss = train(model, train_loader, optimizer, criterion, epoch)
        print(f"Epoch [{epoch}/{EPOCHS}] Training Loss: {train_loss:.4f}")

        # 验证
        val_acc = validate(model, val_loader)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, epoch, best_val_acc)


if __name__ == "__main__":
    main()

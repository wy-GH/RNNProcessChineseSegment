import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from collections import Counter

# 自定义数据集类,用于加载和处理文本数据
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        # 将文本转换为张量
        self.texts = [torch.tensor(text) for text in texts]
        # 将标签转换为张量
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# 定义一个collate函数,用于批处理数据
def collate_fn(batch):
    texts, labels = zip(*batch)
    # 对文本序列进行填充,使其长度一致
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    # 记录每个序列的原始长度
    lengths = torch.tensor([len(text) for text in texts])
    return texts_padded, labels, lengths

# 加载和预处理数据
def load_data(file_path):
    texts = []
    labels = []
    # 定义标签到索引的映射
    label_map = {"体育": 0, "财经": 1, "房产": 2, "家居": 3, "教育": 4,
                 "科技": 5, "时尚": 6, "时政": 7, "游戏": 8, "娱乐": 9}
    all_text = []

    # 读取文件并处理数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                label, text = line.strip().split('\t', 1)
                all_text.append(text)
                labels.append(label_map[label])

    # 构建词汇表
    word_counts = Counter(" ".join(all_text).split())
    vocab = {word: idx+1 for idx, (word, _) in enumerate(word_counts.most_common(10000))}

    # 将文本转换为索引序列
    for text in all_text:
        indexed_text = [vocab.get(word, 0) for word in text.split()]
        texts.append(indexed_text)

    return texts, labels

# 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # RNN层
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # 对输入进行词嵌入
        x = self.embedding(x)
        # 打包填充序列
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # 通过RNN层
        out, _ = self.rnn(packed_x)
        # 解包填充序列
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 训练函数
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for texts, labels, lengths in tqdm(data_loader, desc="Training"):
        texts, labels = texts.to(device), labels.to(device)
        # 注意：lengths 保持在 CPU 上
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# 评估函数
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels, lengths in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            # 注意：lengths 保持在 CPU 上
            outputs = model(texts, lengths)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # 设置设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_texts, train_labels = load_data('./data/cnews.train.txt')
    val_texts, val_labels = load_data('./data/cnews.val.txt')
    test_texts, test_labels = load_data('./data/cnews.test.txt')

    # 创建数据集
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)
    test_dataset = TextDataset(test_texts, test_labels)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # 初始化模型
    model = RNNModel(vocab_size=10001, embed_size=args.embed_size, hidden_size=args.hidden_size, output_size=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练循环
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 在测试集上评估
    test_accuracy = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()
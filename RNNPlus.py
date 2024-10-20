import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
import argparse
from tqdm import tqdm
import jieba
from collections import Counter

# 数据预处理
def load_data(file_path, max_vocab_size=50000, max_seq_length=500, vocab=None):
    """
    加载并预处理文本数据
    
    参数:
    file_path: 数据文件路径
    max_vocab_size: 最大词汇表大小
    max_seq_length: 最大序列长度
    vocab: 预定义词汇表（如果有）

    返回:
    indexed_texts: 索引化的文本
    labels: 对应的标签
    vocab: 词汇表
    """
    texts = []
    labels = []
    # 定义标签映射
    label_map = {"体育": 0, "财经": 1, "房产": 2, "家居": 3, "教育": 4,
                 "科技": 5, "时尚": 6, "时政": 7, "游戏": 8, "娱乐": 9}
    all_words = []

    # 读取数据文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                label, text = line.strip().split('\t', 1)
                # 使用jieba分词，并限制序列长度
                words = jieba.lcut(text)[:max_seq_length]
                all_words.extend(words)
                texts.append(words)
                labels.append(label_map[label])

    # 如果没有预定义词汇表，则创建新的词汇表
    if vocab is None:
        word_counts = Counter(all_words)
        vocab = {"<PAD>": 0, "<UNK>": 1}
        vocab.update({word: idx + 2 for idx, (word, _) in enumerate(word_counts.most_common(max_vocab_size - 2))})

    # 将文本转换为索引序列
    indexed_texts = [[vocab.get(word, vocab["<UNK>"]) for word in text] for text in texts]

    return indexed_texts, labels, vocab

# 定义数据集类
class TextDataset(Dataset):
    """
    文本数据集类，用于PyTorch的DataLoader
    """
    def __init__(self, texts, labels):
        self.texts = [torch.tensor(text) for text in texts]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# 定义数据加载器的整理函数
def collate_fn(batch):
    """
    整理函数，用于处理不同长度的序列
    """
    texts, labels = zip(*batch)
    # 对文本序列进行填充
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    # 记录每个序列的实际长度
    lengths = torch.tensor([len(text) for text in texts])
    return texts_padded, labels, lengths

# 模型定义
class AttentionBiLSTM(nn.Module):
    """
    带注意力机制的双向LSTM模型
    """
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, n_layers=2, dropout=0.5):
        super(AttentionBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # 词嵌入
        embedded = self.dropout(self.embedding(x))
        # 打包序列（处理变长序列）
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # 解包序列
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 注意力机制
        attention_weights = F.softmax(self.attention(output), dim=1)
        context_vector = attention_weights * output
        context_vector = torch.sum(context_vector, dim=1)

        # 最终分类
        return self.fc(context_vector)

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10, patience=3):
    """
    模型训练函数
    """
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# 评估函数
def evaluate(model, data_loader, criterion, device):
    """
    模型评估函数
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels, lengths in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()

    # 设置设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载和预处理数据
    print("Loading and preprocessing data...")
    train_texts, train_labels, vocab = load_data('./data/cnews.train.txt')
    val_texts, val_labels, _ = load_data('./data/cnews.val.txt', vocab=vocab)
    test_texts, test_labels, _ = load_data('./data/cnews.test.txt', vocab=vocab)

    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)
    test_dataset = TextDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # 初始化模型
    print("Initializing model...")
    model = AttentionBiLSTM(
        vocab_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        output_size=10,
        dropout=args.dropout
    ).to(device)

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)

    # 训练模型
    print("Starting training...")
    model = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=args.epochs)

    # 在测试集上评估
    print("Evaluating on test set...")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'final_model.pth')
    print("Model saved as 'final_model.pth'")

if __name__ == "__main__":
    main()
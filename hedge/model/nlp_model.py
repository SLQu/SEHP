import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假设数据
sentences = ["我喜欢这个产品", "这个产品很糟糕", "我不确定这个产品"]  # 示例句子
labels = [1, 0, 2]  # 示例标签，0, 1, 2 代表三种分类

# 分割数据为训练集和测试集
train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2)

# 数据集类
class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.encodings = tokenizer(sentences, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 初始化分词器和模型
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

# 准备数据集
train_dataset = SentenceDataset(train_sentences, train_labels, tokenizer)
test_dataset = SentenceDataset(test_sentences, test_labels, tokenizer)

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# 评估模型
def evaluate(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=16)
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(batch['labels'].tolist())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

print("评估结果:", evaluate(model, test_dataset))

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# Define the PyTorch model
class TransformerBinaryClassifier(nn.Module):
    def __init__(self, model_name):
        super(TransformerBinaryClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.output = nn.Linear(self.transformer.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # Get the [CLS] token representation
        logits = self.output(pooled_output)
        return nn.functional.softmax(logits, dim=1)

# Define a custom dataset class to load the data
class YesNoDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, label = self.data[index]
        inputs = self.tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return input_ids, attention_mask, label

# Define the training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Example usage
train_data = [("Will it rain tomorrow?", 1), ("Is the sun yellow?", 0), ...]  # Replace with your own data
model_name = "distilbert-base-uncased"  # Use any other model you prefer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TransformerBinaryClassifier(model_name)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a dataloader for the training data
train_dataset = YesNoDataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train the model for a specified number of epochs
num_epochs = 10  # Or any other number you choose
for epoch in range(num_epochs):
    train(model, train_loader, optimizer, criterion, device)

# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")

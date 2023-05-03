import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained BERT model and freeze the lower layers
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=28)
for param in model.bert.parameters():
    param.requires_grad = False

# Create a new classification layer on top of the BERT model
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 28)
)

# Load the training data
train_inputs = torch.load('train_inputs.pt')
train_masks = torch.load('train_masks.pt')
train_labels = torch.load('train_labels.pt')
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

# Set the hyperparameters for fine-tuning
epochs = 3
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

# Train the model and print the average loss at each epoch
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch_inputs, batch_masks, batch_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")

# Load the test data and create a data loader
test_inputs = torch.load('test_inputs.pt')
test_masks = torch.load('test_masks.pt')
test_labels = torch.load('test_labels.pt')
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)

# Evaluate the model on the test data and print the F1-score
model.eval()
predictions, true_labels = [], []
for batch in test_dataloader:
    batch_inputs, batch_masks, batch_labels = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        outputs = model(batch_inputs, attention_mask=batch_masks)
    logits = outputs.logits
    batch_predictions = torch.sigmoid(logits).cpu().detach().numpy()
    batch_labels = batch_labels.cpu().detach().numpy()
    predictions.append(batch_predictions)
    true_labels.append(batch_labels)
predictions = np.vstack(predictions)
true_labels = np.vstack(true_labels)
thresholds = np.arange(0.1, 1.0, 0.1)
for threshold in thresholds:
    thresholded_preds = (predictions > threshold).astype(int)
    f1_score = sklearn.metrics.f1_score(true_labels, thresholded_preds, average='micro')
    print(f"F1-score at threshold {threshold:.1f}: {f1_score:.4f}")
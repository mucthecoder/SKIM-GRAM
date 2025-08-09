import re
from collections import OrderedDict, Counter
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk import pos_tag
from nltk.tag import map_tag

# Download NLTK data (only need once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')

# --------- 1. Load and preprocess text ---------
with open("C:/Users/charl/Downloads/harry_potter_extracted/harry_potter/HP1.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read().lower()

# Keep only alphanum + spaces
text = re.sub(r'[^a-z0-9\s]', ' ', text)

tokens_all = text.split()

# Filter rare words before slicing to keep consistency
freqs = Counter(tokens_all)
min_count = 5
tokens_all = [t for t in tokens_all if freqs[t] >= min_count]

# Split train/test tokens for embedding training and POS evaluation
split_ratio = 0.8
split_idx = int(len(tokens_all) * split_ratio)
tokens_train = tokens_all[:split_idx]
tokens_test = tokens_all[split_idx:]

# Limit tokens_train size for speed
tokens_train = tokens_train[:100000]

# --------- 2. Build vocabulary from train tokens ---------
word2idx = OrderedDict()
idx2word = []
for w in tokens_train:
    if w not in word2idx:
        word2idx[w] = len(word2idx)
        idx2word.append(w)

vocab_size = len(idx2word)
print("Vocab size:", vocab_size)

# --------- 3. Create skip-gram pairs ---------
def generate_pairs(tokens, window_size=2):
    pairs = []
    N = len(tokens)
    for i, w in enumerate(tokens):
        if w not in word2idx:
            continue
        center_idx = word2idx[w]
        for j in range(1, window_size + 1):
            if i - j >= 0 and tokens[i - j] in word2idx:
                pairs.append((center_idx, word2idx[tokens[i - j]]))
            if i + j < N and tokens[i + j] in word2idx:
                pairs.append((center_idx, word2idx[tokens[i + j]]))
    return pairs

pairs = generate_pairs(tokens_train, window_size=2)
print("Number of training pairs:", len(pairs))

# --------- 4. Dataset and DataLoader ---------
class SkipGramDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)

dataset = SkipGramDataset(pairs)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# --------- 5. SkipGram Model ---------
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)
    def forward(self, center_words):
        e = self.emb(center_words)
        logits = self.decoder(e)
        return logits

embed_dim = 100
model = SkipGramModel(vocab_size, embed_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------- 6. Training ---------
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for centers, contexts in dataloader:
        optimizer.zero_grad()
        logits = model(centers)
        loss = loss_fn(logits, contexts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# --------- 7. POS tagging ---------
# Tag all tokens (train+test)
tagged_all = pos_tag(tokens_all)
tagged_all = [(w, map_tag('en-ptb', 'universal', t)) for (w, t) in tagged_all]

# Extract embeddings for tokens in vocab (only train vocab)
X = []
y = []
words_for_eval = []
tags = []

for w, tag in tagged_all:
    if w in word2idx:
        X.append(model.emb.weight[word2idx[w]].detach().cpu().numpy())
        y.append(tag)
        tags.append(tag)
        words_for_eval.append(w)

X_array = np.array(X)
X_tensor = torch.tensor(X_array, dtype=torch.float32)

tagset = sorted(set(tags))
tag2idx = {t: i for i, t in enumerate(tagset)}
idx2tag = {i: t for t, i in tag2idx.items()}
y_idx = torch.tensor([tag2idx[t] for t in y], dtype=torch.long)

# Split train/test for POS probe (80/20)
num_samples = len(y_idx)
indices = list(range(num_samples))
random.shuffle(indices)
split = int(0.8 * num_samples)
train_idx = indices[:split]
test_idx = indices[split:]

X_train, y_train = X_tensor[train_idx], y_idx[train_idx]
X_test, y_test = X_tensor[test_idx], y_idx[test_idx]
words_test = [words_for_eval[i] for i in test_idx]

# --------- 8. Baseline accuracy ---------
major_class = Counter(y_train.tolist()).most_common(1)[0][0]
baseline_acc = (y_test == major_class).float().mean().item()

# --------- 9. MLP probe ---------
class POSProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

probe = POSProbe(X_train.size(1), len(tagset))
optimizer_probe = torch.optim.Adam(probe.parameters(), lr=0.01)
loss_probe = nn.CrossEntropyLoss()

probe_epochs = 10
for epoch in range(probe_epochs):
    optimizer_probe.zero_grad()
    logits = probe(X_train)
    loss = loss_probe(logits, y_train)
    loss.backward()
    optimizer_probe.step()

with torch.no_grad():
    preds = probe(X_test).argmax(dim=1)
    probe_acc = (preds == y_test).float().mean().item()

print(f"Baseline accuracy: {baseline_acc:.4f}")
print(f"Probe accuracy: {probe_acc:.4f}")
print("Number of POS tags:", len(tagset))
print("Tags:", tagset)

# --------- 10. Show example predictions ---------
print("\nSample predictions (word | true POS | predicted POS) for words with length >= 4:")
count = 0
for i in range(len(words_test)):
    word = words_test[i]
    if len(word) >= 4:
        true_tag = idx2tag[y_test[i].item()]
        pred_tag = idx2tag[preds[i].item()]
        print(f"{word:<12} | {true_tag:<5} | {pred_tag}")
        count += 1
        if count >= 20:  # limit output to 10 examples
            break


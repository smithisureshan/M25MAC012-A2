import torch
import torch.nn as nn
import torch.optim as optim
import random

def data(file):
    with open(file, 'r', encoding='utf-8') as f:#it reads all names 
        names = [line.strip() for line in f if line.strip()]
    #Create set of all uniques characters
    chars = sorted(list(set(''.join(names) + '.')))
    char_index = {ch: i for i, ch in enumerate(chars)} #character to index mapping 
    index_char = {i: ch for i, ch in enumerate(chars)} #index to character mapping
    return names, char_index, index_char, len(chars)
#Vanilla RNN
class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, h_size):
        super().__init__()
        self.h_size = h_size#hidden layer size
        self.input_hidden = nn.Parameter(torch.randn(vocab_size, h_size) * 0.01)#input to hidden weights
        self.hidden_hidden = nn.Parameter(torch.randn(h_size, h_size) * 0.01)#hidden to hidden 
        self.hidden_output = nn.Parameter(torch.randn(h_size, vocab_size) * 0.01)#hidden to output
        self.bias_hidden = nn.Parameter(torch.zeros(1, h_size))#it is bias terms
        self.bias_output = nn.Parameter(torch.zeros(1, vocab_size))

    def forward(self, current, previous):
        input = torch.zeros(1, self.input_hidden.size(0))#Convert input index to one hot vector
        input[0, current] = 1
        previous = torch.tanh(input @ self.input_hidden + previous @ self.hidden_hidden + self.bias_hidden) # Compute new hidden state
        output = previous @ self.hidden_output + self.bias_output # Compute output logits
        return output, previous
#Bidirectional LSTM
class BidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, h_size):
        super().__init__()
        self.h_size = h_size
        self.vocab_size = vocab_size

        self.Wf = nn.Linear(vocab_size + h_size, 4 * h_size)# Forward LSTM layer
        self.Wb = nn.Linear(vocab_size + h_size, 4 * h_size)# Backward LSTM layer

        self.hidden_output = nn.Linear(2 * h_size, vocab_size)#it combines both forward and backward LSTM Layer

    def lstm(self, inputs, hidden_old, cell_old, layer):
        merged = torch.cat((inputs, hidden_old), dim=1)#it combines input and previous hidden
        gates = layer(merged)#it computes gates

        i, f, g, o = gates.chunk(4, dim=1)#it splits into four gates

        i = torch.sigmoid(i)#input gate
        f = torch.sigmoid(f)#forget gate 
        o = torch.sigmoid(o)#output gate 
        g = torch.tanh(g)

        cell_new = f * cell_old + i * g
        hidden_new = o * torch.tanh(cell_new)#update hidden state

        return hidden_new, cell_new

    def forward(self, sequence, char_index):
        x_seq = []
        for ch in sequence:#convert entire sequence into one hot vector
            x = torch.zeros(1, self.vocab_size)
            x[0, char_index[ch]] = 1
            x_seq.append(x)
#forward pass
        hf = torch.zeros(1, self.h_size)
        cf = torch.zeros(1, self.h_size)
        forward_states = []

        for x in x_seq:
            hf, cf = self.lstm(x, hf, cf, self.Wf)
            forward_states.append(hf)
#backward pass
        hb = torch.zeros(1, self.h_size)
        cb = torch.zeros(1, self.h_size)
        backward_states = []

        for x in reversed(x_seq):
            hb, cb = self.lstm(x, hb, cb, self.Wb)
            backward_states.insert(0, hb)
#combine both directions
        outputs = []
        for hf, hb in zip(forward_states, backward_states):
            merged = torch.cat((hf, hb), dim=1)
            out = self.hidden_output(merged)
            outputs.append(out)

        return outputs
#Attention RNN
class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, h_size):
        super().__init__()
        self.h_size = h_size
        self.rnn = VanillaRNN(vocab_size, h_size)
        self.combined_weight = nn.Parameter(torch.randn(2 * h_size, vocab_size) * 0.01)#Combine hidden and context
        self.bias_output = nn.Parameter(torch.zeros(1, vocab_size))

    def forward(self, index, previous, forward_cell):
        output_rnn, previous = self.rnn(index, previous)

        if not forward_cell:
            context = torch.zeros(1, self.h_size)
        else:
            old_tensor = torch.stack(forward_cell).squeeze(1)
            attention_weights = torch.softmax(old_tensor @ previous.T, dim=0)#compute attention weights
            context = attention_weights.T @ old_tensor#compute context vector
#combine hidden state and context
        merged = torch.cat((previous, context), dim=1)
        output = merged @ self.combined_weight + self.bias_output# final output
        return output, previous
#Training
def train_model(model, names, char_index, epochs=25, lr=0.003):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total = 0
        random.shuffle(names)

        for name in names:
            name_seq = name + '.'
            loss_total = 0

            if isinstance(model, BidirectionalLSTM):
                outputs = model(name_seq[:-1], char_index)

                for i, output in enumerate(outputs):
                    target = name_seq[i + 1]
                    weight = 0.3 if (target == '.' and i < len(name_seq) - 3) else 1.0 #Penalize early stopping
                    loss_total += weight * criterion(output, torch.tensor([char_index[target]]))
# Normal RNN
            else:
                hidden_old = torch.zeros(1, model.h_size)
                forward_cell = []

                for i in range(len(name_seq) - 1):
                    if isinstance(model, AttentionRNN):
                        output, hidden_old = model(char_index[name_seq[i]], hidden_old, forward_cell)
                        forward_cell.append(hidden_old)
                    else:
                        output, hidden_old = model(char_index[name_seq[i]], hidden_old)

                    target = torch.tensor([char_index[name_seq[i + 1]]])
                    loss_total += criterion(output, target)
#Backpropagation
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) #prevent exploding gradients
            optimizer.step()

            total += loss_total.item()

        print(f"Epoch {epoch+1}, Loss: {total/len(names):.4f}")

def generate_blstm(model, char_index, index_char, max_len=20):
    model.eval()

    for _ in range(10):
        with torch.no_grad():
            start = [c for c in char_index if c.isalpha() and c.isupper()]
            name = random.choice(start)

            for _ in range(max_len):
                outputs = model(name, char_index)
                last = outputs[-1]

                probs = torch.softmax(last / 1.2, dim=1).squeeze()
                #Top-k sampling 
                top_probs, top_idx = torch.topk(probs, 5)
                top_probs = top_probs / top_probs.sum()

                idx = top_idx[torch.multinomial(top_probs, 1)].item()
                char = index_char[idx]
#Stop condition
                if char == '.' and len(name) > 5:
                    break
                elif char == '.':
                    continue

                name += char

            if len(name) >= 3:
                return name

    return name

def generate(model, char_index, index_char, max_len=20):
    model.eval()
    with torch.no_grad():

        start = [c for c in char_index if c.isalpha() and c.isupper()]
        present_index = char_index[random.choice(start)]
        new_name = index_char[present_index]

        hidden_old = torch.zeros(1, model.h_size)
        forward_cell = []

        for _ in range(max_len):
            if isinstance(model, AttentionRNN):
                output, hidden_old = model(present_index, hidden_old, forward_cell)
                forward_cell.append(hidden_old)
            else:
                output, hidden_old = model(present_index, hidden_old)

            probs = torch.softmax(output / 1.2, dim=1).squeeze()
            present_index = torch.multinomial(probs, 1).item()

            char = index_char[present_index]
            if char == '.':
                if len(new_name) > 5:
                    break
                else:
                    continue

            new_name += char

        if len(new_name) < 3:
            return generate(model, char_index, index_char)

        return new_name
#Evaluation Metrics
def evaluate(new_names, training_names):
    training_set = set(n.lower() for n in training_names)
    name_list = [n.lower() for n in new_names]
#Novelty
    novelty = [n for n in name_list if n not in training_set]
    novelty_rate = (len(novelty) / len(name_list)) * 100
#Diversity
    diversity_score = len(set(name_list)) / len(name_list)

    return novelty_rate, diversity_score

if __name__ == "__main__":
    names, char_to_index, index_to_char, vocabulary_size = data('TrainingNames.txt')#Load dataset 
    h_size = 64 #hidden layer
#initialize all models
    models = [
        VanillaRNN(vocabulary_size, h_size),
        BidirectionalLSTM(vocabulary_size, h_size),
        AttentionRNN(vocabulary_size, h_size)
    ]
#iterate through each model
    for model in models:
        model_name = model.__class__.__name__
        total_parameters = sum(p.numel() for p in model.parameters())# total trainable parameters

        print(f"\nTraining {model_name} | Parameters: {total_parameters}")

        if isinstance(model, BidirectionalLSTM):
            train_model(model, names, char_to_index, epochs=10)
            gen_names = [generate_blstm(model, char_to_index, index_to_char) for _ in range(1000)]
        else:
            train_model(model, names, char_to_index, epochs=20)
            torch.save(model.state_dict(), f'{model_name}.pth')#save trained model weights to file
            gen_names = [generate(model, char_to_index, index_to_char) for _ in range(1000)]

        print(f"\nSample Names from {model_name}:")
        for i in range(20):# show first 20 names
            print(gen_names[i])

        novelty_final, diversity_final = evaluate(gen_names, names)
        print(f"\n[{model_name}] Novelty: {novelty_final:.2f}% | Diversity: {diversity_final:.2f}")
from train import TransformerModel, input_size, d_model, nhead, num_layers, PhonemesDataset
import torch
model = TransformerModel(input_size, d_model, nhead, num_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load('10.cp'))
dataset = PhonemesDataset()

for i in range(10):
    x=dataset[i]
    x=x.unsqueeze(0)
    x=x.to(device)
    model.eval()
    output = model(x)
    output = output.squeeze(0)
    predicted_labels = torch.argmax(output, dim=1)
    print("real")
    print(x)
    print(predicted_labels)
    print("masking real")
    mask = torch.randn(x.shape) <= 0.2
    mask[x == input_size-1] = False
    random_tokens = torch.randint_like(x, input_size)
    x[mask] = random_tokens[mask]
    output = model(x)
    output = output[x]
    x=x[mask]
    predicted_labels = torch.argmax(output, dim=1)
    print("masking predicted")
    print(x)
    print(predicted_labels)


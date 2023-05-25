from train import max_len,input_size,d_model,num_layers,nhead,PhonemesDataset,mask_value,padding_value
from x_transformers import TransformerWrapper, Encoder
import torch

model = TransformerWrapper(
num_tokens = input_size+1,
max_seq_len = max_len,
    attn_layers = Encoder(
        dim = d_model,
        depth = num_layers,
        heads = nhead
    )
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('0.cp',map_location=torch.device('cpu')))

model = model.to(device)
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
    mask = torch.randn(x.shape) <= 0.5
    mask[x == padding_value] = False
    random_tokens = torch.randint_like(x, input_size)
    y=x.clone()
    x[mask] = random_tokens[mask]
    output = model(x)
    output = output[mask]
    y=y[mask]
    predicted_labels = torch.argmax(output, dim=1)
    print("masking predicted")
    print(y)
    print(predicted_labels)
    correct_predictions = (predicted_labels == y).sum().item()
    print("accuracy",correct_predictions/len(y))



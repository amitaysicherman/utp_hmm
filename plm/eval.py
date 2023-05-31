from train import max_len, input_size, d_model, num_layers, nhead, PhonemesDataset, mask_value, padding_value
from x_transformers import TransformerWrapper, Encoder
import torch
from train import  get_model
cp_file = "./models/best.cp"
data_path = 'LR960_PH.npz' # 'TIMIT_PH.npz'
data_len_path ='LR960_PH_LEN.txt' #"TIMIT_PH_LEN.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load(cp_file, map_location=torch.device('cpu')))

model = model.to(device)
dataset = PhonemesDataset(data_path=data_path, data_len_path=data_len_path)

for i in range(10):
    x = dataset[i]
    x = x.unsqueeze(0)
    x = x.to(device)
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
    random_tokens = torch.randint_like(x, input_size)*0+mask_value
    y = x.clone()
    x[mask] = random_tokens[mask]
    output = model(x)
    output = output[mask]
    y = y[mask]
    predicted_labels = torch.argmax(output, dim=1)
    print("masking predicted")
    print(y)
    print(predicted_labels)
    correct_predictions = (predicted_labels == y).sum().item()
    print("accuracy", correct_predictions / len(y))

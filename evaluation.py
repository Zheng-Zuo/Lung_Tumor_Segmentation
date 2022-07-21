from pathlib import Path
import torch
import numpy as np
from dataset import LungDataset
from train import TumorSegmentation
from tqdm import tqdm

class DiceScore(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
                
        #flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)
        
        counter = (pred * mask).sum()  # Counter       
        denum = pred.sum() + mask.sum() + 1e-8  # denominator
        dice = (2*counter)/denum
        
        return dice

model = TumorSegmentation.load_from_checkpoint("weights/epoch=29-step=53759.ckpt")
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

preds = []
labels = []

val_path = Path("Task06_Lung/Preprocessed/val/")
val_dataset = LungDataset(val_path, None)

for slice, label in tqdm(val_dataset):
    slice = torch.tensor(slice).float().to(device).unsqueeze(0)
    with torch.no_grad():
        pred = torch.sigmoid(model(slice))
    preds.append(pred.cpu().numpy())
    labels.append(label)
    
preds = np.array(preds)
labels = np.array(labels)

dice_score = DiceScore()(torch.from_numpy(preds), torch.from_numpy(labels).unsqueeze(0).float())
print(f"The Val Dice Score is: {dice_score}")
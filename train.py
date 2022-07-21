from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import imgaug.augmenters as iaa
import numpy as np
from tqdm import tqdm
from dataset import LungDataset
from model import UNet
import matplotlib.pyplot as plt

class TumorSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = UNet()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, data):
        pred = self.model(data)
        return pred
    
    def training_step(self, batch, batch_idx):
        ct, mask = batch
        mask = mask.float()
        
        pred = self(ct)
        loss = self.loss_fn(pred, mask)
        
        # Logs
        self.log("Train Loss", loss)
        if batch_idx % 50 == 0:
            self.log_images(ct.cpu(), pred.cpu(), mask.cpu(), "Train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        ct, mask = batch
        mask = mask.float()

        pred = self(ct)
        loss = self.loss_fn(pred, mask)
        
        # Logs
        self.log("Val Loss", loss)
        if batch_idx % 2 == 0:
            self.log_images(ct.cpu(), pred.cpu(), mask.cpu(), "Val")
        return loss

    def log_images(self, ct, pred, mask, name):
        
        results = []
        
        pred = pred > 0.5 
        
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(ct[0][0], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][0]==0, mask[0][0])
        axis[0].imshow(mask_, alpha=0.6)
        axis[0].set_title("Ground Truth")
        
        axis[1].imshow(ct[0][0], cmap="bone")
        mask_ = np.ma.masked_where(pred[0][0]==0, pred[0][0])
        axis[1].imshow(mask_, alpha=0.6, cmap="autumn")
        axis[1].set_title("Pred")

        self.logger.experiment.add_figure(f"{name} Prediction vs Label", fig, self.global_step)
 
    def configure_optimizers(self):
        return [self.optimizer]

seq = iaa.Sequential([
        iaa.Affine(translate_percent=(0.15), 
                scale=(0.85, 1.15), # zoom in or out
                rotate=(-45, 45)#
                ),  # rotate up to 45 degrees
        iaa.ElasticTransformation()  # Elastic Transformations
                    ])

# Create the dataset objects
train_path = Path("Task06_Lung/Preprocessed/train/")
val_path = Path("Task06_Lung/Preprocessed/val/")

train_dataset = LungDataset(train_path, seq)
val_dataset = LungDataset(val_path, None)

# Oversampling to handle class imbalance
target_list = []
for _, label in tqdm(train_dataset):
    # Check if mask contains a tumorous pixel:
    if np.any(label):
        target_list.append(1)
    else:
        target_list.append(0)

uniques = np.unique(target_list, return_counts=True)
fraction = uniques[1][0] / uniques[1][1]

weight_list = []
for target in target_list:
    if target == 0:
        weight_list.append(1)
    else:
        weight_list.append(fraction)

sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_list, len(weight_list))  

# Create dataloader
batch_size = 8
num_workers = 4

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                        batch_size=batch_size,
                                        num_workers=num_workers, 
                                        sampler=sampler)
val_loader = torch.utils.data.DataLoader(val_dataset, 
                                        batch_size=batch_size, 
                                        num_workers=num_workers, 
                                        shuffle=False)

if __name__ == '__main__':

    torch.mannual_seed(0)
    model = TumorSegmentation()

    # Create the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='Val Loss',
        save_top_k=30,
        mode='min')

    gpus = 1 
    trainer = pl.Trainer(gpus=gpus, logger=TensorBoardLogger(save_dir="./logs"), log_every_n_steps=1,
                        callbacks=checkpoint_callback,
                        max_epochs=30)

    trainer.fit(model, train_loader, val_loader)


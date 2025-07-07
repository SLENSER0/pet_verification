import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pytorch_metric_learning import losses
from oml.samplers import BalanceSampler
from transformers import AutoModel, AutoProcessor
from sklearn.metrics import roc_curve
from PIL import Image
import wandb
from pytorch_metric_learning.distances import CosineSimilarity
import torch.nn.functional as F
from accelerate import Accelerator 

config = {
    "epochs": 30,
    "learning_rate": 1e-4,
    "num_workers": 12,
    "n_labels": 128  ,
    "n_instances": 2,
    "margin": 0.45,
    "seed": 0,
}

def fix_seed(seed: int):
    """Fix Python, NumPy and PyTorch RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def custom_collate_fn(batch):
    images = [item['image'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    return {'image': images, 'label': labels}

class ImageDs(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        label = int(row["label"])
        image = Image.open(path).convert("RGB")
        return {'image': image, 'label': label}

    def get_labels(self):
        return np.array(self.df['label'])

class FaceRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        
        ckpt = "zer0int/CLIP-GmP-ViT-L-14"
        self.clip = AutoModel.from_pretrained(ckpt)
        self.processor = AutoProcessor.from_pretrained(ckpt)
        
        for param in self.clip.parameters():
            param.requires_grad = False
        
        for layer in self.clip.vision_model.encoder.layers[-5:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, images):
        clip_inputs = self.processor(images=images, return_tensors="pt")
        device = next(self.clip.parameters()).device
        clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
        
        clip_features = self.clip.get_image_features(**clip_inputs)
        return clip_features
def training():
    accelerator = Accelerator()
    
    fix_seed(config["seed"])
    
    if accelerator.is_main_process:
        wandb.init(project="pet_rec_hack", name="zeroint_comb", config=config)
    
    model = FaceRecognizer() 
    
    df_train = pd.read_csv("/home/user1/vasiliy/dataset_processing/train_comb.csv")
    train_dataset = ImageDs(df_train)
    
    main_loss = losses.TripletMarginLoss(margin=config["margin"], distance=CosineSimilarity())
    var_loss = losses.IntraPairVarianceLoss()
    complete_loss = losses.MultipleLosses([main_loss, var_loss], weights=[1, 0.5])
    
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    sampler = BalanceSampler(train_dataset.get_labels(),
                             n_labels=config["n_labels"],
                             n_instances=config["n_instances"])
    
    dataloader = DataLoader(train_dataset,
                           batch_sampler=sampler,
                           collate_fn=custom_collate_fn)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    ckpt_dir = "zer0int_comb"
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
    
    for epoch in range(config["epochs"]):
        model.train()
        
        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch: {epoch + 1}/{config['epochs']}")
        else:
            pbar = dataloader
        
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in pbar:
            images = batch["image"]
            labels = batch["label"]  
            
            embeddings = model(images)
            loss = complete_loss(embeddings, labels)
            
            if accelerator.is_main_process:
                wandb.log({"train_loss": loss.item()})
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            if accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({"batch_loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / n_batches
        
        if accelerator.is_main_process:
            wandb.log({"epoch": epoch, "avg_epoch_loss": avg_epoch_loss})
            
            ckpt_path = os.path.join(ckpt_dir, f"clip_face_rec_epoch_{epoch + 1:03d}_crop.pth")
            
            accelerator.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
            wandb.save(ckpt_path)
            
            print(f"Epoch {epoch + 1} finished. Avg loss: {avg_epoch_loss:.4f}. Saved checkpoint to {ckpt_path}")
        
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    training()

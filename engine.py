import torch
from torch.utils.data import Dataset
from utils import weight_map
from torch import nn
import numpy as np


def train(model, optimizer, dataloader, criterion, effective_batch_size, p_bar = None):
    ''' Training '''
    
    model.train()
    optimizer.zero_grad()
    running_loss = 0
    
    for batch_id, (X, y, weights) in enumerate(dataloader):
        
        if p_bar is not None: 
            p_bar.set_description_str(f'Batch {batch_id + 1}')
            
        # Forward
        y_hat = model(X)
        
        # Compute loss
        loss = criterion(y, y_hat, weights) / effective_batch_size
        running_loss += loss.item()
        loss.backward()
        
        # Backprop
        if ( (batch_id + 1) % effective_batch_size == 0 ) or ( (batch_id + 1) == len(dataloader) ):
            optimizer.step()
            optimizer.zero_grad()
        
        # Update progress bar
        if p_bar is not None:
            p_bar.set_postfix(loss = loss.item())
            p_bar.update(1)
    
    # Compute average loss
    running_loss = running_loss /  len(dataloader) * effective_batch_size
        
    return running_loss


def validation(model, dataloader, criterion):
    ''' Validation '''
    
    # Validation
    model.eval()
    running_loss = 0
    
    with torch.no_grad():
        for X, y, weights in dataloader:

            # Forward
            y_hat = model(X)

            # Compute loss
            loss = criterion(y, y_hat, weights)
            running_loss += loss.item()

    # Compute average loss
    running_loss /= len(dataloader)
    
    return running_loss


class EarlyStopping(object):
    '''Early Stopping'''
    
    def __init__(self, patience, fname):
        self.patience  = patience
        self.best_loss = np.Inf
        self.counter   = 0
        self.filename  = fname
        
    def __call__(self, epoch, loss, optimizer, model):
        
        if loss < self.best_loss:
            self.counter = 0
            self.best_loss = loss
            
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                 loss,
            }, self.filename)
            
        else:
            self.counter += 1
        
        return self.counter == self.patience


class WeightedBCEWithLogitsLoss(nn.Module):
    ''' Pixel-wise weighted BCEWithLogitsLoss'''
    
    def __init__(self, batch_size):
        
        super().__init__()
        self.batch_size = batch_size
        self.unw_loss = nn.BCEWithLogitsLoss(reduction = 'none')
    
    def __call__(self, true, predicted, weights):
        
        # Compute weighted loss
        loss = self.unw_loss(predicted, true) * weights
        
        # Sum over all channels
        loss = loss.sum(dim = 1)
        
        # Flatten and rescale so that loss is approx. in the same interval
        loss = loss.view(self.batch_size, -1) / weights.view(self.batch_size, -1)
        
        # Average over mini-batch
        loss = loss.mean()

        return loss
    

class SegmentationDataset(Dataset):
    
    def __init__(self, images, masks, wmap_w0, wmap_sigma, device, transform = None):
        ''' Initialisation function '''
        
        self.images    = images
        self.masks     = masks
        self.transform = transform
        self.device    = device
        
        # Parameters for weight map calculation
        self.w0    = wmap_w0
        self.sigma = wmap_sigma

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ''' Preprocess and return image, mask, and weight map '''
        
        image = self.images[idx, :, :]
        mask  = self.masks[idx, :, :]
        
        if self.transform:
            
            # Apply transformations
            aug   = self.transform(image = image, mask  = mask)
            image = aug["image"]
            mask  = aug["mask"]
        
        # Compute weight map
        weights = weight_map(mask = mask, w0 = self.w0, sigma = self.sigma)
        
        # Min-max scale image and mask
        image = self.min_max_scale(image, min_val = 0, max_val = 1)
        mask  = self.min_max_scale(mask,  min_val = 0, max_val = 1)
        
        # Add channel dimensions
        image   = np.expand_dims(image,   axis = 0)
        weights = np.expand_dims(weights, axis = 0)
        mask    = np.expand_dims(mask,    axis = 0)
        
        # Convert to tensors and send to device
        weights = torch.from_numpy(weights).double().to(self.device)
        image   = torch.from_numpy(image).double().to(self.device)
        mask    = torch.from_numpy(mask).double().to(self.device)
        
        # Center crop mask and weights (negative padding = cropping - size defined manually)
        mask    = nn.ZeroPad2d(-94)(mask)
        weights = nn.ZeroPad2d(-94)(weights)
        
        return image, mask, weights
    
    @staticmethod
    def min_max_scale(image, max_val, min_val):
        '''Normalization to range of min, max'''
        
        image_new = (image - np.min(image)) * (max_val - min_val) / (np.max(image) - np.min(image)) + min_val
        return image_new
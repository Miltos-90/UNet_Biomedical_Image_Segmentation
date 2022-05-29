from torch import optim
from tqdm.notebook import tqdm

class LRTest(object):
    
    def __init__(self, min_lr, max_lr, no_iter, batch_size):
        ''' Initialisation function '''
        
        self.batch_size    = batch_size
        self.no_iter       = no_iter
        self.lr_multiplier = (max_lr / min_lr) ** (1 / (no_iter))
        self.dataiter      = None
        

    # Function to perform the learning rate range test on one experiment
    def __call__(self, dataloader, criterion, optimizer, model):
        ''' LR Range test '''
        
        # Set model to training mode
        model.train()
        
        # Configure scheduler
        scheduler  = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.lr_multiplier)
        
        # Empty lists to hold results
        loss_arr, lr_arr     = [], []
        
        # Counters' initialisation
        cur_iter, best_loss  = 0, 1e9
        
        with tqdm(total = self.no_iter) as pbar:
            
            while cur_iter < self.no_iter:
                
                # Grab learning rate (before stepping the scheduler)
                lr_arr.append(scheduler.get_lr())
                
                # Train a batch
                cur_loss = self.train_batch(model, criterion, optimizer, scheduler, dataloader)
                
                # Append loss
                loss_arr.append(cur_loss)

                # Check for divergence and exit if needed
                if cur_loss < best_loss: 
                    best_loss = cur_loss

                if cur_loss > 2e2 * best_loss: # Divergence
                    print('Diverged on iteration ' + str(cur_iter) + ' with loss ' + str(cur_loss))
                    break

                # Update progress bar
                pbar.set_postfix(loss = cur_loss)
                pbar.update(1)
                cur_iter += 1

        pbar.close() # Close

        return lr_arr, loss_arr
    
    
    # Return a batch
    def grab_batch(self, dataloader):
        
        # Lazy init
        if self.dataiter is None:
            self.dataiter = iter(dataloader)
        
        # Get next batch
        try:
            X, y, w = next(self.dataiter)
            
        except: # End of dataset -> restart
            
            self.dataiter = iter(dataloader)
            X, y, w  = next(self.dataiter)
        
        return X, y, w
    
    
    # Train batch
    def train_batch(self, model, criterion, optimizer, scheduler, dataloader):

        optimizer.zero_grad()
        
        cur_iter = 0
        run_loss = 0
        while cur_iter < self.batch_size:
        
            # Get sample
            X, y, w = self.grab_batch(dataloader)

            # Predict
            y_hat = model(X)
            
            # Compute normalised gradients
            loss  = criterion(y, y_hat, w) / self.batch_size
            run_loss += loss.item()
            
            # Backprop
            loss.backward()
            
            # Update counter
            cur_iter += 1
        
        # Update
        optimizer.step()
        scheduler.step()

        return run_loss

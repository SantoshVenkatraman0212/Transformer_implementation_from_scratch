'''
This file depicts the entire training pipeline of the transformer model
'''
# Importing necessary libraries
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from config.settings import (DEVICE, D_MODEL, N_BLOCKS, N_HEADS, SRC_VOCAB_SIZE, 
                             TGT_VOCAB_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, DROPOUT, WARMUP_STEPS, N_EPOCHS, PATIENCE)
from config.paths import CHECKPOINT_DIR
from data_pipeline.dataloader import create_dataloaders
from transformer.transformer_model import Transformer
from training.loss import TranslationLoss


def create_transformer(src_vocab_size: int, tgt_vocab_size: int, d_model: int, n_heads: int,  n_blocks: int, 
                 src_max_seq_len: int, tgt_max_seq_len: int, dropout: float) -> Transformer:
    '''
    This function is specifically for creating and returning a Transformer model object.
    The transformer model instance is created with the arguments that the constructor accepts.
    Then the model is moved to the DEVICE which is GPU or CPU based on availability and is returned

    Args:
        src_vocab_size: int
            No of unique Deutsch tokens
        tgt_vocab_size: int
            No of unique English tokens
        d_model: int
            Embedding dimensions
        n_heads: int
            No of attention heads
        n_blocks: int
            No of encoder & decoder blocks, 
        src_max_seq_len: int
            Max context length for Deutsch
        tgt_max_seq_len: int
            Max context length for English
        dropout: float
            Dropout value for avoiding overfitting (Especially in FFN)
    
    Returns:
        A Transformer model instance with the specified hyperparams

    '''
    # Creating model instance
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads,  n_blocks, 
                 src_max_seq_len, tgt_max_seq_len, dropout)

    # Returning the model by moving it to the compute device
    return model.to(DEVICE)

def lambda_lr_scheduler(step: int) -> float:
    '''
    This function adjusts the learning rate according to the training and warmup steps.
    lr = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    '''
    # For 0th step, the function returns 0 and lr starts at 0.001
    if step == 0:
        return 0
    # For non-zero steps the learning rate is give by the "Noam" scheduling technique from
    # Attention Is All You Need 2017 paper
    lr = (D_MODEL ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))

    return lr
    

def train(model: Transformer, train_dataloader: DataLoader, pad_id: int, device: torch.device, 
          criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: LambdaLR) -> float:
    '''
    This function runs the training steps for 1 epoch for the transformer model.
    The following steps are performed:
    1. Model is set to training mode 
    2. Batch-wise iteration on the train dataloader
    3. Moving encoder_input, decoder_input, and label tensors to compute device
    4. Getting the source and target padding masks
    5. Resetting the Adam optimizer to prevent multi-epoch gradient accumulation
    6. Logits are computed by forward pass of the model (probability distribution vector)
    7. Cross entropy loss 
    8. Backpropagation of the loss
    9. Weights & scheduler updation
    10. Overall avg loss across all batches for 1 epoch

    Args:
        model: Transformer
            model instance
        train_dataloader: DataLoader
            batchwise-iterable for train dataset
        pad_id: int
            token ID for pad token
        device: torch.device
            Compute device for training
        
        criterion: nn.Module
            Loss function
        optimizer: torch.optim.Optimizer
            Optimizer for regulating the learning rate, and kicking off backpropagation
        scheduler: LambdaLR
            Scales the learning rate per step
    
    Returns:
        training loss: float
    '''
    # Setting the model to train mode
    model.train()
    # Initializing batch_loss
    batch_loss, loss = 0, 0
    # Creating a tqdm based progress bar for tracking training progress per step
    progress_bar = tqdm(train_dataloader, desc = 'Training', unit = 'Batch')
    for batch in progress_bar:
        # Moving the tensors to the device
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        label = batch['label'].to(device)
        # Getting padding masks
        src_padding_mask = encoder_input != pad_id 
        tgt_padding_mask = decoder_input != pad_id
        # Resetting the optimizer
        optimizer.zero_grad()
        # Logits
        logits = model(encoder_input, decoder_input, src_padding_mask, tgt_padding_mask)
        # Cross Entropy loss per batch
        batch_loss = criterion(logits, label)
        # Accumulated loss (per epoch)
        #.item() here prevents computation graph creation for every batch
        loss += batch_loss.item()
        # Back propagation
        batch_loss.backward()
        # Updating the params
        optimizer.step()
        # Updating the scheduler
        scheduler.step()

        # Printing the learning rate, and loss for every 1k batches
        progress_bar.set_postfix(batch_loss = f'{batch_loss:.4f}', lr = f'{optimizer.param_groups[0]['lr']:.4f}')

    # Overall avg loss for 1 epoch
    return loss / len(train_dataloader)

def evaluate(model: Transformer, eval_dataloader: DataLoader, pad_id: int, criterion: nn.Module, device: torch.device) -> float:
    '''
    This function runs the eval loop for the transformer model for 1 epoch.
    This begins after 1 full epoch of model training. The function does the following:
    1. Stopping gradient updation
    2. Setting the model to eval mode
    3. Moving the eval set's encoder_input, decoder_input and labels to compute device
    4. Getting padding masks
    5. Computing logits
    6. Computing batch loss, and accumulating batch-wise losses
    7. Returning average eval loss across all batches for one epoch

    Args:
        model: Transformer
            Transformer model instance
        eval_dataloader: DataLoader
            Batch-wise iterable for eval set    
        pad_id: int
            token ID for pad token
        criterion: nn.Module
            eval loss function
        device: torch.device
            Compute device for evals
    
    Returns:
        eval loss: float
    '''
    # Initializing batch-wise and overall epoch-wise eval loss
    batch_eval_loss, eval_loss = 0, 0
    # Disabling gradient updation
    with torch.no_grad():
        # Setting the model to eval mode
        model.eval()
        progress_bar = tqdm(eval_dataloader, desc = 'Validation', unit = 'Batch')
        # Iterating through each eval dataset batch
        for batch in progress_bar:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            label = batch['label'].to(device)
            # Getting the source and target padding masks
            src_padding_mask = encoder_input != pad_id
            tgt_padding_mask = decoder_input != pad_id
            # Logits on eval set
            logits = model(encoder_input, decoder_input, src_padding_mask, tgt_padding_mask)
            # Batch-wise eval loss
            batch_eval_loss = criterion(logits, label)
            # Accumulated batch-wise eval loss (sum of losses across all batches)
            eval_loss += batch_eval_loss.item()
            progress_bar.set_postfix(batch_loss = f'{batch_eval_loss:.4f}')
    # Average eval loss for 1 epoch
    return eval_loss / len(eval_dataloader)
        

def save_model_checkpoint(model: Transformer, lr: float, optimizer: torch.optim.Optimizer, 
                          scheduler: LambdaLR, train_loss: float, val_loss: float, epoch: int, save_path: Path) -> None:
    '''
    This function saves model checkpoints as .pt every epoch, and also
    the best model checkpoints when the eval loss is loss than its previous epoch value 
    
    Args:
        model: Transformer
            Transformer model instance
        lr: float
            Learning rate for controlling model convergence and params updation rate
        optimizer: torch.optim.Optimizer
            Adam optimizer
        scheduler: LambdaLR
            Custom learning rate scheduler function
        train_loss: float
            Training loss
        val_loss: float
            Eval loss
        epoch: int
            Epoch number
        save_path: Path
            Checkpoint save path
    '''
    # Creating if the checkpoint dir doesn't exist
    save_path.parent.mkdir(parents = True, exist_ok = True)
    # Model checkpoint saving
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                'scheduler_state_dict': scheduler.state_dict(), 'learning_rate': lr, 'train_loss': train_loss, 
                'val_loss': val_loss, 'epoch': epoch}, save_path)

def main():
    '''
    Orchestrator for the transformer model training pipeline
    '''
    # Getting the dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders()
    # Getting the pad ID
    pad_id = train_dataloader.dataset.pad_id
    # model instance
    model = create_transformer(src_vocab_size = SRC_VOCAB_SIZE, tgt_vocab_size = TGT_VOCAB_SIZE, d_model = D_MODEL, 
                               n_heads = N_HEADS, n_blocks = N_BLOCKS, src_max_seq_len = SRC_SEQ_LEN, 
                               tgt_max_seq_len = TGT_SEQ_LEN, dropout = DROPOUT)
    # Transformer model architecture
    total_params = sum([params.numel() for params in model.parameters()])
    trainable_params = sum([params.numel() for params in model.parameters() if params.requires_grad])
    print(f'Total no of params: {total_params}')
    print(f'No of trainables params: {trainable_params}')
    
    # Loss
    loss = TranslationLoss(pad_id = pad_id)
    # Optimizer
    optimizer = Adam(model.parameters(), lr = 1.0, betas = (0.9, 0.98), eps = 1e-9)
    # Learning rate scheduler
    lr_scheduler = LambdaLR(optimizer = optimizer, lr_lambda = lambda_lr_scheduler)

    # Starting training
    # Initializing epoch wise best val loss as inf
    best_val_loss = float('inf')
    print(f'Transformer training will run on: {DEVICE}')
    # Initializing early stopping patience
    patience = 0
    # Beginning of training epochs
    for i in range(N_EPOCHS):
        print(f'--- Epoch - {i + 1} ---')
        # Training loss
        train_loss = train(model = model, train_dataloader = train_dataloader, pad_id = pad_id, 
                           device = DEVICE, criterion = loss, optimizer = optimizer, scheduler = lr_scheduler)
        # Val loss
        val_loss = evaluate(model = model, eval_dataloader = val_dataloader, pad_id = pad_id, criterion = loss, device = DEVICE)
        # Getting the perplexity scores as e^loss for that epoch
        train_perplexity = math.exp(train_loss)
        val_perplexity = math.exp(val_loss)
        # Getting the learning rate from optimizer's param_groups list
        lr = optimizer.param_groups[0]['lr']
        # Epoch wise train loss, val loss, learning rate, train perplexity and val perplexity scores
        print(f'''Epoch - {i + 1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Learning Rate: {lr} 
                  | Train Perplexity: {train_perplexity} | Val Perplexity: {val_perplexity} ''')

        # Creating a checkpoint file path for saving checkpoint after every epoch
        # The same file gets overwritten epoch after epoch
        latest_chkpt = CHECKPOINT_DIR / 'latest.pt'
        save_model_checkpoint(model = model, lr = lr, optimizer = optimizer, scheduler = lr_scheduler, 
                              train_loss = train_loss, val_loss = val_loss, epoch = (i + 1), save_path = latest_chkpt)
        # Saving the best checkpoint i.e. if current epoch's val loss is less than previous epoch's val loss
        # If current val loss is less than best_val_loss then best_val_loss is updated to the current val loss
        if val_loss < best_val_loss:
            # If val loss reduces then patience should be reset to 0
            patience = 0
            best_val_loss = val_loss
            best_chkpt = CHECKPOINT_DIR / 'best.pt'
            save_model_checkpoint(model = model, lr = lr, optimizer = optimizer, scheduler = lr_scheduler, 
                                          train_loss = train_loss, val_loss = val_loss, epoch = (i + 1), save_path = best_chkpt)
            print('Model val improved. Best checkpoint saved')
        else:
            # Early stopping implementation
            patience += 1
            # If the eval loss doesn't decrease over 3 continuous
            if patience >= PATIENCE:
                print('Model val loss didn\'t improve over 3 steps\n!!!!! Triggering Early Stopping !!!!!')
                break

if __name__ == '__main__':
    main()


            

    
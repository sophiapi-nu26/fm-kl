"""
Flow matching training implementation.

This module implements the training loop for learning the velocity field
v_θ(x,t) to approximate the true velocity u(x,t) = a(t)x using MSE loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

from ground_truth import GroundTruthPath
from model import VelocityFieldMLP


class FlowMatchingTrainer:
    """Trainer for flow matching velocity field."""
    
    def __init__(self, 
                 model: VelocityFieldMLP,
                 ground_truth: GroundTruthPath,
                 learning_rate: float = 1e-3,
                 batch_size: int = 1024,
                 device: str = "cpu"):
        """
        Initialize trainer.
        
        Args:
            model: Velocity field MLP
            ground_truth: Ground truth path
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            device: Device to use ("cpu" or "cuda")
        """
        self.model = model.to(device)
        self.ground_truth = ground_truth
        self.device = device
        self.batch_size = batch_size
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a training batch.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Tuple of (x, t, u_target) where:
            - x: samples from p_t
            - t: uniform times in [0,1]
            - u_target: true velocity u(x,t) = a(t)x
        """
        # Sample times uniformly from [0,1]
        t = torch.rand(batch_size, dtype=torch.float64, device=self.device)
        
        # Sample x from p_t
        x = self.ground_truth.sample_p(t, batch_size).to(self.device)
        
        # Compute true velocity u(x,t) = a(t)x
        u_target = self.ground_truth.u(x, t)
        
        return x, t, u_target
    
    def train_step(self) -> float:
        """Single training step."""
        self.model.train()
        
        # Sample batch
        x, t, u_target = self.sample_batch(self.batch_size)
        
        # Forward pass
        u_pred = self.model(x, t)
        
        # MSE loss
        loss = nn.MSELoss()(u_pred, u_target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, num_samples: int = 2000) -> float:
        """Compute validation loss."""
        self.model.eval()
        
        with torch.no_grad():
            x, t, u_target = self.sample_batch(num_samples)
            u_pred = self.model(x, t)
            val_loss = nn.MSELoss()(u_pred, u_target)
        
        return val_loss.item()
    
    def train(self, 
              num_epochs: int = 1000,
              val_freq: int = 50,
              target_val_loss: float = 1e-4,
              patience: int = 100) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Maximum number of epochs
            val_freq: Validation frequency
            target_val_loss: Target validation loss (stop if reached)
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        print(f"Training for {num_epochs} epochs...")
        print(f"Target validation loss: {target_val_loss}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(num_epochs)):
            # Training step
            train_loss = self.train_step()
            self.train_losses.append(train_loss)
            
            # Validation
            if epoch % val_freq == 0 or epoch == num_epochs - 1:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                
                print(f"Epoch {epoch:4d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Check if target reached
                if val_loss <= target_val_loss:
                    print(f"Target validation loss reached at epoch {epoch}")
                    break
                
                # Check patience
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} (patience exceeded)")
                    break
        
        # Final validation
        final_val_loss = self.validate()
        print(f"Final validation loss: {final_val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_val_loss': final_val_loss,
            'best_val_loss': best_val_loss,
            'num_epochs': len(self.train_losses)
        }
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training and validation curves."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(len(self.train_losses))
        ax.plot(epochs, self.train_losses, label='Training Loss', alpha=0.7)
        
        if self.val_losses:
            val_epochs = np.linspace(0, len(self.train_losses)-1, len(self.val_losses))
            ax.plot(val_epochs, self.val_losses, label='Validation Loss', marker='o')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Training Progress - {self.ground_truth.schedule_type}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"plots/{self.ground_truth.schedule_type}_training_{timestamp}.pdf"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()  # Close figure instead of showing
        print(f"Training plot saved to: {save_path}")
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'ground_truth_schedule': self.ground_truth.schedule_type,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']


def test_training():
    """Test training implementation."""
    print("Testing flow matching training...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create ground truth and model
    gt = GroundTruthPath("sin_pi")
    model = VelocityFieldMLP(hidden_dim=128, num_layers=3, activation="silu")
    model = model.double()
    
    # Create trainer
    trainer = FlowMatchingTrainer(
        model=model,
        ground_truth=gt,
        learning_rate=1e-3,
        batch_size=512,
        device="cpu"
    )
    
    # Train for a few epochs
    history = trainer.train(num_epochs=100, val_freq=20, target_val_loss=1e-3)
    
    print(f"Training completed in {history['num_epochs']} epochs")
    print(f"Final validation loss: {history['final_val_loss']:.6f}")
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Test model performance
    x_test = torch.randn(100, 2, dtype=torch.float64)
    t_test = torch.rand(100, dtype=torch.float64)
    
    with torch.no_grad():
        u_pred = model(x_test, t_test)
        u_true = gt.u(x_test, t_test)
        
        mse = nn.MSELoss()(u_pred, u_true)
        print(f"Test MSE: {mse:.6f}")


if __name__ == "__main__":
    test_training()

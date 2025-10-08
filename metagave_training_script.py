import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== Configuration ====================

class Config:
    """Configuration for MetaGAVE model and training"""
    
    # Dataset configurations (from Table 1 in paper)
    DATASET_CONFIGS = {
        'SMAP': {
            'n_features': 55,
            'embed_dim': 128,
            'latent_dim': 2,
            'window_size': 30,
            'k_neighbors': 30,
        },
        'MSL': {
            'n_features': 27,
            'embed_dim': 64,
            'latent_dim': 5,
            'window_size': 30,
            'k_neighbors': 30,
        },
        'SMD': {
            'n_features': 28,
            'embed_dim': 128,
            'latent_dim': 5,
            'window_size': 30,
            'k_neighbors': 30,
        },
        'SWAT': {
            'n_features': 51,
            'embed_dim': 128,
            'latent_dim': 4,
            'window_size': 30,
            'k_neighbors': 30,
        }
    }
    
    # Training parameters (from Table 2 in paper)
    DEFAULT_PARAMS = {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'hidden_dim': 128,
        'eta': 0.3,  # Fusion hyperparameter
        'dropout': 0.1,
        'optimizer': 'adam',
        'early_stopping_patience': 15,
        'gradient_clip': 1.0,
    }
    
    def __init__(self, dataset_name='SMAP', **kwargs):
        self.dataset_name = dataset_name
        
        # Load dataset-specific config
        if dataset_name in self.DATASET_CONFIGS:
            self.config = {**self.DATASET_CONFIGS[dataset_name]}
        else:
            self.config = {'n_features': 55, 'embed_dim': 128, 'latent_dim': 4}
        
        # Add default training params
        self.config.update(self.DEFAULT_PARAMS)
        
        # Override with custom params
        self.config.update(kwargs)
        
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config = cls()
        config.config = config_dict
        return config


# ==================== Logger ====================

class Logger:
    """Training logger with tensorboard-like functionality"""
    
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        self.metrics_file = os.path.join(log_dir, f'metrics_{timestamp}.json')
        
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def log(self, message, print_console=True):
        """Log message to file and optionally console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        
        if print_console:
            print(log_message)
    
    def log_metrics(self, epoch, metrics):
        """Log metrics for an epoch"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        self.log(f"Epoch {epoch}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
    
    def save_metrics(self):
        """Save metrics history to JSON"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if len(self.metrics_history['train_loss']) > 0:
            axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss')
            if len(self.metrics_history['val_loss']) > 0:
                axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Precision
        if len(self.metrics_history['precision']) > 0:
            axes[0, 1].plot(self.metrics_history['precision'], color='blue')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision Over Time')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Recall
        if len(self.metrics_history['recall']) > 0:
            axes[1, 0].plot(self.metrics_history['recall'], color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].set_title('Recall Over Time')
            axes[1, 0].grid(True, alpha=0.3)
        
        # F1-Score
        if len(self.metrics_history['f1']) > 0:
            axes[1, 1].plot(self.metrics_history['f1'], color='red')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].set_title('F1-Score Over Time')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ==================== Early Stopping ====================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=15, delta=0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
    
    def load_best_model(self, model):
        """Load the best model state"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ==================== Trainer ====================

class MetaGAVETrainer:
    """Complete trainer for MetaGAVE model"""
    
    def __init__(self, model, config, device='cuda', log_dir='./logs'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = Logger(log_dir)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience']
        )
        
        self.best_f1 = 0
        self.best_model_path = None
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc='Training')
        for batch_x, batch_mask in progress_bar:
            batch_x = batch_x.to(self.device)
            batch_mask = batch_mask.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x, batch_mask)
            
            # Compute losses
            vae_loss = self.model.vae.loss_function(
                outputs['recon'], batch_x, outputs['mu'], outputs['logvar']
            )
            pred_loss = nn.MSELoss()(outputs['pred'], batch_x)
            impute_loss = nn.MSELoss()(
                outputs['imputed'] * batch_mask,
                batch_x * batch_mask
            )
            
            loss = vae_loss + pred_loss + 0.5 * impute_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['gradient_clip']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, test_labels=None):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_scores = []
        
        with torch.no_grad():
            for batch_x, batch_mask in dataloader:
                batch_x = batch_x.to(self.device)
                batch_mask = batch_mask.to(self.device)
                
                outputs = self.model(batch_x, batch_mask)
                
                # Compute loss
                vae_loss = self.model.vae.loss_function(
                    outputs['recon'], batch_x, outputs['mu'], outputs['logvar']
                )
                pred_loss = nn.MSELoss()(outputs['pred'], batch_x)
                loss = vae_loss + pred_loss
                
                total_loss += loss.item()
                
                # Compute anomaly scores
                scores = self.model.compute_anomaly_score(batch_x, outputs)
                all_scores.append(scores.cpu())
        
        avg_loss = total_loss / len(dataloader)
        all_scores = torch.cat(all_scores, dim=0)
        
        # Compute metrics if labels available
        metrics = {'val_loss': avg_loss}
        if test_labels is not None:
            threshold = self.model.compute_pot_threshold(all_scores)
            predictions = (all_scores > threshold).float().flatten()
            labels = torch.FloatTensor(test_labels[:len(predictions)])
            
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels.numpy(), predictions.numpy(), average='binary', zero_division=0
            )
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        return metrics
    
    def fit(self, train_loader, val_loader=None, val_labels=None):
        """Complete training loop"""
        self.logger.log("=" * 60)
        self.logger.log("Starting MetaGAVE Training")
        self.logger.log("=" * 60)
        self.logger.log(f"Device: {self.device}")
        self.logger.log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.log(f"Config: {self.config.config}")
        
        for epoch in range(self.config['epochs']):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            metrics = {'train_loss': train_loss}
            if val_loader is not None:
                val_metrics = self.validate(val_loader, val_labels)
                metrics.update(val_metrics)
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['val_loss'])
                
                # Early stopping
                self.early_stopping(val_metrics['val_loss'], self.model)
                
                # Save best model
                if 'f1' in val_metrics and val_metrics['f1'] > self.best_f1:
                    self.best_f1 = val_metrics['f1']
                    self.save_checkpoint(epoch, val_metrics)
            
            # Log metrics
            self.logger.log_metrics(epoch + 1, metrics)
            
            # Early stopping check
            if self.early_stopping.early_stop:
                self.logger.log(f"Early stopping triggered at epoch {epoch + 1}")
                self.early_stopping.load_best_model(self.model)
                break
        
        self.logger.log("=" * 60)
        self.logger.log("Training completed!")
        self.logger.log(f"Best F1-Score: {self.best_f1:.4f}")
        self.logger.save_metrics()
        
        return self.model
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.logger.log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"metagave_epoch{epoch}_f1{metrics.get('f1', 0):.4f}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.config
        }, checkpoint_path)
        
        self.best_model_path = checkpoint_path
        self.logger.log(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.log(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint


# ==================== Evaluator ====================

class MetaGAVEEvaluator:
    """Complete evaluation suite for MetaGAVE"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def evaluate(self, test_loader, test_labels, threshold=None):
        """Full evaluation with multiple metrics"""
        self.model.eval()
        
        all_scores = []
        all_anomalies = []
        
        # Get predictions
        with torch.no_grad():
            for batch_x, batch_mask in tqdm(test_loader, desc='Evaluating'):
                batch_x = batch_x.to(self.device)
                batch_mask = batch_mask.to(self.device)
                
                anomalies, scores, threshold = self.model.detect_anomalies(
                    batch_x, batch_mask, threshold
                )
                
                all_scores.append(scores.cpu())
                all_anomalies.append(anomalies.cpu())
        
        # Concatenate results
        all_scores = torch.cat(all_scores, dim=0)
        all_anomalies = torch.cat(all_anomalies, dim=0)
        
        # Flatten for evaluation
        y_pred = all_anomalies.flatten()
        y_true = torch.FloatTensor(test_labels[:len(y_pred)])
        
        # Compute metrics
        results = self._compute_all_metrics(y_true, y_pred, all_scores.flatten())
        results['threshold'] = threshold.item() if isinstance(threshold, torch.Tensor) else threshold
        
        return results
    
    def _compute_all_metrics(self, y_true, y_pred, scores):
        """Compute comprehensive metrics"""
        from sklearn.metrics import (
            precision_recall_fscore_support,
            roc_auc_score,
            average_precision_score,
            confusion_matrix
        )
        
        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.numpy(), y_pred.numpy(), average='binary', zero_division=0
        )
        
        # ROC AUC and PR AUC
        try:
            roc_auc = roc_auc_score(y_true.numpy(), scores.numpy())
            pr_auc = average_precision_score(y_true.numpy(), scores.numpy())
        except:
            roc_auc = 0.0
            pr_auc = 0.0
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true.numpy(), y_pred.numpy()).ravel()
        
        # Point-adjusted metrics (from paper)
        y_pred_adjusted = self._point_adjust(y_pred, y_true)
        precision_adj, recall_adj, f1_adj, _ = precision_recall_fscore_support(
            y_true.numpy(), y_pred_adjusted.numpy(), average='binary', zero_division=0
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_adjusted': precision_adj,
            'recall_adjusted': recall_adj,
            'f1_adjusted': f1_adj,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    def _point_adjust(self, y_pred, y_true):
        """Point-adjust evaluation (from paper)"""
        adjusted_pred = y_pred.clone()
        
        # Find anomaly segments
        segments = []
        in_segment = False
        start = 0
        
        for i in range(len(y_true)):
            if y_true[i] == 1 and not in_segment:
                start = i
                in_segment = True
            elif y_true[i] == 0 and in_segment:
                segments.append((start, i))
                in_segment = False
        
        if in_segment:
            segments.append((start, len(y_true)))
        
        # Adjust predictions
        for start, end in segments:
            if y_pred[start:end].sum() > 0:
                adjusted_pred[start:end] = 1
        
        return adjusted_pred
    
    def plot_results(self, test_data, anomalies, scores, labels, feature_idx=0, save_path=None):
        """Visualize detection results"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Flatten for plotting
        anomalies_flat = anomalies[:, feature_idx].numpy()
        scores_flat = scores[:, feature_idx].numpy()
        
        # Plot 1: Time series with detections
        axes[0].plot(test_data[:len(anomalies_flat), feature_idx], 
                    label='Time Series', color='blue', alpha=0.7)
        
        detected_idx = np.where(anomalies_flat > 0)[0]
        axes[0].scatter(detected_idx, test_data[detected_idx, feature_idx],
                       color='red', label='Detected', s=50, zorder=5)
        
        true_idx = np.where(labels[:len(anomalies_flat)] > 0)[0]
        if len(true_idx) > 0:
            axes[0].axvspan(true_idx[0], true_idx[-1], alpha=0.2, 
                           color='purple', label='True Anomaly')
        
        axes[0].set_title(f'Anomaly Detection Results - Feature {feature_idx}')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores
        axes[1].plot(scores_flat, color='green', label='Anomaly Score')
        threshold = np.percentile(scores_flat, 95)
        axes[1].axhline(y=threshold, color='orange', linestyle='--', label='Threshold')
        axes[1].fill_between(range(len(scores_flat)), 0, scores_flat,
                            where=(scores_flat > threshold), alpha=0.3, color='red')
        axes[1].set_title('Anomaly Scores Over Time')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Confusion heatmap
        cm = np.zeros((2, 2))
        cm[0, 0] = ((labels[:len(anomalies_flat)] == 0) & (anomalies_flat == 0)).sum()  # TN
        cm[0, 1] = ((labels[:len(anomalies_flat)] == 0) & (anomalies_flat == 1)).sum()  # FP
        cm[1, 0] = ((labels[:len(anomalies_flat)] == 1) & (anomalies_flat == 0)).sum()  # FN
        cm[1, 1] = ((labels[:len(anomalies_flat)] == 1) & (anomalies_flat == 1)).sum()  # TP
        
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', ax=axes[2],
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        axes[2].set_title('Confusion Matrix')
        axes[2].set_ylabel('True Label')
        axes[2].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ==================== Main Execution ====================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='MetaGAVE Training and Evaluation')
    parser.add_argument('--dataset', type=str, default='SMAP',
                       choices=['SMAP', 'MSL', 'SMD', 'SWAT', 'synthetic'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to dataset')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'both'],
                       help='Mode: train, eval, or both')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_plots', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MetaGAVE: Multivariate Time Series Anomaly Detection")
    print("=" * 60)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load configuration
    config = Config(args.dataset, batch_size=args.batch_size, 
                   epochs=args.epochs, learning_rate=args.lr)
    
    # Load or generate data
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == 'synthetic':
        from metagave_data_utils import SyntheticDataGenerator
        generator = SyntheticDataGenerator()
        data = generator.generate_dataset_with_anomalies(
            n_samples=2000,
            n_features=config['n_features'],
            anomaly_ratio=0.05
        )
    else:
        from metagave_data_utils import DatasetManager
        manager = DatasetManager(args.dataset, args.data_path)
        data = manager.get_data()
        print(f"Dataset info: {manager.get_info()}")
    
    # Create dataloaders
    from metagave_model import TimeSeriesDataset
    train_dataset = TimeSeriesDataset(data['train'], config['window_size'])
    test_dataset = TimeSeriesDataset(data['test'], config['window_size'])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    # Create model
    from metagave_model import MetaGAVE
    model = MetaGAVE(
        n_features=config['n_features'],
        window_size=config['window_size'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        eta=config['eta'],
        dropout=config['dropout']
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training
    if args.mode in ['train', 'both']:
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)
        
        trainer = MetaGAVETrainer(model, config, device, args.log_dir)
        model = trainer.fit(train_loader, test_loader, data['test_labels'])
        
        # Plot training curves
        if args.save_plots:
            plot_path = os.path.join(args.log_dir, 'training_curves.png')
            trainer.logger.plot_training_curves(plot_path)
        
        # Save final model
        final_model_path = os.path.join(args.log_dir, 'final_model.pt')
        torch.save(model.state_dict(), final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
    
    # Evaluation
    if args.mode in ['eval', 'both']:
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        
        if args.checkpoint:
            print(f"\nLoading checkpoint: {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        
        evaluator = MetaGAVEEvaluator(model, device)
        results = evaluator.evaluate(test_loader, data['test_labels'])
        
        print("\n" + "-" * 60)
        print("RESULTS")
        print("-" * 60)
        print(f"Precision:        {results['precision']:.4f}")
        print(f"Recall:           {results['recall']:.4f}")
        print(f"F1-Score:         {results['f1']:.4f}")
        print(f"\nPoint-Adjusted Metrics:")
        print(f"Precision (Adj):  {results['precision_adjusted']:.4f}")
        print(f"Recall (Adj):     {results['recall_adjusted']:.4f}")
        print(f"F1-Score (Adj):   {results['f1_adjusted']:.4f}")
        print(f"\nROC AUC:          {results['roc_auc']:.4f}")
        print(f"PR AUC:           {results['pr_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TP: {results['tp']}, FP: {results['fp']}")
        print(f"FN: {results['fn']}, TN: {results['tn']}")
        
        # Save results
        results_path = os.path.join(args.log_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {results_path}")
        
        # Plot results
        if args.save_plots:
            anomalies, scores, _ = evaluator.model.detect_anomalies(
                torch.FloatTensor(data['test']).unsqueeze(0).to(device),
                torch.zeros_like(torch.FloatTensor(data['test']).unsqueeze(0)).to(device)
            )
            plot_path = os.path.join(args.log_dir, 'detection_results.png')
            evaluator.plot_results(
                data['test'], anomalies[0], scores[0], 
                data['test_labels'], feature_idx=0, save_path=plot_path
            )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
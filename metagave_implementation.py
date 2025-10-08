"""
MetaGAVE Model Implementation
Main model architecture including GAT, VAE, BiAT-LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
import math

# ==================== Graph Attention Layer ====================
class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer for learning inter-variable dependencies"""
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, x, adj):
        # x: (batch, nodes, in_features)
        # adj: (nodes, nodes)
        h = self.W(x)  # (batch, nodes, out_features)
        batch_size, N = h.size()[0], h.size()[1]
        
        # Attention mechanism
        a_input = torch.cat([h.repeat(1, 1, N).view(batch_size, N*N, -1),
                            h.repeat(1, N, 1)], dim=2).view(batch_size, N, N, 2*self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # Mask attention with adjacency
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(0) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, h)
        return F.relu(h_prime)


# ==================== Missing Value Imputation Module ====================
class MissingValueImputer(nn.Module):
    """GAT-based imputation module for missing values"""
    def __init__(self, n_features, embed_dim=128, hidden_dim=128, n_heads=4, dropout=0.1):
        super(MissingValueImputer, self).__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        
        # Node embeddings for each sensor
        self.node_embeddings = nn.Parameter(torch.randn(n_features, embed_dim))
        
        # Multi-head GAT layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(n_heads)
        ])
        
        self.feature_projection = nn.Linear(n_features, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim * n_heads, n_features)
        
    def build_adjacency_matrix(self, embeddings, k=30):
        """Build adjacency matrix based on embedding similarity (Top-K)"""
        # Compute normalized dot product
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.matmul(norm_embeddings, norm_embeddings.t())
        
        # Top-K selection
        _, indices = torch.topk(similarity, k, dim=1)
        adj = torch.zeros_like(similarity)
        adj.scatter_(1, indices, 1.0)
        
        return adj
    
    def forward(self, x, mask):
        # x: (batch, window, n_features)
        # mask: (batch, window, n_features) - 1 for missing, 0 for observed
        batch_size, window_size, n_feat = x.size()
        
        # Build adjacency matrix
        adj = self.build_adjacency_matrix(self.node_embeddings)
        
        # Reshape for GAT processing
        x_reshaped = x.view(batch_size * window_size, n_feat)
        h = self.feature_projection(x_reshaped)
        h = h.view(batch_size * window_size, 1, -1).repeat(1, n_feat, 1)
        
        # Multi-head attention
        h_cat = []
        for gat in self.gat_layers:
            h_cat.append(gat(h, adj))
        h_cat = torch.cat(h_cat, dim=2)
        
        # Output projection
        imputed = self.output_projection(h_cat)
        imputed = imputed.mean(dim=1)  # Average over nodes
        imputed = imputed.view(batch_size, window_size, n_feat)
        
        # Fill missing values
        output = x * (1 - mask) + imputed * mask
        
        return output, adj


# ==================== VAE for Reconstruction ====================
class VAEEncoder(nn.Module):
    """Encoder for VAE"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder for VAE"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


class VAE(nn.Module):
    """Variational Autoencoder for reconstruction-based anomaly detection"""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=4):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # x: (batch, window, features)
        batch_size, window_size, n_features = x.size()
        x_flat = x.view(batch_size * window_size, n_features)
        
        # Encode
        mu, logvar = self.encoder(x_flat)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decoder(z)
        recon = recon.view(batch_size, window_size, n_features)
        
        return recon, mu, logvar
    
    def loss_function(self, recon, x, mu, logvar):
        batch_size, window_size, n_features = x.size()
        x_flat = x.view(batch_size * window_size, n_features)
        recon_flat = recon.view(batch_size * window_size, n_features)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_flat, x_flat, reduction='sum')
        
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kld


# ==================== BiAT-LSTM for Prediction ====================
class BiATLSTM(nn.Module):
    """Bidirectional LSTM with Attention for prediction-based anomaly detection"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super(BiATLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=4, dropout=dropout, batch_first=True
        )
        
        self.fc_out = nn.Linear(hidden_dim * 2, input_dim)
        
    def forward(self, x, value_matrix=None):
        # x: (batch, window, features)
        batch_size, seq_len, n_features = x.size()
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, window, hidden*2)
        
        # Attention mechanism
        if value_matrix is not None:
            # Use value_matrix from GAT
            attn_out, _ = self.attention(lstm_out, lstm_out, value_matrix)
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Output projection
        output = self.fc_out(attn_out)
        
        return output


# ==================== MetaGAVE Model ====================
class MetaGAVE(nn.Module):
    """
    Complete MetaGAVE model integrating:
    - Missing value imputation (GAT-based)
    - Reconstruction path (VAE)
    - Prediction path (BiAT-LSTM)
    """
    def __init__(self, n_features, window_size=30, embed_dim=128, 
                 hidden_dim=128, latent_dim=4, eta=0.3, dropout=0.1):
        super(MetaGAVE, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.eta = eta  # Fusion hyperparameter
        
        # Missing value imputation module
        self.imputer = MissingValueImputer(
            n_features, embed_dim, hidden_dim, dropout=dropout
        )
        
        # VAE for reconstruction
        self.vae = VAE(n_features, hidden_dim, latent_dim)
        
        # BiAT-LSTM for prediction
        self.biat_lstm = BiATLSTM(n_features, hidden_dim, dropout=dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input time series (batch, window, features)
            mask: Missing value mask (batch, window, features)
        Returns:
            recon: Reconstructed data from VAE
            pred: Predicted data from BiAT-LSTM
            imputed: Data after imputation
        """
        # Step 1: Impute missing values
        if mask is None:
            mask = torch.zeros_like(x)
        
        imputed, adj = self.imputer(x, mask)
        
        # Step 2: Reconstruction path (VAE)
        recon, mu, logvar = self.vae(imputed)
        
        # Step 3: Prediction path (BiAT-LSTM)
        pred = self.biat_lstm(imputed, imputed)
        
        return {
            'recon': recon,
            'pred': pred,
            'imputed': imputed,
            'mu': mu,
            'logvar': logvar,
            'adj': adj
        }
    
    def compute_anomaly_score(self, x, outputs):
        """Compute anomaly score combining reconstruction and prediction"""
        recon = outputs['recon']
        pred = outputs['pred']
        
        # Prediction error
        pred_error = torch.mean((x - pred) ** 2, dim=2)
        
        # Reconstruction probability (negative log-likelihood)
        recon_error = torch.mean((x - recon) ** 2, dim=2)
        
        # Combined anomaly score (Equation 25)
        score = (pred_error + self.eta * recon_error) / (1 + self.eta)
        
        return score
    
    def detect_anomalies(self, x, mask=None, threshold=None, q=0.01):
        """
        Detect anomalies using POT (Peaks Over Threshold)
        
        Args:
            x: Input time series
            mask: Missing value mask
            threshold: Pre-computed threshold (optional)
            q: Probability for POT threshold
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, mask)
            scores = self.compute_anomaly_score(x, outputs)
            
            if threshold is None:
                # Use POT to compute dynamic threshold
                threshold = self.compute_pot_threshold(scores, q)
            
            anomalies = (scores > threshold).float()
            
        return anomalies, scores, threshold
    
    def compute_pot_threshold(self, scores, q=0.01):
        """Compute dynamic threshold using Peaks Over Threshold (POT)"""
        scores_flat = scores.flatten().cpu().numpy()
        scores_sorted = np.sort(scores_flat)
        
        # Initial threshold (e.g., 98th percentile)
        init_threshold = np.percentile(scores_sorted, 98)
        
        # Find peaks over initial threshold
        peaks = scores_flat[scores_flat > init_threshold]
        
        if len(peaks) < 10:
            return torch.tensor(init_threshold).to(scores.device)
        
        # Simple GPD parameter estimation
        beta = np.std(peaks)
        gamma = 0.1  # Shape parameter
        
        N = len(scores_flat)
        N_thd = len(peaks)
        
        # Final threshold (Equation 23)
        final_threshold = init_threshold - (beta / gamma) * ((q * N / N_thd) ** (-gamma) - 1)
        
        return torch.tensor(final_threshold).to(scores.device)


# ==================== Data Preprocessing ====================
class TimeSeriesDataset(torch.utils.data.Dataset):
    """Dataset for sliding window time series"""
    def __init__(self, data, window_size=30, stride=1, missing_ratio=0.0):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.missing_ratio = missing_ratio
        
        # Create sliding windows
        self.windows = []
        for i in range(0, len(data) - window_size + 1, stride):
            self.windows.append(data[i:i+window_size])
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = torch.FloatTensor(self.windows[idx])
        
        # Create missing value mask
        mask = torch.zeros_like(window)
        if self.missing_ratio > 0:
            missing_indices = torch.rand_like(window) < self.missing_ratio
            mask[missing_indices] = 1.0
            window[missing_indices] = 0.0
        
        return window, mask


# ==================== Anomaly Explanation Module ====================
class AnomalyExplainer:
    """Probabilistic graph-based anomaly explanation"""
    def __init__(self, model):
        self.model = model
    
    def explain_anomaly(self, x, anomaly_idx, top_k=5):
        """
        Explain anomaly by finding root causes
        
        Args:
            x: Input time series (batch, window, features)
            anomaly_idx: Index of anomaly to explain
            top_k: Number of top contributing features
        
        Returns:
            root_causes: Indices of features that are root causes
            contributions: Contribution scores for each feature
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            
            # Get reconstruction and prediction errors per feature
            recon_error = torch.mean((x - outputs['recon']) ** 2, dim=1)  # (batch, features)
            pred_error = torch.mean((x - outputs['pred']) ** 2, dim=1)
            
            # Combined error for the anomaly
            combined_error = recon_error[anomaly_idx] + pred_error[anomaly_idx]
            
            # Find top-k features with highest errors
            top_k_values, top_k_indices = torch.topk(combined_error, top_k)
            
            # Get adjacency matrix to find dependencies
            adj = outputs['adj']
            
            # Find root causes (nodes with high out-degree, low in-degree)
            in_degree = adj.sum(dim=0)
            out_degree = adj.sum(dim=1)
            
            root_cause_score = out_degree - in_degree
            root_causes = []
            
            for idx in top_k_indices:
                if root_cause_score[idx] > 0:  # More outgoing than incoming
                    root_causes.append(idx.item())
            
        return root_causes, top_k_indices.tolist(), top_k_values.tolist()
    
    def trace_propagation_path(self, adj, root_cause, max_depth=3):
        """Trace anomaly propagation path from root cause"""
        visited = set()
        path = []
        
        def dfs(node, depth):
            if depth > max_depth or node in visited:
                return
            visited.add(node)
            path.append(node)
            
            # Find neighbors
            neighbors = torch.where(adj[node] > 0)[0]
            for neighbor in neighbors:
                dfs(neighbor.item(), depth + 1)
        
        dfs(root_cause, 0)
        return path


# ==================== Pipeline ====================
class MetaGAVEPipeline:
    """Complete pipeline for training and evaluation"""
    def __init__(self, n_features, window_size=30, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = MetaGAVE(
            n_features=n_features,
            window_size=window_size
        ).to(self.device)
        self.explainer = AnomalyExplainer(self.model)
    
    def train(self, train_data, epochs=100, batch_size=32, lr=0.001):
        """Train the model"""
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(train_data, self.model.window_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_x, batch_mask in dataloader:
                batch_x = batch_x.to(self.device)
                batch_mask = batch_mask.to(self.device)
                
                optimizer.zero_grad()
                
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
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        return self.model
    
    def evaluate(self, test_data, test_labels, threshold=None):
        """Evaluate model on test data"""
        dataset = TimeSeriesDataset(test_data, self.model.window_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False
        )
        
        all_anomalies = []
        all_scores = []
        
        self.model.eval()
        for batch_x, batch_mask in dataloader:
            batch_x = batch_x.to(self.device)
            batch_mask = batch_mask.to(self.device)
            
            anomalies, scores, threshold = self.model.detect_anomalies(
                batch_x, batch_mask, threshold
            )
            
            all_anomalies.append(anomalies.cpu())
            all_scores.append(scores.cpu())
        
        all_anomalies = torch.cat(all_anomalies, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        
        # Flatten predictions
        y_pred = all_anomalies.flatten()
        y_true = torch.FloatTensor(test_labels[:len(y_pred)])
        
        # Compute metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.numpy(), y_pred.numpy(), average='binary', zero_division=0
        )
        
        # Point-adjusted metrics
        y_pred_adjusted = self._point_adjust(y_pred, y_true)
        precision_adj, recall_adj, f1_adj, _ = precision_recall_fscore_support(
            y_true.numpy(), y_pred_adjusted.numpy(), average='binary', zero_division=0
        )
        
        return {
            'raw_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'adjusted_metrics': {
                'precision': precision_adj,
                'recall': recall_adj,
                'f1': f1_adj
            },
            'threshold': threshold,
            'scores': all_scores
        }
    
    def _point_adjust(self, y_pred, y_true):
        """Point-adjust evaluation"""
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

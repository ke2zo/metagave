import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

# ==================== Data Loaders for Benchmark Datasets ====================

class SMAPDataLoader:
    """Loader for SMAP (Soil Moisture Active Passive) dataset"""
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load(self):
        """Load SMAP dataset"""
        train_data = np.load(os.path.join(self.data_path, 'SMAP_train.npy'))
        test_data = np.load(os.path.join(self.data_path, 'SMAP_test.npy'))
        test_labels = np.load(os.path.join(self.data_path, 'SMAP_test_label.npy'))
        
        return {
            'train': train_data,
            'test': test_data,
            'test_labels': test_labels,
            'name': 'SMAP'
        }


class MSLDataLoader:
    """Loader for MSL (Mars Science Laboratory) dataset"""
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load(self):
        """Load MSL dataset"""
        train_data = np.load(os.path.join(self.data_path, 'MSL_train.npy'))
        test_data = np.load(os.path.join(self.data_path, 'MSL_test.npy'))
        test_labels = np.load(os.path.join(self.data_path, 'MSL_test_label.npy'))
        
        return {
            'train': train_data,
            'test': test_data,
            'test_labels': test_labels,
            'name': 'MSL'
        }


class SMDDataLoader:
    """Loader for SMD (Server Machine Dataset)"""
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load(self):
        """Load SMD dataset"""
        train_data = np.load(os.path.join(self.data_path, 'SMD_train.npy'))
        test_data = np.load(os.path.join(self.data_path, 'SMD_test.npy'))
        test_labels = np.load(os.path.join(self.data_path, 'SMD_test_label.npy'))
        
        return {
            'train': train_data,
            'test': test_data,
            'test_labels': test_labels,
            'name': 'SMD'
        }


class SWATDataLoader:
    """Loader for SWaT (Secure Water Treatment) dataset"""
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load(self):
        """Load SWaT dataset"""
        # SWaT often comes in CSV format
        train_df = pd.read_csv(os.path.join(self.data_path, 'SWaT_train.csv'))
        test_df = pd.read_csv(os.path.join(self.data_path, 'SWaT_test.csv'))
        
        # Assuming last column is label
        train_data = train_df.iloc[:, :-1].values
        test_data = test_df.iloc[:, :-1].values
        test_labels = test_df.iloc[:, -1].values
        
        return {
            'train': train_data,
            'test': test_data,
            'test_labels': test_labels,
            'name': 'SWAT'
        }


# ==================== Data Preprocessing ====================

class DataPreprocessor:
    """Preprocessing utilities for time series data"""
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type
        self.scaler = None
        
    def fit(self, data):
        """Fit scaler on training data"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        self.scaler.fit(data)
        return self
    
    def transform(self, data):
        """Transform data using fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(data)
    
    def fit_transform(self, data):
        """Fit and transform in one step"""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data):
        """Inverse transform to original scale"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted.")
        return self.scaler.inverse_transform(data)
    
    @staticmethod
    def create_missing_mask(data, missing_ratio=0.1, pattern='random'):
        """
        Create missing value mask
        
        Args:
            data: Input data array
            missing_ratio: Ratio of missing values
            pattern: 'random', 'block', or 'column'
        """
        mask = np.zeros_like(data)
        
        if pattern == 'random':
            # Random missing values
            missing_idx = np.random.rand(*data.shape) < missing_ratio
            mask[missing_idx] = 1
            
        elif pattern == 'block':
            # Block missing values (simulate sensor failures)
            n_samples, n_features = data.shape
            block_size = int(n_samples * missing_ratio)
            
            for feat in range(n_features):
                if np.random.rand() < 0.3:  # 30% chance per feature
                    start = np.random.randint(0, n_samples - block_size)
                    mask[start:start+block_size, feat] = 1
                    
        elif pattern == 'column':
            # Entire columns missing (some sensors completely fail)
            n_features = data.shape[1]
            n_missing = int(n_features * missing_ratio)
            missing_features = np.random.choice(n_features, n_missing, replace=False)
            mask[:, missing_features] = 1
        
        return mask
    
    @staticmethod
    def inject_anomalies(data, anomaly_ratio=0.05, anomaly_type='spike'):
        """
        Inject synthetic anomalies into data
        
        Args:
            data: Input data array
            anomaly_ratio: Ratio of anomalous points
            anomaly_type: 'spike', 'drift', or 'pattern'
        """
        anomaly_data = data.copy()
        labels = np.zeros(len(data))
        n_samples = len(data)
        n_anomalies = int(n_samples * anomaly_ratio)
        
        if anomaly_type == 'spike':
            # Point anomalies (sudden spikes)
            anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
            anomaly_data[anomaly_indices] += np.random.randn(n_anomalies, data.shape[1]) * 5
            labels[anomaly_indices] = 1
            
        elif anomaly_type == 'drift':
            # Contextual anomalies (gradual drift)
            start = np.random.randint(0, n_samples - n_anomalies)
            end = start + n_anomalies
            drift = np.linspace(0, 5, n_anomalies)
            anomaly_data[start:end] += drift[:, np.newaxis]
            labels[start:end] = 1
            
        elif anomaly_type == 'pattern':
            # Collective anomalies (unusual patterns)
            start = np.random.randint(0, n_samples - n_anomalies)
            end = start + n_anomalies
            anomaly_data[start:end] = -anomaly_data[start:end]
            labels[start:end] = 1
        
        return anomaly_data, labels


# ==================== Data Augmentation ====================

class TimeSeriesAugmentor:
    """Data augmentation for time series"""
    
    @staticmethod
    def jitter(data, sigma=0.03):
        """Add random noise"""
        return data + np.random.normal(0, sigma, data.shape)
    
    @staticmethod
    def scaling(data, sigma=0.1):
        """Scale by random factor"""
        factor = np.random.normal(1, sigma, (1, data.shape[1]))
        return data * factor
    
    @staticmethod
    def time_warp(data, sigma=0.2):
        """Random time warping"""
        from scipy.interpolate import interp1d
        
        orig_steps = np.arange(len(data))
        random_warps = np.random.normal(1, sigma, len(data))
        warped_steps = np.cumsum(random_warps)
        warped_steps = (warped_steps - warped_steps[0]) / (warped_steps[-1] - warped_steps[0]) * (len(data) - 1)
        
        warped_data = np.zeros_like(data)
        for dim in range(data.shape[1]):
            f = interp1d(orig_steps, data[:, dim], kind='linear')
            warped_data[:, dim] = f(warped_steps)
        
        return warped_data
    
    @staticmethod
    def window_slice(data, slice_ratio=0.9):
        """Random window slicing"""
        n_samples = len(data)
        slice_len = int(n_samples * slice_ratio)
        start = np.random.randint(0, n_samples - slice_len)
        return data[start:start+slice_len]
    
    @staticmethod
    def augment_batch(data, methods=['jitter', 'scaling']):
        """Apply multiple augmentation methods"""
        augmented = data.copy()
        
        for method in methods:
            if method == 'jitter':
                augmented = TimeSeriesAugmentor.jitter(augmented)
            elif method == 'scaling':
                augmented = TimeSeriesAugmentor.scaling(augmented)
            elif method == 'time_warp':
                augmented = TimeSeriesAugmentor.time_warp(augmented)
        
        return augmented


# ==================== Dataset Manager ====================

class DatasetManager:
    """Manage loading and preprocessing of benchmark datasets"""
    
    DATASETS = {
        'SMAP': SMAPDataLoader,
        'MSL': MSLDataLoader,
        'SMD': SMDDataLoader,
        'SWAT': SWATDataLoader
    }
    
    def __init__(self, dataset_name, data_path, normalize=True, scaler_type='standard'):
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.normalize = normalize
        self.preprocessor = DataPreprocessor(scaler_type) if normalize else None
        
        # Load data
        loader = self.DATASETS[dataset_name](data_path)
        self.data = loader.load()
        
        # Preprocess
        if self.normalize:
            self.data['train'] = self.preprocessor.fit_transform(self.data['train'])
            self.data['test'] = self.preprocessor.transform(self.data['test'])
    
    def get_data(self):
        """Get processed data"""
        return self.data
    
    def get_info(self):
        """Get dataset information"""
        return {
            'name': self.dataset_name,
            'train_shape': self.data['train'].shape,
            'test_shape': self.data['test'].shape,
            'n_features': self.data['train'].shape[1],
            'n_train_samples': len(self.data['train']),
            'n_test_samples': len(self.data['test']),
            'anomaly_ratio': self.data['test_labels'].sum() / len(self.data['test_labels'])
        }
    
    def save_processed(self, save_path):
        """Save preprocessed data"""
        os.makedirs(save_path, exist_ok=True)
        
        np.save(os.path.join(save_path, f'{self.dataset_name}_train_processed.npy'), 
                self.data['train'])
        np.save(os.path.join(save_path, f'{self.dataset_name}_test_processed.npy'), 
                self.data['test'])
        np.save(os.path.join(save_path, f'{self.dataset_name}_test_labels.npy'), 
                self.data['test_labels'])
        
        if self.preprocessor is not None:
            with open(os.path.join(save_path, f'{self.dataset_name}_scaler.pkl'), 'wb') as f:
                pickle.dump(self.preprocessor.scaler, f)
        
        print(f"Saved processed data to {save_path}")


# ==================== Synthetic Data Generator ====================

class SyntheticDataGenerator:
    """Generate synthetic time series data for testing"""
    
    @staticmethod
    def generate_normal_data(n_samples=1000, n_features=10, pattern='sine'):
        """Generate normal time series data"""
        time = np.linspace(0, 100, n_samples)
        data = np.zeros((n_samples, n_features))
        
        for i in range(n_features):
            if pattern == 'sine':
                data[:, i] = np.sin(time * 0.1 + i * 0.5) + np.random.normal(0, 0.1, n_samples)
            elif pattern == 'linear':
                data[:, i] = time * 0.01 + i + np.random.normal(0, 0.1, n_samples)
            elif pattern == 'random_walk':
                data[:, i] = np.cumsum(np.random.randn(n_samples)) * 0.1
            elif pattern == 'mixed':
                if i % 3 == 0:
                    data[:, i] = np.sin(time * 0.1 + i) + np.random.normal(0, 0.1, n_samples)
                elif i % 3 == 1:
                    data[:, i] = time * 0.01 + np.random.normal(0, 0.1, n_samples)
                else:
                    data[:, i] = np.cumsum(np.random.randn(n_samples)) * 0.1
        
        return data
    
    @staticmethod
    def generate_dataset_with_anomalies(n_samples=1000, n_features=10, 
                                       anomaly_ratio=0.05, pattern='sine'):
        """Generate complete dataset with anomalies"""
        # Generate normal data
        data = SyntheticDataGenerator.generate_normal_data(n_samples, n_features, pattern)
        
        # Inject anomalies
        preprocessor = DataPreprocessor()
        data_with_anomalies, labels = preprocessor.inject_anomalies(
            data, anomaly_ratio, anomaly_type='spike'
        )
        
        # Split train/test
        train_size = int(n_samples * 0.7)
        train_data = data[:train_size]  # No anomalies in training
        test_data = data_with_anomalies[train_size:]
        test_labels = labels[train_size:]
        
        return {
            'train': train_data,
            'test': test_data,
            'test_labels': test_labels,
            'name': 'Synthetic'
        }


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("MetaGAVE Data Utilities")
    print("=" * 60)
    
    # Example 1: Generate synthetic data
    print("\n1. Generating synthetic dataset...")
    generator = SyntheticDataGenerator()
    dataset = generator.generate_dataset_with_anomalies(
        n_samples=1000,
        n_features=20,
        anomaly_ratio=0.05,
        pattern='mixed'
    )
    
    print(f"Train data shape: {dataset['train'].shape}")
    print(f"Test data shape: {dataset['test'].shape}")
    print(f"Anomaly ratio: {dataset['test_labels'].sum() / len(dataset['test_labels']):.2%}")
    
    # Example 2: Data preprocessing
    print("\n2. Testing data preprocessing...")
    preprocessor = DataPreprocessor(scaler_type='standard')
    train_normalized = preprocessor.fit_transform(dataset['train'])
    test_normalized = preprocessor.transform(dataset['test'])
    
    print(f"Original train mean: {dataset['train'].mean():.4f}, std: {dataset['train'].std():.4f}")
    print(f"Normalized train mean: {train_normalized.mean():.4f}, std: {train_normalized.std():.4f}")
    
    # Example 3: Create missing value mask
    print("\n3. Creating missing value masks...")
    for pattern in ['random', 'block', 'column']:
        mask = DataPreprocessor.create_missing_mask(
            dataset['test'], missing_ratio=0.1, pattern=pattern
        )
        print(f"{pattern.capitalize()} mask - Missing ratio: {mask.sum() / mask.size:.2%}")
    
    # Example 4: Data augmentation
    print("\n4. Testing data augmentation...")
    augmentor = TimeSeriesAugmentor()
    sample = dataset['train'][:100]
    
    augmented_jitter = augmentor.jitter(sample)
    augmented_scaling = augmentor.scaling(sample)
    augmented_batch = augmentor.augment_batch(sample, ['jitter', 'scaling'])
    
    print(f"Original data range: [{sample.min():.4f}, {sample.max():.4f}]")
    print(f"Jittered data range: [{augmented_jitter.min():.4f}, {augmented_jitter.max():.4f}]")
    print(f"Scaled data range: [{augmented_scaling.min():.4f}, {augmented_scaling.max():.4f}]")
    
    # Example 5: Load benchmark dataset (if available)
    print("\n5. Loading benchmark dataset (example)...")
    try:
        manager = DatasetManager('SMAP', './data/SMAP', normalize=True)
        info = manager.get_info()
        print(f"Dataset: {info['name']}")
        print(f"Features: {info['n_features']}")
        print(f"Train samples: {info['n_train_samples']}")
        print(f"Test samples: {info['n_test_samples']}")
        print(f"Anomaly ratio: {info['anomaly_ratio']:.2%}")
    except Exception as e:
        print(f"Could not load benchmark dataset: {e}")
        print("Using synthetic data instead.")
    
    print("\n" + "=" * 60)
    print("Done!")

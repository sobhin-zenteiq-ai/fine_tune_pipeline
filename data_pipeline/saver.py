import pandas as pd
import torch
import json
import os
from typing import Dict, Any
from .tokenizer_handler import TokenizedDataset
from torch.utils.data import Dataset

class TokenizedDataset(Dataset):
    def __init__(self, input_ids, attention_mask,labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels if labels is not None else input_ids.clone()

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

class DataSaver:
    """Handles saving processed data to various formats"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        output_config = config.get('output', {})
        
        self.save_cleaned = output_config.get('save_cleaned', True)
        self.save_tokenized = output_config.get('save_tokenized', True)
        self.save_stats = output_config.get('save_stats', True)
        self.output_dir = output_config.get('output_dir', 'output')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save(self, splits: Dict[str, pd.DataFrame], 
             tokenized_splits: Dict[str, TokenizedDataset],
             stats: Dict[str, Any]) -> Dict[str, str]:
        """
        Save all processed data
        
        Args:
            splits: Dictionary of DataFrames (train/validation)
            tokenized_splits: Dictionary of TokenizedDatasets
            stats: Processing statistics
            
        Returns:
            Dictionary of saved file paths
        """
        print("Starting data saving...")
        saved_files = {}
        
        # Save cleaned datasets
        if self.save_cleaned:
            cleaned_files = self._save_cleaned_data(splits)
            saved_files.update(cleaned_files)
        
        # Save tokenized datasets
        if self.save_tokenized:
            tokenized_files = self._save_tokenized_data(tokenized_splits)
            saved_files.update(tokenized_files)
        
        # Save statistics
        if self.save_stats:
            stats_file = self._save_stats(stats)
            saved_files['stats'] = stats_file
        
        print(f"Saved {len(saved_files)} files to {self.output_dir}")
        return saved_files
    
    def _save_cleaned_data(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Save cleaned datasets as JSON and CSV"""
        saved_files = {}
        
        for split_name, df in splits.items():
            # Save as JSON
            json_path = os.path.join(self.output_dir, f"{split_name}_cleaned.json")
            df.to_json(json_path, orient='records', indent=2)
            saved_files[f'{split_name}_json'] = json_path
            
            # Save as CSV
            csv_path = os.path.join(self.output_dir, f"{split_name}_cleaned.csv")
            df.to_csv(csv_path, index=False)
            saved_files[f'{split_name}_csv'] = csv_path
            
            print(f"Saved {split_name} dataset: {len(df)} examples")
        
        return saved_files
    
    def _save_tokenized_data(self, tokenized_splits: Dict[str, TokenizedDataset]) -> Dict[str, str]:
        """Save tokenized datasets as PyTorch tensors"""
        saved_files = {}
        
        for split_name, dataset in tokenized_splits.items():
            # Save tokenized data
            pt_path = os.path.join(self.output_dir, f"{split_name}_tokenized.pt")
            
            # Create a dictionary with all the tokenized data
            input_ids = dataset.input_ids
            attention_mask = dataset.attention_mask
            labels = dataset.labels
            
            tokenized_dataset_obj = TokenizedDataset(input_ids, attention_mask,labels)

            torch.save(tokenized_dataset_obj, pt_path)
            saved_files[f'{split_name}_tokenized'] = pt_path
            
            print(f"Saved {split_name} tokenized data: {len(dataset)} examples")
        
        return saved_files
    
    def _save_stats(self, stats: Dict[str, Any]) -> str:
        """Save processing statistics as JSON"""
        stats_path = os.path.join(self.output_dir, "pipeline_stats.json")
        
        # Convert numpy types to Python types for JSON serialization
        clean_stats = self._clean_stats_for_json(stats)
        
        with open(stats_path, 'w') as f:
            json.dump(clean_stats, f, indent=2)
        
        print(f"Saved pipeline statistics")
        return stats_path
    
    def _clean_stats_for_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Clean statistics dictionary for JSON serialization"""
        clean_stats = {}
        
        for key, value in stats.items():
            if hasattr(value, 'item'):  # numpy scalar
                clean_stats[key] = value.item()
            elif isinstance(value, (int, float, str, bool, list, dict)):
                clean_stats[key] = value
            else:
                clean_stats[key] = str(value)
        
        return clean_stats
    
    def create_summary_report(self, stats: Dict[str, Any]) -> str:
        """Create a human-readable summary report"""
        report_path = os.path.join(self.output_dir, "pipeline_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("Data Pipeline Processing Report\n")
            f.write("=" * 40 + "\n\n")
            
            # Dataset info
            if 'total_size' in stats:
                f.write(f"Total Examples: {stats['total_size']}\n")
            if 'train_size' in stats:
                f.write(f"Training Examples: {stats['train_size']}\n")
            if 'validation_size' in stats:
                f.write(f"Validation Examples: {stats['validation_size']}\n")
            
            f.write("\n")
            
            # Processing stats
            if 'removed_count' in stats:
                f.write(f"Examples Removed: {stats['removed_count']}\n")
            if 'removal_percentage' in stats:
                f.write(f"Removal Percentage: {stats['removal_percentage']:.2f}%\n")
            
            f.write("\n")
            
            # Tokenization stats
            if 'train_avg_tokens' in stats:
                f.write(f"Average Tokens (Train): {stats['train_avg_tokens']:.1f}\n")
            if 'validation_avg_tokens' in stats:
                f.write(f"Average Tokens (Validation): {stats['validation_avg_tokens']:.1f}\n")
        
        print("Created summary report")
        return report_path
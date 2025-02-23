import os
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict
import torch
from torch.utils.data import Dataset

class NIHFindingLabels(Dataset):
    """
    Dataset class for NIH CXR-14 Original Finding Labels.
    """
    def __init__(
            self,
            root_dir: Union[str, Path],
            transform: Optional[callable] = None
    ):
        self.root = Path(root_dir)
        self.labels_path = self.root / 'Data_Entry_2017.csv'
        
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        
        self.transform = transform
        self.df = pd.read_csv(self.labels_path)
        self.label_counts = {}
        self.df = self._create_binary_label_columns()
        
    def _create_binary_label_columns(self) -> pd.DataFrame:
        """Create binary label columns for each finding using vectorized operations"""
        all_labels = set()
        for findings in self.df['Finding Labels'].str.split('|'):
            all_labels.update(findings)

        counts = {}
        for label in all_labels:
            self.df[label] = self.df['Finding Labels'].str.contains(label).astype(int)
            counts[label] = int(self.df[label].sum())
    
        self.label_counts = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
        return self.df
    
    def create_top_k_dataset(self, k: int):
        """Create a new dataset instance with only the top k findings"""
        top_k_findings = list(self.label_counts.keys())[:k]
        new_instance = NIHFindingLabels(
            root_dir=self.root,
            transform=self.transform
        )
        top_k_filter = self.df[top_k_findings].sum(axis=1) > 0
        new_instance.df = self.df[top_k_filter].copy()
        counts = {label: int(new_instance.df[label].sum()) for label in top_k_findings}
        new_instance.label_counts = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
        return new_instance

    def balance_labels(self, limit: int):
        """Balance the dataset by limiting the number of samples per label"""
        new_instance = NIHFindingLabels(
            root_dir=self.root,
            transform=self.transform
        )
        balanced_df = self.df.copy()
        for label in self.label_counts:
            label_count = balanced_df[label].sum()
            if label_count > limit:
                label_rows = balanced_df[balanced_df[label] == 1]
                other_labels = [l for l in self.label_counts if l != label]
                impact_scores = label_rows[other_labels].sum(axis=1)
                to_keep = impact_scores.nsmallest(limit).index
                to_remove = set(label_rows.index) - set(to_keep)
                balanced_df = balanced_df.drop(to_remove)
        new_instance.df = balanced_df
        counts = {label: int(balanced_df[label].sum()) for label in self.label_counts}
        new_instance.label_counts = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
        return new_instance

    def merge_labels(self, labels_to_merge: List[str], new_label_name: str):
        """Merge two or more labels into a single new label"""
        for label in labels_to_merge:
            if label not in self.label_counts:
                raise ValueError(f"Label '{label}' not found in dataset")

        new_instance = NIHFindingLabels(
            root_dir=self.root,
            transform=self.transform
        )
        new_instance.df = self.df.copy()
        new_instance.df[new_label_name] = new_instance.df[labels_to_merge].max(axis=1)
        new_instance.df = new_instance.df.drop(columns=labels_to_merge)
        new_counts = {k: v for k, v in self.label_counts.items() if k not in labels_to_merge}
        new_counts[new_label_name] = int(new_instance.df[new_label_name].sum())
        new_instance.label_counts = dict(sorted(new_counts.items(), key=lambda x: (-x[1], x[0])))
        return new_instance

    def save(self, save_path: Union[str, Path]):
        """Save the processed DataFrame with only Image Index and label columns"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Get essential columns
        essential_cols = ['Image Index']
        label_cols = list(self.label_counts.keys())
        
        # Create a filtered DataFrame with only essential columns and label columns
        filtered_df = self.df[essential_cols + label_cols].copy()
        
        # Save filtered DataFrame
        df_path = save_path / 'processed_findings_label_data.csv'
        filtered_df.to_csv(df_path, index=False)

    @classmethod
    def load_from_processed(cls, 
                           processed_dir: Union[str, Path],
                           transform: Optional[callable] = None):
        """Create a dataset instance from preprocessed data"""
        processed_dir = Path(processed_dir)
        df_path = processed_dir / 'processed_findings_label_data.csv'

        if not df_path.exists():
            raise FileNotFoundError(f"Processed data file not found in {processed_dir}")

        instance = cls.__new__(cls)
        instance.root = processed_dir
        instance.transform = transform
        instance.df = pd.read_csv(df_path)
        
        # Reconstruct label_counts from the columns
        label_cols = [col for col in instance.df.columns if col != 'Image Index']
        counts = {label: int(instance.df[label].sum()) for label in label_cols}
        instance.label_counts = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
        
        return instance

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: Union[int, str]) -> Tuple[torch.Tensor, str, List[str]]:
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.df):
                raise IndexError("Index out of range")
            row = self.df.iloc[idx]
        elif isinstance(idx, str):
            row = self.df[self.df['Image Index'] == idx]
            if len(row) == 0:
                raise KeyError(f"Image ID '{idx}' not found in dataset")
            row = row.iloc[0]
        else:
            raise ValueError("Index must be an integer or a string")
        
        labels = torch.tensor(row[self.label_counts.keys()].values.astype(float))
        
        if self.transform:
            labels = self.transform(labels)
        
        return labels, row['Image Index'], list(self.label_counts.keys())
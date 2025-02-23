import os
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict

import torch
from torch.utils.data import Dataset
import copy
import json


class NIHFindingLabels(Dataset):
    """
    Dataset class for NIH CXR-14 Original Finding Labels.
    
    The dataset reads labels from a CSV file containing original annotations
    for 14 findings.
    
    Args:
        root_dir (str): Root directory containing the Data_Entry_2017.csv file
        transform (Optional[callable]): Optional transform to be applied to labels
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
        
        # Read and process the data
        self.df = pd.read_csv(self.labels_path)
        self.label_counts = {}  # Store counts of each label
        self.df = self._create_binary_label_columns()
        
    def _create_binary_label_columns(self) -> pd.DataFrame:
        """
        Create binary label columns for each finding using vectorized operations

        Returns:
            pd.DataFrame: DataFrame with binary columns for each finding
        """
        # Get all unique labels first
        all_labels = set()
        for findings in self.df['Finding Labels'].str.split('|'):
            all_labels.update(findings)

        # Create binary columns for each label using vectorized operations
        # Temporary dict to hold counts
        counts = {}
        for label in all_labels:
            self.df[label] = self.df['Finding Labels'].str.contains(label).astype(int)
            counts[label] = int(self.df[label].sum())  # Convert to regular integer
    
        # Sort dictionary by values in descending order and store as label_counts
        self.label_counts = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    
        return self.df
    
    def create_top_k_dataset(self, k: int):
        """
        Create a new dataset instance with only the top k findings
        """
        # Get top k findings (already sorted by count)
        top_k_findings = list(self.label_counts.keys())[:k]

        # Create new instance
        new_instance = NIHFindingLabels(
            root_dir=self.root,
            transform=self.transform
        )

        # Filter DataFrame
        top_k_filter = self.df[top_k_findings].sum(axis=1) > 0
        new_instance.df = self.df[top_k_filter].copy()

        # Update label counts for new instance (maintain sorting)
        counts = {label: int(new_instance.df[label].sum()) for label in top_k_findings}
        new_instance.label_counts = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))

        return new_instance

    def balance_labels(self, limit: int):
        """
        Balance the dataset by limiting the number of samples per label
        """
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

        # Update label counts and maintain sorting
        counts = {label: int(balanced_df[label].sum()) for label in self.label_counts}
        new_instance.label_counts = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))

        return new_instance
    

    def merge_labels(self, labels_to_merge: List[str], new_label_name: str):
        """
        Merge two or more labels into a single new label.

        Args:
            labels_to_merge (List[str]): List of label names to merge
            new_label_name (str): Name of the new merged label

        Returns:
            NIHFindingLabels: New dataset instance with merged labels
        """
        # Validate input labels
        for label in labels_to_merge:
            if label not in self.label_counts:
                raise ValueError(f"Label '{label}' not found in dataset")

        # Create new instance
        new_instance = NIHFindingLabels(
            root_dir=self.root,
            transform=self.transform
        )

        # Copy DataFrame
        new_instance.df = self.df.copy()

        # Create new merged label column
        new_instance.df[new_label_name] = new_instance.df[labels_to_merge].max(axis=1)

        # Remove old label columns
        new_instance.df = new_instance.df.drop(columns=labels_to_merge)

        # Update label counts
        new_counts = {k: v for k, v in self.label_counts.items() if k not in labels_to_merge}
        new_counts[new_label_name] = int(new_instance.df[new_label_name].sum())

        # Sort and store new label counts
        new_instance.label_counts = dict(sorted(new_counts.items(), key=lambda x: (-x[1], x[0])))

        return new_instance
    def save(self, save_path: Union[str, Path]):
        """
        Save the processed DataFrame and label counts to disk.

        Args:
            save_path (Union[str, Path]): Directory path to save the processed data
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save DataFrame
        df_path = save_path / 'processed_findings_label_data.csv'
        self.df.to_csv(df_path, index=False)

        # Save label counts
        counts_path = save_path / 'label_counts.json'
        with open(counts_path, 'w') as f:
            json.dump(self.label_counts, f)

    @classmethod
    def load_from_processed(cls, 
                           processed_dir: Union[str, Path],
                           transform: Optional[callable] = None):
        """
        Create a dataset instance from preprocessed data.

        Args:
            processed_dir (Union[str, Path]): Directory containing processed data files
            transform (Optional[callable]): Optional transform to be applied to labels

        Returns:
            NIHFindingLabels: Dataset instance with loaded preprocessed data
        """
        processed_dir = Path(processed_dir)
        df_path = processed_dir / 'processed_findings_label_data.csv'
        counts_path = processed_dir / 'label_counts.json'

        if not df_path.exists() or not counts_path.exists():
            raise FileNotFoundError(
                f"Processed data files not found in {processed_dir}"
            )

        # Create instance with minimal initialization
        instance = cls.__new__(cls)
        instance.root = processed_dir
        instance.transform = transform

        # Load preprocessed data
        instance.df = pd.read_csv(df_path)
        with open(counts_path, 'r') as f:
            instance.label_counts = json.load(f)

        return instance


    def __len__(self) -> int:
        """Returns the total number of labeled images."""
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
            row = row.iloc[0]  # Get first matching row
        else:
            raise ValueError("Index must be an integer or a string")
        
        labels = torch.tensor(row[self.label_counts.keys()].values.astype(float))
        
        if self.transform:
            labels = self.transform(labels)
        
        return labels, row['Image Index'], list(self.label_counts.keys())
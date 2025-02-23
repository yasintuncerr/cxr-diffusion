import os
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict
import torch
from torch.utils.data import Dataset

class NIHFourFindingsDataset(Dataset):
    """
    Dataset class for NIH CXR-14 Four Findings expert labels.
    
    The dataset reads labels from a CSV file containing expert annotations
    for four findings: Fracture, Pneumothorax, Airspace opacity, and Nodule/mass.
    Additionally creates a "No Findings" label when all findings are negative.
    
    Args:
        root_dir (str): Root directory containing the google-labels directory
        aggregation_method (str): How to aggregate multiple reader labels ('majority', 'any', 'all')
        split (Optional[str]): Which set to use ('train', 'test', None for all)
        transform (Optional[callable]): Optional transform to be applied to labels
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        aggregation_method: str = 'majority',
        split: Optional[str] = None,
        transform: Optional[callable] = None
    ):
        self.root = Path(root_dir)
        self.labels_path = self.root / 'google-labels' / 'four_findings_expert_labels_individual_readers.csv'
        
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
            
        self.aggregation_method = aggregation_method
        self.transform = transform
        self.findings = ['Fracture', 'Pneumothorax', 'Airspace opacity', 'Nodule/mass', 'No Findings']
        
        # Read and process the labels
        self._load_and_process_labels(split)

    def _load_and_process_labels(self, split: Optional[str]) -> None:
        """
        Load and process the labels CSV file.
        
        Args:
            split: Which dataset split to use ('train', 'test', None for all)
        """
        # Read CSV
        df = pd.read_csv(self.labels_path)
        
        # Filter by split if specified
        if split is not None:
            df = df[df['Set ID'] == split.lower()]
        
        # Convert YES/NO to 1/0
        original_findings = self.findings[:-1]  # Exclude 'No Findings'
        for finding in original_findings:
            df[finding] = (df[finding] == 'YES').astype(int)
        
        # Aggregate labels from multiple readers
        self.labels_dict = {}
        
        for img_id, group in df.groupby('Image ID'):
            labels = []
            for finding in original_findings:
                reader_labels = group[finding].values
                
                if self.aggregation_method == 'majority':
                    # More than 50% of readers said YES
                    label = int(reader_labels.mean() > 0.5)
                elif self.aggregation_method == 'any':
                    # Any reader said YES
                    label = int(reader_labels.any())
                elif self.aggregation_method == 'all':
                    # All readers said YES
                    label = int(reader_labels.all())
                else:
                    raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
                    
                labels.append(label)
            
            # Add "No Findings" label (1 if all other findings are 0)
            no_findings = int(all(label == 0 for label in labels))
            labels.append(no_findings)
                
            self.labels_dict[img_id] = torch.tensor(labels, dtype=torch.float32)
            
        self.image_ids = list(self.labels_dict.keys())

    def __len__(self) -> int:
        """Returns the total number of labeled images."""
        return len(self.image_ids)

    def __getitem__(self, idx: Union[int, str]) -> Tuple[torch.Tensor, str]:
        """
        Retrieve labels for an image.
        
        Args:
            idx: Either an integer index or string image ID
                
        Returns:
            tuple: (labels, image_id) where labels is a tensor of binary labels 
                   for each finding including "No Findings"
        """
        try:
            if isinstance(idx, str):
                img_id = idx
                if img_id not in self.labels_dict:
                    raise KeyError(f"Image ID '{img_id}' not found in dataset")
            elif isinstance(idx, int):
                if idx < 0 or idx >= len(self.image_ids):
                    raise IndexError("Index out of range")
                img_id = self.image_ids[idx]
            else:
                raise ValueError("Index must be an integer or a string")

            labels = self.labels_dict[img_id]
            
            if self.transform:
                labels = self.transform(labels)
                
            return labels, img_id
            
        except Exception as e:
            raise RuntimeError(f"Error accessing item at {idx}: {str(e)}")
            
    def get_finding_names(self) -> List[str]:
        """Returns the list of finding names in order."""
        return self.findings
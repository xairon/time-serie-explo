"""
Registry for prepared datasets.

Allows saving and loading datasets with their preprocessing config.
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class PreparedDataset:
    """Information about a prepared dataset."""
    name: str
    path: Path
    source_file: str
    station_column: Optional[str]
    stations: List[str]
    target_column: str
    covariate_columns: List[str]
    preprocessing: Dict[str, Any]
    creation_date: str
    n_rows: int
    date_range: Tuple[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'source_file': self.source_file,
            'station_column': self.station_column,
            'stations': self.stations,
            'target_column': self.target_column,
            'covariate_columns': self.covariate_columns,
            'preprocessing': self.preprocessing,
            'creation_date': self.creation_date,
            'n_rows': self.n_rows,
            'date_range': list(self.date_range)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: Path) -> 'PreparedDataset':
        return cls(
            name=data['name'],
            path=path,
            source_file=data['source_file'],
            station_column=data.get('station_column'),
            stations=data.get('stations', []),
            target_column=data['target_column'],
            covariate_columns=data.get('covariate_columns', []),
            preprocessing=data.get('preprocessing', {}),
            creation_date=data.get('creation_date', ''),
            n_rows=data.get('n_rows', 0),
            date_range=tuple(data.get('date_range', ['', '']))
        )


class DatasetRegistry:
    """Registry for managing prepared datasets."""
    
    def __init__(self, datasets_dir: Path):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def save_dataset(
        self,
        name: str,
        df: pd.DataFrame,
        source_file: str,
        station_column: Optional[str],
        stations: List[str],
        target_column: str,
        covariate_columns: List[str],
        preprocessing_config: Dict[str, Any]
    ) -> Path:
        """
        Save a prepared dataset with its config.
        
        Args:
            name: Dataset name (will be sanitized)
            df: Prepared DataFrame (index should be datetime)
            source_file: Original data file name
            station_column: Column for station identification (if multi-station)
            stations: List of stations in the dataset
            target_column: Target variable column name
            covariate_columns: List of covariate column names
            preprocessing_config: Preprocessing settings used
        
        Returns:
            Path to saved dataset directory
        """
        # Sanitize name
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        
        # Append timestamp to allow multiple versions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = self.datasets_dir / f"{safe_name}_{timestamp}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_path = dataset_dir / "data.csv"
        df.to_csv(data_path)
        
        # Get date range
        date_range = (
            str(df.index.min()) if hasattr(df.index, 'min') else '',
            str(df.index.max()) if hasattr(df.index, 'max') else ''
        )
        
        # Save config
        config = {
            'name': name,
            'source_file': source_file,
            'station_column': station_column,
            'stations': stations,
            'target_column': target_column,
            'covariate_columns': covariate_columns,
            'preprocessing': preprocessing_config,
            'creation_date': datetime.now().isoformat(),
            'n_rows': len(df),
            'date_range': list(date_range)
        }
        
        config_path = dataset_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return dataset_dir
    
    def scan_datasets(self) -> List[PreparedDataset]:
        """
        Scan for all prepared datasets.
        
        Returns:
            List of PreparedDataset info objects
        """
        datasets = []
        
        for config_path in self.datasets_dir.rglob("config.yaml"):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                dataset = PreparedDataset.from_dict(config, config_path.parent)
                datasets.append(dataset)
            except Exception as e:
                print(f"Warning: Could not load dataset from {config_path}: {e}")
        
        # Sort by creation date (newest first)
        datasets.sort(key=lambda d: d.creation_date, reverse=True)
        
        return datasets
    
    def load_dataset(self, dataset: PreparedDataset) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a prepared dataset.
        
        Args:
            dataset: PreparedDataset info object
        
        Returns:
            Tuple of (DataFrame, config dict)
        """
        data_path = dataset.path / "data.csv"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset data not found: {data_path}")
        
        # Load data with index
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        config = {
            'source_file': dataset.source_file,
            'station_column': dataset.station_column,
            'stations': dataset.stations,
            'target_column': dataset.target_column,
            'covariate_columns': dataset.covariate_columns,
            'preprocessing': dataset.preprocessing
        }
        
        return df, config
    
    def delete_dataset(self, dataset: PreparedDataset):
        """Delete a prepared dataset."""
        import shutil
        resolved = dataset.path.resolve()
        if not str(resolved).startswith(str(self.datasets_dir.resolve())):
            raise ValueError("Refusing to delete path outside datasets directory")
        if dataset.path.exists():
            shutil.rmtree(dataset.path)


def get_dataset_registry() -> DatasetRegistry:
    """Get the dataset registry instance."""
    from dashboard.config import BASE_DIR
    datasets_dir = BASE_DIR / 'data' / 'prepared'
    return DatasetRegistry(datasets_dir)

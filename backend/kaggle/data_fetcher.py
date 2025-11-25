"""
Kaggle Data Fetcher

Responsible for downloading competition data from Kaggle, parsing problem descriptions, getting evaluation metrics, etc.
"""
import os
import re
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from backend.config import config
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CompetitionInfo:
    """
    Competition Info Class
    
    Contains all competition related information fetched from Kaggle
    """
    # Basic Info
    competition_id: str  # Competition ID (extracted from URL)
    competition_name: str  # Competition Name
    competition_url: str  # Competition URL
    
    # Problem Info
    title: str = ""
    description: str = ""
    evaluation_metric: str = ""
    problem_type: str = ""  # classification, regression, time_series, etc.
    
    # Data Info
    data_path: Optional[Path] = None
    train_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    sample_submission_file: Optional[str] = None
    
    # Data Statistics
    train_shape: Optional[tuple] = None
    test_shape: Optional[tuple] = None
    columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    
    # Extra Info
    deadline: Optional[str] = None
    reward: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)  # Store detailed info of all files
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "competition_id": self.competition_id,
            "competition_name": self.competition_name,
            "competition_url": self.competition_url,
            "title": self.title,
            "description": self.description,
            "evaluation_metric": self.evaluation_metric,
            "problem_type": self.problem_type,
            "data_path": str(self.data_path) if self.data_path else None,
            "train_files": self.train_files,
            "test_files": self.test_files,
            "sample_submission_file": self.sample_submission_file,
            "train_shape": self.train_shape,
            "test_shape": self.test_shape,
            "columns": self.columns,
            "column_types": self.column_types,
            "deadline": self.deadline,
            "reward": self.reward,
            "tags": self.tags,
            "extra_info": self.extra_info
        }
    
    def save(self, path: Path):
        """Save competition info to file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class KaggleDataFetcher:
    """
    Kaggle Data Fetcher
    
    Features:
    1. Extract competition ID from Kaggle URL
    2. Download competition data
    3. Parse competition description and rules
    4. Analyze dataset structure
    5. Identify train/test files
    """
    
    def __init__(self):
        """Initialize Kaggle API"""
        self.api = KaggleApi()
        try:
            self.api.authenticate()
            logger.info("Kaggle API authentication successful")
        except Exception as e:
            logger.error(f"Kaggle API authentication failed: {e}")
            logger.info("Please ensure ~/.kaggle/kaggle.json is configured or environment variables KAGGLE_USERNAME and KAGGLE_KEY are set")
            raise
    
    @staticmethod
    def extract_competition_id(url: str) -> str:
        """
        Extract competition ID from Kaggle URL
        
        Supported URL formats:
        - https://www.kaggle.com/competitions/store-sales-time-series-forecasting
        - https://www.kaggle.com/c/store-sales-time-series-forecasting
        - store-sales-time-series-forecasting
        
        Args:
            url: Kaggle competition URL or ID
            
        Returns:
            Competition ID
        """
        # If it is already an ID format
        if not url.startswith("http"):
            return url
        
        # Extract ID
        patterns = [
            r'kaggle\.com/competitions/([^/\?]+)',
            r'kaggle\.com/c/([^/\?]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Cannot extract competition ID from URL: {url}")
    
    def fetch_competition_info(self, competition_url: str) -> CompetitionInfo:
        """
        Fetch basic competition info
        
        Args:
            competition_url: Competition URL or ID
            
        Returns:
            CompetitionInfo object
        """
        competition_id = self.extract_competition_id(competition_url)
        logger.info(f"Fetching competition info: {competition_id}")
        
        try:
            # Get competition details
            competition = self.api.competition_view(competition_id)
            
            # Create info object
            info = CompetitionInfo(
                competition_id=competition_id,
                competition_name=competition.title or competition_id,
                competition_url=f"https://www.kaggle.com/competitions/{competition_id}",
                title=competition.title or "",
                description=competition.description or "",
                evaluation_metric=competition.evaluationMetric or "",
                deadline=str(competition.deadline) if competition.deadline else None,
                reward=competition.reward or None,
                tags=competition.tags or []
            )
            
            # Infer problem type
            info.problem_type = self._infer_problem_type(info)
            
            logger.info(f"✓ Competition info fetched successfully: {info.competition_name}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to fetch competition info: {e}")
            # Return basic info
            return CompetitionInfo(
                competition_id=competition_id,
                competition_name=competition_id,
                competition_url=f"https://www.kaggle.com/competitions/{competition_id}"
            )
    
    def download_data(
        self,
        competition_id: str,
        download_path: Optional[Path] = None,
        force: bool = False
    ) -> Path:
        """
        Download competition data
        
        Args:
            competition_id: Competition ID
            download_path: Download path (default is data/competitions/{competition_id})
            force: Whether to force re-download
            
        Returns:
            Data directory path
        """
        # Set download path
        if download_path is None:
            download_path = config.competitions_dir / competition_id
        
        download_path.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if not force and list(download_path.glob("*.csv")):
            logger.info(f"Data already exists, skipping download: {download_path}")
            return download_path
        
        logger.info(f"Starting to download competition data: {competition_id}")
        logger.info(f"Download path: {download_path}")
        
        try:
            # Download all files
            self.api.competition_download_files(
                competition_id,
                path=str(download_path),
                quiet=False
            )
            
            # Extract zip files
            self._extract_zip_files(download_path)
            
            logger.info(f"✓ Data download completed: {download_path}")
            return download_path
            
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise
    
    def _extract_zip_files(self, directory: Path):
        """Extract all zip files in the directory"""
        for zip_file in directory.glob("*.zip"):
            try:
                logger.info(f"Extracting: {zip_file.name}")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(directory)
                # Delete zip file
                zip_file.unlink()
                logger.info(f"✓ Extraction completed: {zip_file.name}")
            except Exception as e:
                logger.warning(f"Extraction failed {zip_file.name}: {e}")
    
    def analyze_data(self, data_path: Path, info: CompetitionInfo) -> CompetitionInfo:
        """
        Analyze dataset structure
        
        Args:
            data_path: Data directory path
            info: CompetitionInfo object (will be updated)
            
        Returns:
            Updated CompetitionInfo
        """
        logger.info(f"Starting to analyze dataset: {data_path}")
        
        info.data_path = data_path
        
        # Identify all CSV files
        csv_files = list(data_path.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        # Store detailed info of all files
        all_files_info = {}
        
        for csv_file in csv_files:
            filename = csv_file.name.lower()
            
            # Classify files
            if 'train' in filename:
                info.train_files.append(csv_file.name)
            elif 'test' in filename:
                info.test_files.append(csv_file.name)
            elif 'sample' in filename or 'submission' in filename:
                info.sample_submission_file = csv_file.name
            
            # Analyze structure of each CSV file
            try:
                logger.info(f"Analyzing file: {csv_file.name}")
                df = pd.read_csv(csv_file, nrows=100)  # Read first 100 rows to understand structure
                
                file_info = {
                    "filename": csv_file.name,
                    "rows_sample": len(df),
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "sample_data": df.head(3).to_dict('records')  # First 3 rows sample data
                }
                
                all_files_info[csv_file.name] = file_info
                logger.info(f"  ✓ {csv_file.name}: {len(df.columns)} columns")
                
            except Exception as e:
                logger.warning(f"  ✗ Failed to analyze {csv_file.name}: {e}")
        
        # Save all file info to extra field
        if not hasattr(info, 'extra_info'):
            info.extra_info = {}
        info.extra_info['all_files'] = all_files_info
        
        # Analyze training data (main data)
        if info.train_files:
            train_file = data_path / info.train_files[0]
            try:
                logger.info(f"Detailed analysis of training data: {train_file.name}")
                df = pd.read_csv(train_file, nrows=1000)
                
                info.train_shape = (len(df), len(df.columns))
                info.columns = df.columns.tolist()
                info.column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
                
                logger.info(f"✓ Training data shape: {info.train_shape}")
                logger.info(f"✓ Number of columns: {len(info.columns)}")
                
            except Exception as e:
                logger.warning(f"Failed to analyze training data: {e}")
        
        # Analyze test data
        if info.test_files:
            test_file = data_path / info.test_files[0]
            try:
                logger.info(f"Detailed analysis of test data: {test_file.name}")
                df = pd.read_csv(test_file, nrows=100)
                info.test_shape = (len(df), len(df.columns))
                logger.info(f"✓ Test data shape: {info.test_shape}")
            except Exception as e:
                logger.warning(f"Failed to analyze test data: {e}")
        
        logger.info("✓ Data analysis completed")
        return info
    
    def fetch_complete_info(
        self,
        competition_url: str,
        download_data: bool = True,
        force_download: bool = False
    ) -> CompetitionInfo:
        """
        Fetch complete competition info (one-stop method)
        
        Args:
            competition_url: Competition URL or ID
            download_data: Whether to download data
            force_download: Whether to force re-download
            
        Returns:
            Complete CompetitionInfo
        """
        logger.info("=" * 60)
        logger.info("Starting to fetch complete competition info")
        logger.info("=" * 60)
        
        # 1. Get basic info
        info = self.fetch_competition_info(competition_url)
        
        # 2. Download data
        if download_data:
            data_path = self.download_data(
                info.competition_id,
                force=force_download
            )
            
            # 3. Analyze data
            info = self.analyze_data(data_path, info)
        
        # 4. Save info
        if info.data_path:
            info_file = info.data_path / "competition_info.json"
            info.save(info_file)
            logger.info(f"Competition info saved: {info_file}")
        
        logger.info("=" * 60)
        logger.info("✓ Complete competition info fetched successfully")
        logger.info("=" * 60)
        
        return info
    
    def _infer_problem_type(self, info: CompetitionInfo) -> str:
        """
        Infer problem type based on description and metrics
        
        Returns:
            Problem type: classification, regression, time_series, ranking, etc.
        """
        text = (info.description + " " + info.evaluation_metric + " " + info.title).lower()
        
        # Time series
        if any(keyword in text for keyword in ['time series', 'forecasting', 'forecast', 'temporal']):
            return "time_series_forecasting"
        
        # Classification
        if any(keyword in text for keyword in ['classification', 'classify', 'class', 'accuracy', 'f1', 'auc', 'roc']):
            return "classification"
        
        # Regression
        if any(keyword in text for keyword in ['regression', 'predict', 'rmse', 'mae', 'mse', 'r2']):
            return "regression"
        
        # Ranking
        if any(keyword in text for keyword in ['ranking', 'recommend', 'retrieval']):
            return "ranking"
        
        # Clustering
        if any(keyword in text for keyword in ['clustering', 'cluster', 'segmentation']):
            return "clustering"
        
        # NLP
        if any(keyword in text for keyword in ['nlp', 'text', 'sentiment', 'language']):
            return "nlp"
        
        # Computer Vision
        if any(keyword in text for keyword in ['image', 'vision', 'detection', 'segmentation', 'object']):
            return "computer_vision"
        
        return "unknown"
    
    def get_data_summary(self, info: CompetitionInfo) -> str:
        """
        Generate data summary (for LLM)
        
        Args:
            info: Competition info
            
        Returns:
            Formatted data summary text
        """
        summary = []
        summary.append(f"# {info.title or info.competition_name}")
        summary.append("")
        summary.append("## Competition Info")
        summary.append(f"- Competition ID: {info.competition_id}")
        summary.append(f"- Problem Type: {info.problem_type}")
        summary.append(f"- Evaluation Metric: {info.evaluation_metric}")
        
        if info.description:
            summary.append("")
            summary.append("## Problem Description")
            summary.append(info.description[:500] + "..." if len(info.description) > 500 else info.description)
        
        summary.append("")
        summary.append("## Main Data Files")
        summary.append(f"- Train Files: {', '.join(info.train_files)}")
        summary.append(f"- Test Files: {', '.join(info.test_files)}")
        if info.sample_submission_file:
            summary.append(f"- Sample Submission: {info.sample_submission_file}")
        
        # Add detailed info of all files
        if info.extra_info and 'all_files' in info.extra_info:
            summary.append("")
            summary.append("## All Available Data Files Details")
            all_files = info.extra_info['all_files']
            
            for filename, file_info in all_files.items():
                summary.append(f"\n### {filename}")
                summary.append(f"- Columns: {', '.join(file_info['columns'])}")
                summary.append(f"- Data Types:")
                for col, dtype in file_info['dtypes'].items():
                    summary.append(f"  - {col}: {dtype}")
                
                # Add sample data
                if file_info.get('sample_data'):
                    summary.append(f"- Sample Data (First 3 rows):")
                    for i, row in enumerate(file_info['sample_data'][:3], 1):
                        summary.append(f"  Row {i}: {row}")
        
        if info.train_shape:
            summary.append("")
            summary.append("## Data Size")
            summary.append(f"- Train Shape: {info.train_shape}")
            if info.test_shape:
                summary.append(f"- Test Shape: {info.test_shape}")
        
        if info.columns:
            summary.append("")
            summary.append("## Train Data Column Details")
            summary.append(f"Total {len(info.columns)} columns:")
            for col in info.columns[:20]:
                col_type = info.column_types.get(col, "unknown")
                summary.append(f"  - {col}: {col_type}")
            if len(info.columns) > 20:
                summary.append(f"  ... and {len(info.columns) - 20} more columns")
        
        summary.append("")
        summary.append("## Important Notes")
        summary.append("- Please make full use of all available data files to build features")
        summary.append("- Auxiliary data files (such as stores, oil, holidays, etc.) may contain important predictive features")
        summary.append("- Use appropriate join/merge operations to combine data")
        
        return "\n".join(summary)

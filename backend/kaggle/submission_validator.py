"""
Submission File Validator

Validates if the generated submission.csv meets Kaggle requirements
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SubmissionValidator:
    """
    Submission File Validator
    
    Features:
    1. Validate submission.csv format
    2. Check column names and count
    3. Validate ID integrity
    4. Check prediction value range
    """
    
    def __init__(self, sample_submission_path: Optional[Path] = None):
        """
        Initialize Validator
        
        Args:
            sample_submission_path: Path to sample submission file
        """
        self.sample_submission_path = sample_submission_path
        self.sample_df: Optional[pd.DataFrame] = None
        
        if sample_submission_path and sample_submission_path.exists():
            try:
                self.sample_df = pd.read_csv(sample_submission_path)
                logger.info(f"Loaded sample submission file: {sample_submission_path}")
                logger.info(f"Sample shape: {self.sample_df.shape}")
                logger.info(f"Sample columns: {list(self.sample_df.columns)}")
            except Exception as e:
                logger.warning(f"Failed to load sample submission file: {e}")
    
    def validate(self, submission_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate submission file
        
        Args:
            submission_path: Path to submission file
            
        Returns:
            (is_valid, error_list)
        """
        errors = []
        
        # 1. Check if file exists
        if not submission_path.exists():
            errors.append(f"Submission file does not exist: {submission_path}")
            return False, errors
        
        try:
            # 2. Try to read file
            submission_df = pd.read_csv(submission_path)
            logger.info(f"Submission file shape: {submission_df.shape}")
            logger.info(f"Submission file columns: {list(submission_df.columns)}")
            
        except Exception as e:
            errors.append(f"Cannot read submission file: {e}")
            return False, errors
        
        # 3. If sample file exists, validate against it
        if self.sample_df is not None:
            errors.extend(self._validate_against_sample(submission_df))
        else:
            # 4. Basic validation (when no sample file)
            errors.extend(self._basic_validation(submission_df))
        
        # 5. Check for missing values
        if submission_df.isnull().any().any():
            null_cols = submission_df.columns[submission_df.isnull().any()].tolist()
            errors.append(f"Columns with missing values: {null_cols}")
        
        # 6. Check for infinite values
        numeric_cols = submission_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if (submission_df[col] == float('inf')).any() or (submission_df[col] == float('-inf')).any():
                errors.append(f"Column '{col}' contains infinite values")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("✓ Submission file validation passed")
        else:
            logger.warning(f"✗ Submission file validation failed, found {len(errors)} issues")
            for error in errors:
                logger.warning(f"  - {error}")
        
        return is_valid, errors
    
    def _validate_against_sample(self, submission_df: pd.DataFrame) -> List[str]:
        """Validate against sample file"""
        errors = []
        
        # Check column names
        expected_cols = list(self.sample_df.columns)
        actual_cols = list(submission_df.columns)
        
        if expected_cols != actual_cols:
            errors.append(
                f"Column names mismatch. Expected: {expected_cols}, Actual: {actual_cols}"
            )
        
        # Check row count
        expected_rows = len(self.sample_df)
        actual_rows = len(submission_df)
        
        if expected_rows != actual_rows:
            errors.append(
                f"Row count mismatch. Expected: {expected_rows}, Actual: {actual_rows}"
            )
        
        # Check ID column (usually first column is ID)
        if len(expected_cols) > 0 and len(actual_cols) > 0:
            id_col = expected_cols[0]
            
            if id_col in submission_df.columns and id_col in self.sample_df.columns:
                expected_ids = set(self.sample_df[id_col])
                actual_ids = set(submission_df[id_col])
                
                missing_ids = expected_ids - actual_ids
                extra_ids = actual_ids - expected_ids
                
                if missing_ids:
                    errors.append(f"Missing {len(missing_ids)} IDs")
                if extra_ids:
                    errors.append(f"Extra {len(extra_ids)} IDs")
        
        # Check data types
        for col in expected_cols:
            if col in actual_cols:
                expected_dtype = self.sample_df[col].dtype
                actual_dtype = submission_df[col].dtype
                
                # Allow int/float compatibility
                if not self._dtypes_compatible(expected_dtype, actual_dtype):
                    errors.append(
                        f"Column '{col}' data type mismatch. Expected: {expected_dtype}, Actual: {actual_dtype}"
                    )
        
        return errors
    
    def _basic_validation(self, submission_df: pd.DataFrame) -> List[str]:
        """Basic validation (when no sample file)"""
        errors = []
        
        # Check if empty
        if len(submission_df) == 0:
            errors.append("Submission file is empty")
        
        # Check column count
        if len(submission_df.columns) < 2:
            errors.append(f"Too few columns: {len(submission_df.columns)}, usually need at least ID column and prediction column")
        
        # Check if ID column exists (common ID column names)
        id_col_names = ['id', 'Id', 'ID', 'index']
        has_id_col = any(col in submission_df.columns for col in id_col_names)
        
        if not has_id_col:
            logger.warning("No explicit ID column found, please confirm if the first column is ID")
        
        return errors
    
    @staticmethod
    def _dtypes_compatible(dtype1, dtype2) -> bool:
        """Check if two data types are compatible"""
        # Numeric types are compatible
        numeric_types = ['int64', 'int32', 'float64', 'float32']
        if str(dtype1) in numeric_types and str(dtype2) in numeric_types:
            return True
        
        # Object and string types are compatible
        string_types = ['object', 'string']
        if str(dtype1) in string_types and str(dtype2) in string_types:
            return True
        
        # Exactly same
        return dtype1 == dtype2
    
    def get_submission_summary(self, submission_path: Path) -> Dict:
        """
        Get submission file summary
        
        Args:
            submission_path: Path to submission file
            
        Returns:
            Summary info dictionary
        """
        try:
            df = pd.read_csv(submission_path)
            
            summary = {
                "file_path": str(submission_path),
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "has_null": bool(df.isnull().any().any()),
                "null_counts": df.isnull().sum().to_dict(),
            }
            
            # Numeric column stats
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                summary["numeric_stats"] = {}
                for col in numeric_cols:
                    summary["numeric_stats"][col] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std())
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get submission file summary: {e}")
            return {"error": str(e)}
    
    def fix_common_issues(
        self,
        submission_path: Path,
        output_path: Optional[Path] = None
    ) -> Tuple[bool, Path]:
        """
        Try to fix common issues
        
        Args:
            submission_path: Path to submission file
            output_path: Output path (default overwrite original file)
            
        Returns:
            (is_success, output_file_path)
        """
        if output_path is None:
            output_path = submission_path
        
        try:
            df = pd.read_csv(submission_path)
            modified = False
            
            # 1. Fill missing values (with 0 or median)
            if df.isnull().any().any():
                logger.info("Fixing: Filling missing values")
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    if df[col].isnull().any():
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                
                # Object types fill with empty string
                object_cols = df.select_dtypes(include=['object']).columns
                for col in object_cols:
                    if df[col].isnull().any():
                        df[col].fillna("", inplace=True)
                
                modified = True
            
            # 2. Replace infinite values
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if (df[col] == float('inf')).any():
                    logger.info(f"Fixing: Replacing positive infinity in column '{col}'")
                    df[col].replace(float('inf'), df[col][df[col] != float('inf')].max(), inplace=True)
                    modified = True
                
                if (df[col] == float('-inf')).any():
                    logger.info(f"Fixing: Replacing negative infinity in column '{col}'")
                    df[col].replace(float('-inf'), df[col][df[col] != float('-inf')].min(), inplace=True)
                    modified = True
            
            # 3. If sample file exists, align column order
            if self.sample_df is not None:
                expected_cols = list(self.sample_df.columns)
                if list(df.columns) != expected_cols:
                    logger.info("Fixing: Aligning column order")
                    # Keep only expected columns and in expected order
                    df = df[[col for col in expected_cols if col in df.columns]]
                    modified = True
            
            # Save fixed file
            if modified:
                df.to_csv(output_path, index=False)
                logger.info(f"✓ Fixed file saved: {output_path}")
                return True, output_path
            else:
                logger.info("No fix needed")
                return True, submission_path
            
        except Exception as e:
            logger.error(f"Fix failed: {e}")
            return False, submission_path


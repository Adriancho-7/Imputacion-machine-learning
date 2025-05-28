import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Utility class for data processing and preparation for machine learning imputation.
    """
    
    def __init__(self):
        self.column_types = {}
        self.categorical_mappings = {}
        
    def analyze_data_types(self, data: pd.DataFrame) -> Dict:
        """
        Analyze data types and categorize columns.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe to analyze
            
        Returns:
        --------
        Dict containing categorized columns
        """
        analysis = {
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'text_columns': [],
            'missing_data_summary': {}
        }
        
        for column in data.columns:
            col_data = data[column]
            
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(col_data):
                analysis['datetime_columns'].append(column)
            # Check for numeric
            elif pd.api.types.is_numeric_dtype(col_data):
                analysis['numeric_columns'].append(column)
            # Check for categorical or object
            elif pd.api.types.is_object_dtype(col_data):
                # Determine if it's categorical or text based on unique values
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < 0.1 or col_data.nunique() < 20:
                    analysis['categorical_columns'].append(column)
                else:
                    analysis['text_columns'].append(column)
            else:
                analysis['categorical_columns'].append(column)
            
            # Missing data summary
            missing_count = col_data.isnull().sum()
            if missing_count > 0:
                analysis['missing_data_summary'][column] = {
                    'missing_count': missing_count,
                    'missing_percentage': (missing_count / len(data)) * 100,
                    'data_type': str(col_data.dtype)
                }
        
        return analysis
    
    def prepare_for_imputation(self, data: pd.DataFrame, target_columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare data for SVR imputation by encoding categorical variables and scaling.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        target_columns : List[str]
            Columns to be imputed
            
        Returns:
        --------
        Tuple of prepared dataframe and encoding information
        """
        prepared_data = data.copy()
        encoding_info = {
            'categorical_mappings': {},
            'original_dtypes': {},
            'processed_columns': []
        }
        
        # Store original dtypes
        for col in data.columns:
            encoding_info['original_dtypes'][col] = data[col].dtype
        
        # Handle categorical columns
        for column in data.columns:
            if column in target_columns:
                continue
                
            if pd.api.types.is_object_dtype(data[column]) or pd.api.types.is_categorical_dtype(data[column]):
                # Create label encoding for categorical variables
                unique_values = data[column].dropna().unique()
                if len(unique_values) < 50:  # Only encode if reasonable number of categories
                    mapping = {val: idx for idx, val in enumerate(unique_values)}
                    mapping[np.nan] = -1  # Handle NaN values
                    
                    prepared_data[column] = data[column].map(mapping).fillna(-1)
                    encoding_info['categorical_mappings'][column] = mapping
                    encoding_info['processed_columns'].append(column)
                else:
                    # Drop columns with too many categories
                    prepared_data = prepared_data.drop(columns=[column])
        
        return prepared_data, encoding_info
    
    def reverse_encoding(self, data: pd.DataFrame, encoding_info: Dict) -> pd.DataFrame:
        """
        Reverse the encoding applied during preparation.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Encoded dataframe
        encoding_info : Dict
            Encoding information from prepare_for_imputation
            
        Returns:
        --------
        DataFrame with original encodings restored
        """
        result_data = data.copy()
        
        # Reverse categorical mappings
        for column, mapping in encoding_info['categorical_mappings'].items():
            if column in result_data.columns:
                # Create reverse mapping
                reverse_mapping = {idx: val for val, idx in mapping.items()}
                reverse_mapping[-1] = np.nan
                
                result_data[column] = result_data[column].map(reverse_mapping)
        
        # Restore original dtypes where possible
        for column, dtype in encoding_info['original_dtypes'].items():
            if column in result_data.columns:
                try:
                    if not pd.api.types.is_object_dtype(dtype):
                        result_data[column] = result_data[column].astype(dtype)
                except:
                    pass  # Keep current dtype if conversion fails
        
        return result_data
    
    def validate_data_quality(self, original_data: pd.DataFrame, imputed_data: pd.DataFrame) -> Dict:
        """
        Validate the quality of imputed data compared to original.
        
        Parameters:
        -----------
        original_data : pd.DataFrame
            Original dataframe before imputation
        imputed_data : pd.DataFrame
            Dataframe after imputation
            
        Returns:
        --------
        Dictionary containing quality metrics
        """
        quality_report = {
            'shape_consistency': original_data.shape == imputed_data.shape,
            'column_consistency': list(original_data.columns) == list(imputed_data.columns),
            'missing_data_reduction': {},
            'statistical_consistency': {},
            'data_type_consistency': {}
        }
        
        # Missing data reduction
        original_missing = original_data.isnull().sum()
        imputed_missing = imputed_data.isnull().sum()
        
        for column in original_data.columns:
            if original_missing[column] > 0:
                reduction = original_missing[column] - imputed_missing[column]
                quality_report['missing_data_reduction'][column] = {
                    'original_missing': original_missing[column],
                    'imputed_missing': imputed_missing[column],
                    'reduction': reduction,
                    'reduction_percentage': (reduction / original_missing[column]) * 100
                }
        
        # Statistical consistency for numeric columns
        numeric_columns = original_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in imputed_data.columns:
                orig_stats = original_data[column].describe()
                imp_stats = imputed_data[column].describe()
                
                quality_report['statistical_consistency'][column] = {
                    'mean_difference': abs(orig_stats['mean'] - imp_stats['mean']),
                    'std_difference': abs(orig_stats['std'] - imp_stats['std']),
                    'median_difference': abs(orig_stats['50%'] - imp_stats['50%']),
                    'range_difference': abs((orig_stats['max'] - orig_stats['min']) - 
                                          (imp_stats['max'] - imp_stats['min']))
                }
        
        # Data type consistency
        for column in original_data.columns:
            if column in imputed_data.columns:
                quality_report['data_type_consistency'][column] = {
                    'original_dtype': str(original_data[column].dtype),
                    'imputed_dtype': str(imputed_data[column].dtype),
                    'consistent': str(original_data[column].dtype) == str(imputed_data[column].dtype)
                }
        
        return quality_report
    
    def get_correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric columns.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        Correlation matrix as DataFrame
        """
        numeric_data = data.select_dtypes(include=[np.number])
        return numeric_data.corr()
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> Dict:
        """
        Detect outliers in numeric columns.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        method : str
            Method for outlier detection ('iqr' or 'zscore')
            
        Returns:
        --------
        Dictionary containing outlier information
        """
        outliers_info = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            col_data = data[column].dropna()
            outliers_info[column] = {'outlier_indices': [], 'outlier_values': []}
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(col_data))
                outlier_mask = z_scores > 3
            
            outlier_indices = col_data[outlier_mask].index.tolist()
            outlier_values = col_data[outlier_mask].values.tolist()
            
            outliers_info[column] = {
                'outlier_indices': outlier_indices,
                'outlier_values': outlier_values,
                'outlier_count': len(outlier_indices),
                'outlier_percentage': (len(outlier_indices) / len(col_data)) * 100
            }
        
        return outliers_info

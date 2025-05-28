import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SVRImputer:
    """
    Support Vector Regression-based imputer for missing values.
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', epsilon=0.1, 
                 degree=3, coef0=0.0, max_iter=1000, random_state=42):
        """
        Initialize SVR Imputer with specified parameters.
        
        Parameters:
        -----------
        kernel : str, default='rbf'
            Specifies the kernel type to be used in the algorithm
        C : float, default=1.0
            Regularization parameter
        gamma : {'scale', 'auto'} or float, default='scale'
            Kernel coefficient
        epsilon : float, default=0.1
            Epsilon-tube within which no penalty is associated
        degree : int, default=3
            Degree of the polynomial kernel function
        coef0 : float, default=0.0
            Independent term in kernel function
        max_iter : int, default=1000
            Hard limit on iterations within solver
        random_state : int, default=42
            Random state for reproducibility
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.scalers = {}
        self.models = {}
        self.feature_columns = {}
        self.imputation_order = []
        
    def _prepare_features(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target vector for SVR training.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        target_column : str
            Column to be predicted/imputed
            
        Returns:
        --------
        Tuple of feature matrix and target vector
        """
        # Select numeric columns as features (excluding target)
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_columns:
            numeric_columns.remove(target_column)
        
        # Use columns with less missing data as features
        feature_columns = []
        for col in numeric_columns:
            missing_ratio = data[col].isnull().sum() / len(data)
            if missing_ratio < 0.5:  # Use columns with less than 50% missing data
                feature_columns.append(col)
        
        if not feature_columns:
            # If no good features, use all available numeric columns
            feature_columns = [col for col in numeric_columns if col != target_column]
        
        # Store feature columns for this target
        self.feature_columns[target_column] = feature_columns
        
        # Get complete cases (rows without missing values in features and target)
        complete_mask = data[feature_columns + [target_column]].notna().all(axis=1)
        complete_data = data[complete_mask]
        
        if len(complete_data) < 10:
            raise ValueError(f"Not enough complete cases for training SVR model for {target_column}")
        
        X = complete_data[feature_columns].values
        y = complete_data[target_column].values
        
        return X, y
    
    def _determine_imputation_order(self, data: pd.DataFrame, target_columns: List[str]) -> List[str]:
        """
        Determine the order of imputation based on missing data patterns.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        target_columns : List[str]
            Columns to be imputed
            
        Returns:
        --------
        List of columns in imputation order
        """
        # Calculate missing data percentage for each column
        missing_percentages = {}
        for col in target_columns:
            missing_percentages[col] = data[col].isnull().sum() / len(data)
        
        # Sort by missing percentage (ascending - impute columns with less missing data first)
        ordered_columns = sorted(missing_percentages.items(), key=lambda x: x[1])
        return [col for col, _ in ordered_columns]
    
    def _train_svr_model(self, X: np.ndarray, y: np.ndarray, target_column: str) -> Tuple[SVR, StandardScaler, float]:
        """
        Train SVR model for a specific target column.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        target_column : str
            Name of target column
            
        Returns:
        --------
        Tuple of trained SVR model, scaler, and cross-validation score
        """
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize and train SVR model
        svr_model = SVR(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            epsilon=self.epsilon,
            degree=self.degree,
            coef0=self.coef0,
            max_iter=self.max_iter
        )
        
        # Train the model
        svr_model.fit(X_scaled, y)
        
        # Evaluate model performance using cross-validation
        try:
            cv_scores = cross_val_score(svr_model, X_scaled, y, cv=min(5, len(X)//2), 
                                      scoring='r2', error_score='raise')
            cv_score = np.mean(cv_scores)
        except:
            # Fallback to simple train score if cross-validation fails
            y_pred = svr_model.predict(X_scaled)
            cv_score = r2_score(y, y_pred)
        
        return svr_model, scaler, cv_score
    
    def _impute_column(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Impute missing values for a specific column using SVR.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        target_column : str
            Column to impute
            
        Returns:
        --------
        Tuple of dataframe with imputed values and imputation statistics
        """
        result_data = data.copy()
        missing_mask = result_data[target_column].isnull()
        
        if not missing_mask.any():
            return result_data, {'imputed_count': 0, 'model_score': None}
        
        try:
            # Prepare training data
            X_train, y_train = self._prepare_features(data, target_column)
            
            # Train SVR model
            model, scaler, cv_score = self._train_svr_model(X_train, y_train, target_column)
            
            # Store model and scaler
            self.models[target_column] = model
            self.scalers[target_column] = scaler
            
            # Prepare features for imputation
            feature_cols = self.feature_columns[target_column]
            X_missing = result_data.loc[missing_mask, feature_cols]
            
            # Handle missing values in features by using mean imputation
            X_missing_filled = X_missing.fillna(X_missing.mean())
            
            # Scale features and predict
            X_missing_scaled = scaler.transform(X_missing_filled.values)
            imputed_values = model.predict(X_missing_scaled)
            
            # Fill missing values
            result_data.loc[missing_mask, target_column] = imputed_values
            
            imputation_stats = {
                'imputed_count': missing_mask.sum(),
                'model_score': cv_score,
                'feature_columns_used': feature_cols,
                'training_samples': len(X_train)
            }
            
            return result_data, imputation_stats
            
        except Exception as e:
            print(f"Warning: Could not impute {target_column} using SVR. Error: {str(e)}")
            # Fallback to mean imputation
            mean_value = data[target_column].mean()
            result_data.loc[missing_mask, target_column] = mean_value
            
            imputation_stats = {
                'imputed_count': missing_mask.sum(),
                'model_score': None,
                'feature_columns_used': [],
                'training_samples': 0,
                'fallback_method': 'mean'
            }
            
            return result_data, imputation_stats
    
    def fit_transform(self, data: pd.DataFrame, target_columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit SVR models and transform data by imputing missing values.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe with missing values
        target_columns : List[str]
            List of columns to impute
            
        Returns:
        --------
        Tuple of imputed dataframe and detailed imputation report
        """
        if not target_columns:
            return data.copy(), {'imputed_counts': {}, 'model_scores': {}}
        
        # Validate target columns
        valid_columns = []
        for col in target_columns:
            if col not in data.columns:
                print(f"Warning: Column '{col}' not found in data")
                continue
            if not pd.api.types.is_numeric_dtype(data[col]):
                print(f"Warning: Column '{col}' is not numeric, skipping SVR imputation")
                continue
            if data[col].isnull().sum() == 0:
                print(f"Warning: Column '{col}' has no missing values")
                continue
            valid_columns.append(col)
        
        if not valid_columns:
            return data.copy(), {'imputed_counts': {}, 'model_scores': {}}
        
        # Determine imputation order
        self.imputation_order = self._determine_imputation_order(data, valid_columns)
        
        # Initialize result dataframe and report
        result_data = data.copy()
        imputation_report = {
            'imputed_counts': {},
            'model_scores': {},
            'feature_columns': {},
            'training_samples': {},
            'imputation_order': self.imputation_order,
            'fallback_methods': {}
        }
        
        # Impute each column in order
        for column in self.imputation_order:
            print(f"Imputing column: {column}")
            result_data, column_stats = self._impute_column(result_data, column)
            
            # Update report
            imputation_report['imputed_counts'][column] = column_stats['imputed_count']
            imputation_report['model_scores'][column] = column_stats['model_score']
            imputation_report['feature_columns'][column] = column_stats.get('feature_columns_used', [])
            imputation_report['training_samples'][column] = column_stats.get('training_samples', 0)
            
            if 'fallback_method' in column_stats:
                imputation_report['fallback_methods'][column] = column_stats['fallback_method']
        
        return result_data, imputation_report
    
    def get_model_details(self) -> Dict:
        """
        Get details about trained models.
        
        Returns:
        --------
        Dictionary containing model information
        """
        model_details = {}
        
        for column, model in self.models.items():
            model_details[column] = {
                'kernel': model.kernel,
                'C': model.C,
                'gamma': model.gamma,
                'epsilon': model.epsilon,
                'n_support': len(model.support_),
                'dual_coef_shape': model.dual_coef_.shape if hasattr(model, 'dual_coef_') else None,
                'feature_columns': self.feature_columns.get(column, [])
            }
        
        return model_details
    
    def predict_missing_only(self, data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Predict values only for missing entries using trained models.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataframe with missing values
        columns : List[str], optional
            Specific columns to impute. If None, use all trained models.
            
        Returns:
        --------
        DataFrame with predictions only for missing values
        """
        if columns is None:
            columns = list(self.models.keys())
        
        result_data = data.copy()
        
        for column in columns:
            if column not in self.models:
                print(f"Warning: No trained model found for column '{column}'")
                continue
            
            missing_mask = result_data[column].isnull()
            if not missing_mask.any():
                continue
            
            # Get model and scaler
            model = self.models[column]
            scaler = self.scalers[column]
            feature_cols = self.feature_columns[column]
            
            # Prepare features
            X_missing = result_data.loc[missing_mask, feature_cols]
            X_missing_filled = X_missing.fillna(X_missing.mean())
            X_missing_scaled = scaler.transform(X_missing_filled.values)
            
            # Predict and fill
            imputed_values = model.predict(X_missing_scaled)
            result_data.loc[missing_mask, column] = imputed_values
        
        return result_data

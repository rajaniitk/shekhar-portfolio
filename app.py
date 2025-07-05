from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import io
import base64
import warnings
from datetime import datetime, timedelta
import traceback
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import uuid
import sys
import re
import sqlite3
from collections import Counter
import math

# Scientific and statistical libraries
import scipy.stats as stats
from scipy.stats import (
    shapiro, normaltest, kstest, jarque_bera, anderson,
    ttest_1samp, ttest_ind, ttest_rel, levene, bartlett,
    pearsonr, spearmanr, kendalltau, chi2_contingency,
    fisher_exact, mannwhitneyu, wilcoxon, kruskal,
    friedmanchisquare, f_oneway, boxcox, yeojohnson,
    skew, kurtosis, zscore, iqr
)
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Machine learning libraries
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    OneHotEncoder, QuantileTransformer, PowerTransformer,
    PolynomialFeatures, Normalizer, MaxAbsScaler, Binarizer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, chi2, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold, SelectPercentile,
    SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, FactorAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
    MeanShift, AffinityPropagation, Birch, OPTICS
)
from sklearn.ensemble import (
    RandomForestClassifier, IsolationForest, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM, SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, r2_score, silhouette_score, adjusted_rand_score
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Time series analysis
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import linear_rainbow, linear_harvey_collier

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyo
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Text processing
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, HashingVectorizer
)

# Image processing for charts
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO

# Network analysis
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Word cloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eda_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def convert_np(obj):
    """Convert numpy objects to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Global configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = str(uuid.uuid4())
app.config['SESSION_TIMEOUT'] = 3600  # 1 hour

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for session data
session_data = {}

class SecurityManager:
    """Handle security and validation"""
    
    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """Validate session ID format"""
        if not session_id or not isinstance(session_id, str):
            return False
        return re.match(r'^[a-zA-Z0-9_-]+$', session_id) is not None
    
    @staticmethod
    def sanitize_column_name(name: str) -> str:
        """Sanitize column names"""
        return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    
    @staticmethod
    def validate_file_type(filename: str) -> bool:
        """Validate file type"""
        allowed_extensions = {'csv', 'xlsx', 'xls', 'json', 'parquet', 'txt'}
        return filename.lower().split('.')[-1] in allowed_extensions

class DataAnalyzer:


    def __init__(self, session_id: str):
        self.session_id = session_id
        self.data = None
        self.original_data = None
        self.metadata = {}
        self.transformation_history = []
        self.statistical_tests = []
        self.insights = []

    def get_data_preview(self, preview_type: str = 'head', n: int = 10) -> Dict[str, Any]:
        """Return a preview of the data (head or tail)"""
        try:
            if self.data is None:
                return {'data': [], 'columns': []}
            if preview_type == 'tail':
                preview_df = self.data.tail(n)
            else:
                preview_df = self.data.head(n)
            return {
                'data': preview_df.to_dict('records'),
                'columns': list(preview_df.columns)
            }
        except Exception as e:
            logger.error(f"Error in get_data_preview: {str(e)}")
            return {'data': [], 'columns': [], 'error': str(e)}
    """Comprehensive data analysis class with advanced EDA capabilities"""
        
    def load_data(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Load data from various file formats with error handling"""
        try:
            logger.info(f"Loading {file_type} file: {file_path}")
            
            if file_type == 'csv':
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        self.data = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode file with any encoding")
                    
            elif file_type in ['xlsx', 'xls']:
                self.data = pd.read_excel(file_path, engine='openpyxl' if file_type == 'xlsx' else None)
            elif file_type == 'json':
                self.data = pd.read_json(file_path, lines=True)
            elif file_type == 'parquet':
                self.data = pd.read_parquet(file_path)
            elif file_type == 'txt':
                # Try to read as CSV with tab separator
                self.data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Store original data
            self.original_data = self.data.copy()
            
            # Basic data cleaning
            self._basic_cleaning()
            
            # Generate metadata
            self.metadata = self._generate_metadata()
            
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            
            return {
                'success': True,
                'data': self.data.head(100).to_dict('records'),
                'columns': list(self.data.columns),
                'shape': self.data.shape,
                'metadata': self.metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _basic_cleaning(self):
        """Perform basic data cleaning"""
        try:
            # Remove completely empty rows and columns
            self.data = self.data.dropna(how='all')
            self.data = self.data.dropna(axis=1, how='all')
            
            # Clean column names
            self.data.columns = [self._clean_column_name(col) for col in self.data.columns]
            
            # Handle duplicate columns
            cols = pd.Series(self.data.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
            self.data.columns = cols
            
            # Basic type inference
            self._infer_data_types()
            
        except Exception as e:
            logger.warning(f"Basic cleaning failed: {str(e)}")
    
    def _clean_column_name(self, col_name: str) -> str:
        """Clean column name by removing special characters"""
        # Convert to string and handle None
        name = str(col_name) if col_name is not None else 'unnamed_column'
        
        # Remove leading/trailing whitespace
        name = name.strip()
        
        # Replace problematic characters
        name = re.sub(r'[^\w\s]', '_', name)
        name = re.sub(r'\s+', '_', name)
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Ensure it starts with a letter or underscore
        if name and not name[0].isalpha() and name[0] != '_':
            name = f'col_{name}'
        
        return name if name else 'unnamed_column'
    
    def _infer_data_types(self):
        """Infer and convert data types"""
        try:
            for col in self.data.columns:
                # Try to convert to numeric
                if self.data[col].dtype == 'object':
                    # Check if it's numeric
                    numeric_data = pd.to_numeric(self.data[col], errors='coerce')
                    if not numeric_data.isna().all():
                        non_null_ratio = numeric_data.notna().sum() / len(self.data[col])
                        if non_null_ratio > 0.8:  # If 80% can be converted to numeric
                            self.data[col] = numeric_data
                            continue
                    
                    # Try to convert to datetime
                    try:
                        datetime_data = pd.to_datetime(self.data[col], errors='coerce', infer_datetime_format=True)
                        if not datetime_data.isna().all():
                            non_null_ratio = datetime_data.notna().sum() / len(self.data[col])
                            if non_null_ratio > 0.8:  # If 80% can be converted to datetime
                                self.data[col] = datetime_data
                                continue
                    except:
                        pass
                    
                    # Check for boolean
                    unique_values = self.data[col].dropna().unique()
                    if len(unique_values) <= 2:
                        bool_mapping = {
                            'true': True, 'false': False, 'yes': True, 'no': False,
                            '1': True, '0': False, 'y': True, 'n': False,
                            'on': True, 'off': False
                        }
                        lower_values = [str(v).lower() for v in unique_values]
                        if all(v in bool_mapping for v in lower_values):
                            self.data[col] = self.data[col].str.lower().map(bool_mapping)
                            
        except Exception as e:
            logger.warning(f"Type inference failed: {str(e)}")
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate comprehensive metadata about the dataset"""
        try:
            metadata = {
                'shape': self.data.shape,
                'memory_usage': int(self.data.memory_usage(deep=True).sum()),
                'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
                'missing_values': self.data.isnull().sum().to_dict(),
                'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
                'unique_values': {col: int(self.data[col].nunique()) for col in self.data.columns},
                'duplicated_rows': int(self.data.duplicated().sum()),
                'column_types': self._classify_columns(),
                'data_quality_score': self._calculate_data_quality_score()
            }
            
            # Numerical statistics
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                desc_stats = self.data[numerical_cols].describe()
                metadata['numerical_stats'] = {
                    col: desc_stats[col].to_dict() for col in desc_stats.columns
                }
                
                # Add skewness and kurtosis
                metadata['skewness'] = self.data[numerical_cols].skew().to_dict()
                metadata['kurtosis'] = self.data[numerical_cols].kurtosis().to_dict()
            
            # Categorical statistics
            categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                metadata['categorical_stats'] = {}
                for col in categorical_cols:
                    try:
                        value_counts = self.data[col].value_counts().head(10)
                        metadata['categorical_stats'][col] = {
                            'unique_count': int(self.data[col].nunique()),
                            'top_value': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                            'top_value_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                            'value_counts': {str(k): int(v) for k, v in value_counts.items()},
                            'cardinality_ratio': self.data[col].nunique() / len(self.data)
                        }
                    except Exception as e:
                        logger.warning(f"Error processing categorical column {col}: {str(e)}")
                        metadata['categorical_stats'][col] = {'error': str(e)}
            
            # DateTime statistics
            # Fix: use valid pandas selectors for datetime columns
            datetime_cols = self.data.select_dtypes(include=[np.datetime64]).columns
            if len(datetime_cols) > 0:
                metadata['datetime_stats'] = {}
                for col in datetime_cols:
                    try:
                        metadata['datetime_stats'][col] = {
                            'min_date': str(self.data[col].min()),
                            'max_date': str(self.data[col].max()),
                            'date_range': str(self.data[col].max() - self.data[col].min()),
                            'frequency_analysis': self._analyze_date_frequency(self.data[col])
                        }
                    except Exception as e:
                        logger.warning(f"Error processing datetime column {col}: {str(e)}")
            return metadata
        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}")
            return {'error': str(e)}
    
    def _classify_columns(self) -> Dict[str, List[str]]:
        """Classify columns by their data types and characteristics"""
        try:
            column_types = {
                'numerical': [],
                'categorical': [],
                'datetime': [],
                'text': [],
                'boolean': [],
                'id_like': [],
                'constant': [],
                'high_cardinality': []
            }
            
            for col in self.data.columns:
                dtype = str(self.data[col].dtype)
                
                # Check for constant columns
                if self.data[col].nunique() <= 1:
                    column_types['constant'].append(col)
                    continue
                
                # Numerical columns
                if self.data[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int16', 'int8', 'float16']:
                    column_types['numerical'].append(col)
                    
                    # Check if it's ID-like (high cardinality, mostly unique)
                    uniqueness_ratio = self.data[col].nunique() / len(self.data)
                    if uniqueness_ratio > 0.95:
                        column_types['id_like'].append(col)
                
                # Boolean columns
                elif self.data[col].dtype == 'bool':
                    column_types['boolean'].append(col)
                
                # DateTime columns
                elif 'datetime' in dtype:
                    column_types['datetime'].append(col)
                
                # Object/string columns
                elif dtype == 'object':
                    unique_ratio = self.data[col].nunique() / len(self.data)
                    avg_length = self.data[col].astype(str).str.len().mean()
                    
                    # High cardinality check
                    if self.data[col].nunique() > 50 and unique_ratio > 0.7:
                        column_types['high_cardinality'].append(col)
                        if uniqueness_ratio > 0.95:
                            column_types['id_like'].append(col)
                    
                    # Text vs categorical
                    if avg_length > 50 or unique_ratio > 0.8:
                        column_types['text'].append(col)
                    else:
                        column_types['categorical'].append(col)
                
                else:
                    column_types['categorical'].append(col)
            
            return column_types
            
        except Exception as e:
            logger.error(f"Error classifying columns: {str(e)}")
            return {}
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        try:
            score = 100.0
            
            # Penalize for missing data
            missing_ratio = self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])
            score -= missing_ratio * 30
            
            # Penalize for duplicate rows
            duplicate_ratio = self.data.duplicated().sum() / len(self.data)
            score -= duplicate_ratio * 20
            
            # Penalize for constant columns
            constant_columns = sum(1 for col in self.data.columns if self.data[col].nunique() <= 1)
            constant_ratio = constant_columns / len(self.data.columns)
            score -= constant_ratio * 15
            
            # Bonus for diverse data types
            unique_types = len(set(str(dtype) for dtype in self.data.dtypes))
            if unique_types > 3:
                score += 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating data quality score: {str(e)}")
            return 0.0
    
    def _analyze_date_frequency(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze frequency patterns in datetime data"""
        try:
            clean_series = series.dropna()
            if len(clean_series) < 2:
                return {}
            
            # Sort and calculate differences
            sorted_series = clean_series.sort_values()
            diffs = sorted_series.diff().dropna()
            
            if len(diffs) == 0:
                return {}
            
            # Most common difference
            most_common_diff = diffs.mode()
            if len(most_common_diff) > 0:
                most_common_diff = most_common_diff.iloc[0]
            else:
                most_common_diff = diffs.median()
            
            return {
                'most_common_interval': str(most_common_diff),
                'min_interval': str(diffs.min()),
                'max_interval': str(diffs.max()),
                'median_interval': str(diffs.median()),
                'regular_intervals': len(diffs.unique()) <= 1,
                'interval_count': len(diffs.unique())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing date frequency: {str(e)}")
            return {}
    
    def perform_comprehensive_eda(self, columns: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive exploratory data analysis"""
        try:
            if columns is None:
                columns = list(self.data.columns)
            
            logger.info(f"Performing EDA on {len(columns)} columns")
            
            results = {
                'basic_info': self._get_basic_info(),
                'univariate_analysis': self._perform_univariate_analysis(columns),
                'bivariate_analysis': self._perform_bivariate_analysis(columns),
                'correlation_analysis': self._perform_correlation_analysis(),
                'outlier_analysis': self._perform_outlier_analysis(columns),
                'missing_value_analysis': self._analyze_missing_values(),
                'data_quality_report': self._generate_data_quality_report(),
                'insights': self._generate_insights(),
                'distribution_analysis': self._analyze_distributions(columns),
                'feature_importance': self._calculate_feature_importance(columns)
            }
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            logger.error(f"Error performing EDA: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _get_basic_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        try:
            info = {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
                'memory_usage': int(self.data.memory_usage(deep=True).sum()),
                'missing_values': int(self.data.isnull().sum().sum()),
                'duplicates': int(self.data.duplicated().sum()),
                'column_types': self.metadata.get('column_types', {}),
                'data_quality_score': self.metadata.get('data_quality_score', 0)
            }
            
            # Add file size information
            info['file_size_mb'] = info['memory_usage'] / (1024 * 1024)
            
            # Add sparsity information
            if self.data.shape[0] > 0 and self.data.shape[1] > 0:
                total_cells = self.data.shape[0] * self.data.shape[1]
                missing_cells = info['missing_values']
                info['sparsity'] = missing_cells / total_cells
                info['density'] = 1 - info['sparsity']
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting basic info: {str(e)}")
            return {}
    
    def _perform_univariate_analysis(self, columns: List[str]) -> Dict[str, Any]:
        """Perform comprehensive univariate analysis"""
        try:
            results = {}
            
            for col in columns:
                if col not in self.data.columns:
                    continue
                
                col_analysis = {
                    'dtype': str(self.data[col].dtype),
                    'missing_count': int(self.data[col].isnull().sum()),
                    'missing_percentage': float(self.data[col].isnull().sum() / len(self.data) * 100),
                    'unique_count': int(self.data[col].nunique()),
                    'unique_percentage': float(self.data[col].nunique() / len(self.data) * 100)
                }
                
                # Numerical analysis
                if self.data[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                    col_analysis.update(self._analyze_numerical_column(col))
                
                # Categorical analysis
                elif self.data[col].dtype in ['object', 'category']:
                    col_analysis.update(self._analyze_categorical_column(col))
                
                # DateTime analysis
                elif 'datetime' in str(self.data[col].dtype):
                    col_analysis.update(self._analyze_datetime_column(col))
                
                # Boolean analysis
                elif self.data[col].dtype == 'bool':
                    col_analysis.update(self._analyze_boolean_column(col))
                
                results[col] = col_analysis
            
            return results
            
        except Exception as e:
            logger.error(f"Error in univariate analysis: {str(e)}")
            return {}
    
    def _analyze_numerical_column(self, col: str) -> Dict[str, Any]:
        """Analyze a numerical column"""
        try:
            series = self.data[col].dropna()
            if len(series) == 0:
                return {'error': 'No non-null values'}
            
            analysis = {
                'statistics': {
                    'count': len(series),
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'q25': float(series.quantile(0.25)),
                    'q75': float(series.quantile(0.75)),
                    'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                    'range': float(series.max() - series.min()),
                    'variance': float(series.var())
                },
                'shape_measures': {
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis())
                },
                'outliers': self._detect_outliers_comprehensive(series),
                'distribution': self._identify_distribution(series),
                'normality_tests': self._test_normality(series),
                'percentiles': {
                    f'p{p}': float(series.quantile(p/100)) 
                    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
                }
            }
            
            # Add coefficient of variation
            if analysis['statistics']['mean'] != 0:
                analysis['statistics']['cv'] = abs(analysis['statistics']['std'] / analysis['statistics']['mean'])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing numerical column {col}: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_categorical_column(self, col: str) -> Dict[str, Any]:
        """Analyze a categorical column"""
        try:
            series = self.data[col].dropna()
            if len(series) == 0:
                return {'error': 'No non-null values'}
            
            value_counts = series.value_counts()
            
            analysis = {
                'value_counts': {str(k): int(v) for k, v in value_counts.head(20).items()},
                'entropy': self._calculate_entropy(series),
                'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'mode_percentage': float(value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0,
                'cardinality_ratio': float(series.nunique() / len(series)),
                'is_high_cardinality': series.nunique() > 50,
                'concentration_ratio': self._calculate_concentration_ratio(value_counts)
            }
            
            # Check for patterns in categorical data
            if all(str(val).isdigit() for val in series.unique()[:10]):
                analysis['appears_numeric'] = True
            
            if all(len(str(val)) > 20 for val in series.unique()[:10]):
                analysis['appears_textual'] = True
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing categorical column {col}: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_datetime_column(self, col: str) -> Dict[str, Any]:
        """Analyze a datetime column"""
        try:
            series = self.data[col].dropna()
            if len(series) == 0:
                return {'error': 'No non-null values'}
            
            analysis = {
                'date_range': {
                    'min': str(series.min()),
                    'max': str(series.max()),
                    'span': str(series.max() - series.min())
                },
                'frequency_analysis': self._analyze_date_frequency(series),
                'temporal_patterns': self._analyze_temporal_patterns(series)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing datetime column {col}: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_boolean_column(self, col: str) -> Dict[str, Any]:
        """Analyze a boolean column"""
        try:
            series = self.data[col].dropna()
            if len(series) == 0:
                return {'error': 'No non-null values'}
            
            value_counts = series.value_counts()
            
            analysis = {
                'true_count': int(value_counts.get(True, 0)),
                'false_count': int(value_counts.get(False, 0)),
                'true_percentage': float(value_counts.get(True, 0) / len(series) * 100),
                'false_percentage': float(value_counts.get(False, 0) / len(series) * 100),
                'balance_ratio': float(min(value_counts) / max(value_counts)) if len(value_counts) == 2 else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing boolean column {col}: {str(e)}")
            return {'error': str(e)}
    
    def _detect_outliers_comprehensive(self, series: pd.Series) -> Dict[str, Any]:
        """Comprehensive outlier detection using multiple methods"""
        try:
            outliers = {}
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            outliers['iqr'] = {
                'count': len(iqr_outliers),
                'percentage': float(len(iqr_outliers) / len(series) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_values': iqr_outliers.head(20).tolist()
            }
            
            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            zscore_outliers = series[z_scores > 3]
            outliers['zscore'] = {
                'count': len(zscore_outliers),
                'percentage': float(len(zscore_outliers) / len(series) * 100),
                'threshold': 3.0,
                'outlier_values': zscore_outliers.head(20).tolist()
            }
            
            # Modified Z-score (using MAD)
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else np.zeros(len(series))
            mad_outliers = series[np.abs(modified_z_scores) > 3.5]
            outliers['modified_zscore'] = {
                'count': len(mad_outliers),
                'percentage': float(len(mad_outliers) / len(series) * 100),
                'threshold': 3.5,
                'outlier_values': mad_outliers.head(20).tolist()
            }
            
            # Isolation Forest (if series is large enough)
            if len(series) > 10:
                try:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_preds = iso_forest.fit_predict(series.values.reshape(-1, 1))
                    iso_outliers = series[outlier_preds == -1]
                    outliers['isolation_forest'] = {
                        'count': len(iso_outliers),
                        'percentage': float(len(iso_outliers) / len(series) * 100),
                        'outlier_values': iso_outliers.head(20).tolist()
                    }
                except Exception as e:
                    outliers['isolation_forest'] = {'error': str(e)}
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return {}
    
    def _identify_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Identify the distribution type of a numerical series"""
        try:
            clean_series = series.dropna()
            if len(clean_series) < 10:
                return {'type': 'insufficient_data'}
            
            # Sample large datasets
            if len(clean_series) > 5000:
                clean_series = clean_series.sample(5000, random_state=42)
            
            distribution_info = {
                'skewness': float(clean_series.skew()),
                'kurtosis': float(clean_series.kurtosis()),
                'type': 'unknown'
            }
            
            # Classify based on skewness and kurtosis
            skewness = distribution_info['skewness']
            kurtosis = distribution_info['kurtosis']
            
            if abs(skewness) < 0.5:
                if abs(kurtosis) < 0.5:
                    distribution_info['type'] = 'normal'
                elif kurtosis > 0.5:
                    distribution_info['type'] = 'leptokurtic'
                else:
                    distribution_info['type'] = 'platykurtic'
            elif skewness > 0.5:
                distribution_info['type'] = 'right_skewed'
            elif skewness < -0.5:
                distribution_info['type'] = 'left_skewed'
            
            # Additional distribution tests
            try:
                # Test for uniform distribution
                ks_stat, ks_p = stats.kstest(clean_series, 'uniform')
                if ks_p > 0.05:
                    distribution_info['possible_uniform'] = True
                
                # Test for exponential distribution
                exp_stat, exp_p = stats.kstest(clean_series, 'expon')
                if exp_p > 0.05:
                    distribution_info['possible_exponential'] = True
                    
            except Exception as e:
                logger.warning(f"Distribution testing failed: {str(e)}")
            
            return distribution_info
            
        except Exception as e:
            logger.error(f"Error identifying distribution: {str(e)}")
            return {'type': 'error', 'error': str(e)}
    
    def _test_normality(self, series: pd.Series) -> Dict[str, Any]:
        """Test for normality using multiple tests"""
        try:
            clean_series = series.dropna()
            if len(clean_series) < 3:
                return {'error': 'Insufficient data for normality testing'}
            
            # Sample large datasets for normality tests
            if len(clean_series) > 5000:
                clean_series = clean_series.sample(5000, random_state=42)
            
            tests = {}
            
            # Shapiro-Wilk test (best for smaller samples)
            try:
                stat, p_value = stats.shapiro(clean_series)
                tests['shapiro_wilk'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
            except Exception as e:
                tests['shapiro_wilk'] = {'error': str(e)}
            
            # D'Agostino-Pearson test
            try:
                stat, p_value = stats.normaltest(clean_series)
                tests['dagostino_pearson'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
            except Exception as e:
                tests['dagostino_pearson'] = {'error': str(e)}
            
            # Anderson-Darling test
            try:
                result = stats.anderson(clean_series, dist='norm')
                tests['anderson_darling'] = {
                    'statistic': float(result.statistic),
                    'critical_values': result.critical_values.tolist(),
                    'significance_levels': result.significance_level.tolist()
                }
            except Exception as e:
                tests['anderson_darling'] = {'error': str(e)}
            
            # Jarque-Bera test
            try:
                stat, p_value = stats.jarque_bera(clean_series)
                tests['jarque_bera'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
            except Exception as e:
                tests['jarque_bera'] = {'error': str(e)}
            
            return tests
            
        except Exception as e:
            logger.error(f"Error testing normality: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate Shannon entropy for categorical data"""
        try:
            value_counts = series.value_counts()
            probabilities = value_counts / len(series)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return float(entropy)
        except Exception as e:
            logger.error(f"Error calculating entropy: {str(e)}")
            return 0.0
    
    def _calculate_concentration_ratio(self, value_counts: pd.Series) -> float:
        """Calculate concentration ratio (top 5 values percentage)"""
        try:
            total = value_counts.sum()
            top5_sum = value_counts.head(5).sum()
            return float(top5_sum / total)
        except Exception as e:
            logger.error(f"Error calculating concentration ratio: {str(e)}")
            return 0.0
    
    def _analyze_temporal_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze temporal patterns in datetime data"""
        try:
            patterns = {}
            
            # Day of week patterns
            dow_counts = series.dt.day_name().value_counts()
            patterns['day_of_week'] = dow_counts.to_dict()
            
            # Month patterns
            month_counts = series.dt.month_name().value_counts()
            patterns['month'] = month_counts.to_dict()
            
            # Hour patterns (if time component exists)
            if series.dt.hour.nunique() > 1:
                hour_counts = series.dt.hour.value_counts().sort_index()
                patterns['hour'] = hour_counts.to_dict()
            
            # Yearly trends
            if series.dt.year.nunique() > 1:
                year_counts = series.dt.year.value_counts().sort_index()
                patterns['year'] = year_counts.to_dict()
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {str(e)}")
            return {}
    
    def _perform_bivariate_analysis(self, columns: List[str]) -> Dict[str, Any]:
        """Perform comprehensive bivariate analysis"""
        try:
            results = {}
            
            numerical_cols = [col for col in columns if self.data[col].dtype in ['int64', 'int32', 'float64', 'float32']]
            categorical_cols = [col for col in columns if self.data[col].dtype in ['object', 'category']]
            
            # Numerical vs Numerical
            if len(numerical_cols) > 1:
                results['numerical_pairs'] = self._analyze_numerical_pairs(numerical_cols)
            
            # Numerical vs Categorical
            if numerical_cols and categorical_cols:
                results['numerical_categorical_pairs'] = self._analyze_numerical_categorical_pairs(numerical_cols, categorical_cols)
            
            # Categorical vs Categorical
            if len(categorical_cols) > 1:
                results['categorical_pairs'] = self._analyze_categorical_pairs(categorical_cols)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in bivariate analysis: {str(e)}")
            return {}
    
    def _analyze_numerical_pairs(self, numerical_cols: List[str]) -> Dict[str, Any]:
        """Analyze relationships between numerical column pairs"""
        try:
            pairs = {}
            
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    pair_key = f"{col1}_vs_{col2}"
                    
                    # Get paired data (remove rows with missing values in either column)
                    paired_data = self.data[[col1, col2]].dropna()
                    if len(paired_data) < 3:
                        continue
                    
                    analysis = {}
                    
                    # Correlation analysis
                    analysis['correlations'] = {
                        'pearson': float(paired_data[col1].corr(paired_data[col2], method='pearson')),
                        'spearman': float(paired_data[col1].corr(paired_data[col2], method='spearman')),
                        'kendall': float(paired_data[col1].corr(paired_data[col2], method='kendall'))
                    }
                    
                    # Covariance
                    analysis['covariance'] = float(paired_data[col1].cov(paired_data[col2]))
                    
                    # Linear regression
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            paired_data[col1], paired_data[col2]
                        )
                        analysis['linear_regression'] = {
                            'slope': float(slope),
                            'intercept': float(intercept),
                            'r_squared': float(r_value ** 2),
                            'p_value': float(p_value),
                            'std_error': float(std_err)
                        }
                    except Exception as e:
                        analysis['linear_regression'] = {'error': str(e)}
                    
                    # Mutual information
                    try:
                        from sklearn.feature_selection import mutual_info_regression
                        mi_score = mutual_info_regression(
                            paired_data[[col1]], paired_data[col2], random_state=42
                        )[0]
                        analysis['mutual_information'] = float(mi_score)
                    except Exception as e:
                        analysis['mutual_information'] = {'error': str(e)}
                    
                    pairs[pair_key] = analysis
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error analyzing numerical pairs: {str(e)}")
            return {}
    
    def _analyze_numerical_categorical_pairs(self, numerical_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
        """Analyze relationships between numerical and categorical variables"""
        try:
            pairs = {}
            
            for num_col in numerical_cols[:5]:  # Limit to 5 to avoid long processing
                for cat_col in categorical_cols[:5]:
                    pair_key = f"{num_col}_vs_{cat_col}"
                    
                    try:
                        # Group statistics
                        grouped_stats = self.data.groupby(cat_col)[num_col].agg([
                            'count', 'mean', 'median', 'std', 'min', 'max'
                        ]).to_dict()
                        
                        # ANOVA test
                        groups = [group[num_col].dropna() for name, group in self.data.groupby(cat_col)]
                        groups = [group for group in groups if len(group) > 0]
                        
                        if len(groups) > 1:
                            f_stat, p_value = stats.f_oneway(*groups)
                            anova_result = {
                                'f_statistic': float(f_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                        else:
                            anova_result = {'error': 'Insufficient groups for ANOVA'}
                        
                        # Effect size (eta-squared)
                        try:
                            ss_total = self.data[num_col].var() * (len(self.data[num_col]) - 1)
                            ss_between = sum(
                                len(group) * (group.mean() - self.data[num_col].mean()) ** 2 
                                for group in groups
                            )
                            eta_squared = ss_between / ss_total if ss_total > 0 else 0
                            anova_result['eta_squared'] = float(eta_squared)
                        except:
                            pass
                        
                        pairs[pair_key] = {
                            'grouped_statistics': grouped_stats,
                            'anova': anova_result
                        }
                        
                    except Exception as e:
                        pairs[pair_key] = {'error': str(e)}
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error analyzing numerical-categorical pairs: {str(e)}")
            return {}
    
    def _analyze_categorical_pairs(self, categorical_cols: List[str]) -> Dict[str, Any]:
        """Analyze relationships between categorical variable pairs"""
        try:
            pairs = {}
            
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    pair_key = f"{col1}_vs_{col2}"
                    
                    try:
                        # Create contingency table
                        contingency_table = pd.crosstab(self.data[col1], self.data[col2])
                        
                        if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                            continue
                        
                        # Chi-square test
                        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                        
                        # Cramér's V
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                        
                        # Phi coefficient (for 2x2 tables)
                        phi = None
                        if contingency_table.shape == (2, 2):
                            phi = np.sqrt(chi2_stat / n)
                        
                        analysis = {
                            'contingency_table': {
                                str(idx): {str(col): int(val) for col, val in row.items()}
                                for idx, row in contingency_table.to_dict('index').items()
                            },
                            'chi_square': {
                                'statistic': float(chi2_stat),
                                'p_value': float(p_value),
                                'degrees_freedom': int(dof),
                                'significant': p_value < 0.05
                            },
                            'cramers_v': float(cramers_v)
                        }
                        
                        if phi is not None:
                            analysis['phi_coefficient'] = float(phi)
                        
                        pairs[pair_key] = analysis
                        
                    except Exception as e:
                        pairs[pair_key] = {'error': str(e)}
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error analyzing categorical pairs: {str(e)}")
            return {}
    
    def _perform_correlation_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis"""
        try:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) < 2:
                return {'error': 'Insufficient numerical columns for correlation analysis'}
            
            # Calculate different correlation matrices
            correlation_results = {}
            
            for method in ['pearson', 'spearman', 'kendall']:
                try:
                    corr_matrix = self.data[numerical_cols].corr(method=method)
                    correlation_results[method] = corr_matrix.to_dict()
                except Exception as e:
                    correlation_results[method] = {'error': str(e)}
            
            # Find high correlations
            high_correlations = []
            if 'pearson' in correlation_results and 'error' not in correlation_results['pearson']:
                pearson_corr = pd.DataFrame(correlation_results['pearson'])
                
                for i in range(len(pearson_corr.columns)):
                    for j in range(i+1, len(pearson_corr.columns)):
                        col1, col2 = pearson_corr.columns[i], pearson_corr.columns[j]
                        corr_value = pearson_corr.iloc[i, j]
                        
                        if not pd.isna(corr_value) and abs(corr_value) > 0.5:
                            high_correlations.append({
                                'variable1': col1,
                                'variable2': col2,
                                'correlation': float(corr_value),
                                'strength': self._interpret_correlation(abs(corr_value)),
                                'direction': 'positive' if corr_value > 0 else 'negative'
                            })
                
                # Sort by absolute correlation value
                high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
                correlation_results['high_correlations'] = high_correlations
            
            # Correlation network analysis
            if NETWORKX_AVAILABLE and len(high_correlations) > 0:
                try:
                    correlation_results['network_metrics'] = self._analyze_correlation_network(high_correlations)
                except Exception as e:
                    logger.warning(f"Network analysis failed: {str(e)}")
            
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {}
    
    def _interpret_correlation(self, corr_value: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(corr_value)
        if abs_corr >= 0.9:
            return 'very_strong'
        elif abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    def _analyze_correlation_network(self, high_correlations: List[Dict]) -> Dict[str, Any]:
        """Analyze correlation as a network"""
        try:
            if not NETWORKX_AVAILABLE:
                return {'error': 'NetworkX not available'}
            
            # Create network graph
            G = nx.Graph()
            
            for corr in high_correlations:
                G.add_edge(
                    corr['variable1'], 
                    corr['variable2'], 
                    weight=abs(corr['correlation'])
                )
            
            if len(G.nodes) == 0:
                return {'error': 'No network nodes'}
            
            # Calculate network metrics
            metrics = {
                'number_of_nodes': G.number_of_nodes(),
                'number_of_edges': G.number_of_edges(),
                'density': nx.density(G),
                'is_connected': nx.is_connected(G)
            }
            
            if G.number_of_nodes() > 0:
                # Centrality measures
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                
                metrics['most_central_variable'] = max(degree_centrality, key=degree_centrality.get)
                metrics['highest_betweenness'] = max(betweenness_centrality, key=betweenness_centrality.get)
                
                # Community detection
                if G.number_of_edges() > 0:
                    try:
                        communities = list(nx.community.greedy_modularity_communities(G))
                        metrics['communities'] = [list(community) for community in communities]
                        metrics['modularity'] = nx.community.modularity(G, communities)
                    except:
                        pass
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in network analysis: {str(e)}")
            return {'error': str(e)}
    
    def _perform_outlier_analysis(self, columns: List[str]) -> Dict[str, Any]:
        """Perform comprehensive outlier analysis"""
        try:
            numerical_cols = [col for col in columns if self.data[col].dtype in ['int64', 'int32', 'float64', 'float32']]
            outlier_results = {}
            
            for col in numerical_cols:
                try:
                    col_results = self._detect_outliers_comprehensive(self.data[col])
                    
                    # Add multivariate outlier detection if enough columns
                    if len(numerical_cols) > 2:
                        col_results['multivariate'] = self._detect_multivariate_outliers(numerical_cols, col)
                    
                    outlier_results[col] = col_results
                    
                except Exception as e:
                    outlier_results[col] = {'error': str(e)}
            
            # Overall outlier summary
            if outlier_results:
                outlier_results['summary'] = self._summarize_outliers(outlier_results)
            
            return outlier_results
            
        except Exception as e:
            logger.error(f"Error in outlier analysis: {str(e)}")
            return {}
    
    def _detect_multivariate_outliers(self, numerical_cols: List[str], target_col: str) -> Dict[str, Any]:
        """Detect multivariate outliers"""
        try:
            # Use other numerical columns to detect outliers in target column
            feature_cols = [col for col in numerical_cols if col != target_col][:5]  # Limit to 5
            
            if len(feature_cols) < 2:
                return {'error': 'Insufficient columns for multivariate analysis'}
            
            # Prepare data
            features = self.data[feature_cols + [target_col]].dropna()
            if len(features) < 10:
                return {'error': 'Insufficient data points'}
            
            results = {}
            
            # Mahalanobis distance
            try:
                mean = features.mean().values
                cov = features.cov().values
                
                # Check if covariance matrix is invertible
                if np.linalg.det(cov) != 0:
                    inv_cov = np.linalg.inv(cov)
                    
                    mahal_distances = []
                    for _, row in features.iterrows():
                        diff = row.values - mean
                        mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                        mahal_distances.append(mahal_dist)
                    
                    # Outliers based on chi-square distribution
                    threshold = np.sqrt(stats.chi2.ppf(0.975, len(feature_cols) + 1))
                    outlier_indices = [i for i, dist in enumerate(mahal_distances) if dist > threshold]
                    
                    results['mahalanobis'] = {
                        'outlier_count': len(outlier_indices),
                        'outlier_percentage': len(outlier_indices) / len(features) * 100,
                        'threshold': float(threshold)
                    }
                
            except Exception as e:
                results['mahalanobis'] = {'error': str(e)}
            
            # Isolation Forest on multivariate data
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_preds = iso_forest.fit_predict(features)
                outlier_count = sum(1 for pred in outlier_preds if pred == -1)
                
                results['isolation_forest_multivariate'] = {
                    'outlier_count': outlier_count,
                    'outlier_percentage': outlier_count / len(features) * 100
                }
                
            except Exception as e:
                results['isolation_forest_multivariate'] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multivariate outlier detection: {str(e)}")
            return {'error': str(e)}
    
    def _summarize_outliers(self, outlier_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize outlier analysis across all columns"""
        try:
            summary = {
                'total_columns_analyzed': 0,
                'columns_with_outliers': 0,
                'average_outlier_percentage': 0,
                'method_comparison': {}
            }
            
            method_totals = {}
            
            for col, results in outlier_results.items():
                if col == 'summary' or 'error' in results:
                    continue
                
                summary['total_columns_analyzed'] += 1
                
                column_has_outliers = False
                for method, method_results in results.items():
                    if isinstance(method_results, dict) and 'percentage' in method_results:
                        if method_results['percentage'] > 0:
                            column_has_outliers = True
                        
                        if method not in method_totals:
                            method_totals[method] = []
                        method_totals[method].append(method_results['percentage'])
                
                if column_has_outliers:
                    summary['columns_with_outliers'] += 1
            
            # Calculate averages for each method
            for method, percentages in method_totals.items():
                summary['method_comparison'][method] = {
                    'average_percentage': sum(percentages) / len(percentages),
                    'max_percentage': max(percentages),
                    'min_percentage': min(percentages)
                }
            
            if summary['total_columns_analyzed'] > 0:
                summary['percentage_columns_with_outliers'] = (
                    summary['columns_with_outliers'] / summary['total_columns_analyzed'] * 100
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing outliers: {str(e)}")
            return {}
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing value patterns comprehensively"""
        try:
            missing_analysis = {
                'overview': {
                    'total_missing': int(self.data.isnull().sum().sum()),
                    'total_cells': int(self.data.shape[0] * self.data.shape[1]),
                    'missing_percentage': float(self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1]) * 100)
                },
                'by_column': {},
                'by_row': {},
                'patterns': {},
                'correlations': {}
            }
            
            # By column analysis
            for col in self.data.columns:
                missing_count = self.data[col].isnull().sum()
                missing_analysis['by_column'][col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float(missing_count / len(self.data) * 100),
                    'data_type': str(self.data[col].dtype)
                }
            
            # By row analysis
            row_missing = self.data.isnull().sum(axis=1)
            missing_analysis['by_row'] = {
                'rows_with_missing': int((row_missing > 0).sum()),
                'max_missing_per_row': int(row_missing.max()),
                'avg_missing_per_row': float(row_missing.mean()),
                'rows_completely_missing': int((row_missing == len(self.data.columns)).sum())
            }
            
            # Missing value patterns
            missing_patterns = self.data.isnull().value_counts().head(10)
            missing_analysis['patterns']['top_patterns'] = {
                str(pattern): int(count) for pattern, count in missing_patterns.items()
            }
            
            # Missing value correlations
            if len(self.data.columns) > 1:
                missing_df = self.data.isnull().astype(int)
                missing_corr = missing_df.corr()
                
                # Find highly correlated missing patterns
                high_corr_pairs = []
                for i in range(len(missing_corr.columns)):
                    for j in range(i+1, len(missing_corr.columns)):
                        col1, col2 = missing_corr.columns[i], missing_corr.columns[j]
                        corr_value = missing_corr.iloc[i, j]
                        
                        if not pd.isna(corr_value) and abs(corr_value) > 0.5:
                            high_corr_pairs.append({
                                'column1': col1,
                                'column2': col2,
                                'correlation': float(corr_value)
                            })
                
                missing_analysis['correlations']['high_correlations'] = high_corr_pairs
            
            return missing_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing missing values: {str(e)}")
            return {}
    
    def _generate_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        try:
            quality_report = {
                'overall_score': self._calculate_data_quality_score(),
                'completeness': self._assess_completeness(),
                'consistency': self._assess_consistency(),
                'validity': self._assess_validity(),
                'uniqueness': self._assess_uniqueness(),
                'accuracy': self._assess_accuracy(),
                'recommendations': []
            }
            
            # Generate recommendations based on quality issues
            quality_report['recommendations'] = self._generate_quality_recommendations(quality_report)
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {str(e)}")
            return {}
    
    def _assess_completeness(self) -> Dict[str, Any]:
        """Assess data completeness"""
        try:
            total_cells = self.data.shape[0] * self.data.shape[1]
            missing_cells = self.data.isnull().sum().sum()
            
            completeness = {
                'overall_completeness': float((total_cells - missing_cells) / total_cells * 100),
                'column_completeness': {},
                'row_completeness_distribution': {}
            }
            
            # Column-wise completeness
            for col in self.data.columns:
                missing_count = self.data[col].isnull().sum()
                completeness['column_completeness'][col] = float((len(self.data) - missing_count) / len(self.data) * 100)
            
            # Row completeness distribution
            row_completeness = (self.data.shape[1] - self.data.isnull().sum(axis=1)) / self.data.shape[1] * 100
            completeness['row_completeness_distribution'] = {
                '100%': int((row_completeness == 100).sum()),
                '90-99%': int(((row_completeness >= 90) & (row_completeness < 100)).sum()),
                '75-89%': int(((row_completeness >= 75) & (row_completeness < 90)).sum()),
                '50-74%': int(((row_completeness >= 50) & (row_completeness < 75)).sum()),
                '<50%': int((row_completeness < 50).sum())
            }
            
            return completeness
            
        except Exception as e:
            logger.error(f"Error assessing completeness: {str(e)}")
            return {}
    
    def _assess_consistency(self) -> Dict[str, Any]:
        """Assess data consistency"""
        try:
            consistency = {
                'data_type_consistency': {},
                'value_format_consistency': {},
                'encoding_consistency': {}
            }
            
            # Data type consistency
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    # Check for mixed types in object columns
                    sample_values = self.data[col].dropna().head(100)
                    type_diversity = len(set(type(val).__name__ for val in sample_values))
                    consistency['data_type_consistency'][col] = {
                        'type_diversity': type_diversity,
                        'predominantly_numeric': self._check_numeric_strings(sample_values),
                        'predominantly_datetime': self._check_datetime_strings(sample_values)
                    }
            
            # Value format consistency (for string columns)
            string_cols = self.data.select_dtypes(include=['object']).columns
            for col in string_cols:
                consistency['value_format_consistency'][col] = self._check_format_consistency(col)
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error assessing consistency: {str(e)}")
            return {}
    
    def _check_numeric_strings(self, series: pd.Series) -> Dict[str, Any]:
        """Check if string values appear to be numeric"""
        try:
            numeric_pattern = series.astype(str).str.match(r'^-?\d*\.?\d+$')
            return {
                'percentage': float(numeric_pattern.sum() / len(series) * 100),
                'count': int(numeric_pattern.sum())
            }
        except:
            return {'percentage': 0, 'count': 0}
    
    def _check_datetime_strings(self, series: pd.Series) -> Dict[str, Any]:
        """Check if string values appear to be datetime"""
        try:
            # Simple heuristic: look for date-like patterns
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                r'\d{4}/\d{2}/\d{2}'   # YYYY/MM/DD
            ]
            
            date_like = 0
            for pattern in date_patterns:
                date_like += series.astype(str).str.match(pattern).sum()
            
            return {
                'percentage': float(min(100, date_like / len(series) * 100)),
                'count': int(min(len(series), date_like))
            }
        except:
            return {'percentage': 0, 'count': 0}
    
    def _check_format_consistency(self, col: str) -> Dict[str, Any]:
        """Check format consistency for a column"""
        try:
            sample_values = self.data[col].dropna().astype(str).head(1000)
            
            # Check length consistency
            lengths = sample_values.str.len()
            length_consistency = {
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'avg_length': float(lengths.mean()),
                'length_std': float(lengths.std()),
                'consistent_length': lengths.nunique() == 1
            }
            
            # Check case consistency
            has_upper = sample_values.str.contains(r'[A-Z]').any()
            has_lower = sample_values.str.contains(r'[a-z]').any()
            case_consistency = {
                'has_mixed_case': has_upper and has_lower,
                'all_upper': sample_values.str.isupper().all(),
                'all_lower': sample_values.str.islower().all()
            }
            
            return {
                'length_consistency': length_consistency,
                'case_consistency': case_consistency
            }
            
        except Exception as e:
            logger.error(f"Error checking format consistency for {col}: {str(e)}")
            return {}
    
    def _assess_validity(self) -> Dict[str, Any]:
        """Assess data validity"""
        try:
            validity = {
                'range_violations': {},
                'format_violations': {},
                'constraint_violations': {}
            }
            
            # Check numerical ranges
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                col_data = self.data[col].dropna()
                if len(col_data) > 0:
                    # Check for extreme values
                    q1, q3 = col_data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    extreme_lower = q1 - 3 * iqr
                    extreme_upper = q3 + 3 * iqr
                    
                    extreme_values = col_data[(col_data < extreme_lower) | (col_data > extreme_upper)]
                    validity['range_violations'][col] = {
                        'extreme_values_count': len(extreme_values),
                        'extreme_values_percentage': float(len(extreme_values) / len(col_data) * 100),
                        'negative_values': int((col_data < 0).sum()) if col.lower() in ['age', 'price', 'count', 'amount'] else None
                    }
            
            return validity
            
        except Exception as e:
            logger.error(f"Error assessing validity: {str(e)}")
            return {}
    
    def _assess_uniqueness(self) -> Dict[str, Any]:
        """Assess data uniqueness"""
        try:
            uniqueness = {
                'duplicate_analysis': {
                    'total_duplicates': int(self.data.duplicated().sum()),
                    'duplicate_percentage': float(self.data.duplicated().sum() / len(self.data) * 100),
                    'unique_rows': int(len(self.data) - self.data.duplicated().sum())
                },
                'column_uniqueness': {}
            }
            
            # Column-wise uniqueness
            for col in self.data.columns:
                unique_count = self.data[col].nunique()
                total_count = self.data[col].notna().sum()
                
                uniqueness['column_uniqueness'][col] = {
                    'unique_count': int(unique_count),
                    'uniqueness_ratio': float(unique_count / total_count if total_count > 0 else 0),
                    'is_potential_key': unique_count == total_count and total_count > 0
                }
            
            return uniqueness
            
        except Exception as e:
            logger.error(f"Error assessing uniqueness: {str(e)}")
            return {}
    
    def _assess_accuracy(self) -> Dict[str, Any]:
        """Assess data accuracy (basic heuristics)"""
        try:
            accuracy = {
                'suspicious_patterns': {},
                'outlier_summary': {},
                'data_type_mismatches': {}
            }
            
            # Look for suspicious patterns
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    sample_values = self.data[col].dropna().astype(str).head(1000)
                    
                    suspicious_patterns = {
                        'placeholder_values': sample_values.isin(['N/A', 'NULL', 'null', 'undefined', '???', 'TBD']).sum(),
                        'test_values': sample_values.str.contains(r'test|Test|TEST|dummy|Dummy', na=False).sum(),
                        'repeated_chars': sample_values.str.match(r'^(.)\1{4,}$').sum()
                    }
                    
                    accuracy['suspicious_patterns'][col] = {k: int(v) for k, v in suspicious_patterns.items()}
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error assessing accuracy: {str(e)}")
            return {}
    
    def _generate_quality_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        try:
            # Completeness recommendations
            if 'completeness' in quality_report:
                overall_completeness = quality_report['completeness'].get('overall_completeness', 100)
                if overall_completeness < 90:
                    recommendations.append(f"Data completeness is {overall_completeness:.1f}%. Consider data imputation strategies.")
                
                # Check for columns with high missing rates
                column_completeness = quality_report['completeness'].get('column_completeness', {})
                low_completeness_cols = [col for col, comp in column_completeness.items() if comp < 50]
                if low_completeness_cols:
                    recommendations.append(f"Consider removing columns with high missing rates: {', '.join(low_completeness_cols[:5])}")
            
            # Uniqueness recommendations
            if 'uniqueness' in quality_report:
                duplicate_percentage = quality_report['uniqueness']['duplicate_analysis'].get('duplicate_percentage', 0)
                if duplicate_percentage > 5:
                    recommendations.append(f"High duplicate rate ({duplicate_percentage:.1f}%). Review data collection process.")
            
            # Validity recommendations
            if 'validity' in quality_report and 'range_violations' in quality_report['validity']:
                extreme_cols = []
                for col, violations in quality_report['validity']['range_violations'].items():
                    if violations.get('extreme_values_percentage', 0) > 5:
                        extreme_cols.append(col)
                
                if extreme_cols:
                    recommendations.append(f"Review extreme values in: {', '.join(extreme_cols[:3])}")
            
            # General recommendations based on overall score
            overall_score = quality_report.get('overall_score', 100)
            if overall_score < 70:
                recommendations.append("Overall data quality is low. Consider comprehensive data cleaning.")
            elif overall_score < 85:
                recommendations.append("Data quality is moderate. Focus on major issues identified above.")
            
            if not recommendations:
                recommendations.append("Data quality appears good. Proceed with analysis.")
            
        except Exception as e:
            logger.error(f"Error generating quality recommendations: {str(e)}")
            recommendations.append("Unable to generate quality recommendations due to analysis error.")
        
        return recommendations
    
    def _analyze_distributions(self, columns: List[str]) -> Dict[str, Any]:
        """Analyze distributions of numerical columns"""
        try:
            numerical_cols = [col for col in columns if self.data[col].dtype in ['int64', 'int32', 'float64', 'float32']]
            distribution_results = {}
            
            for col in numerical_cols[:10]:  # Limit to 10 columns
                try:
                    col_data = self.data[col].dropna()
                    if len(col_data) < 10:
                        continue
                    
                    # Fit common distributions
                    distributions = ['norm', 'lognorm', 'expon', 'gamma', 'beta', 'uniform']
                    fitted_distributions = {}
                    
                    for dist_name in distributions:
                        try:
                            dist = getattr(stats, dist_name)
                            params = dist.fit(col_data)
                            
                            # Goodness of fit test
                            D, p_value = stats.kstest(col_data, lambda x: dist.cdf(x, *params))
                            
                            fitted_distributions[dist_name] = {
                                'parameters': [float(p) for p in params],
                                'ks_statistic': float(D),
                                'p_value': float(p_value),
                                'goodness_of_fit': float(p_value)
                            }
                            
                        except Exception as e:
                            fitted_distributions[dist_name] = {'error': str(e)}
                    
                    # Find best fitting distribution
                    valid_fits = {k: v for k, v in fitted_distributions.items() if 'error' not in v}
                    if valid_fits:
                        best_fit = max(valid_fits, key=lambda x: valid_fits[x]['goodness_of_fit'])
                        fitted_distributions['best_fit'] = best_fit
                    
                    distribution_results[col] = fitted_distributions
                    
                except Exception as e:
                    distribution_results[col] = {'error': str(e)}
            
            return distribution_results
            
        except Exception as e:
            logger.error(f"Error analyzing distributions: {str(e)}")
            return {}
    
    def _calculate_feature_importance(self, columns: List[str]) -> Dict[str, Any]:
        """Calculate feature importance using various methods"""
        try:
            numerical_cols = [col for col in columns if self.data[col].dtype in ['int64', 'int32', 'float64', 'float32']]
            
            if len(numerical_cols) < 2:
                return {'error': 'Insufficient numerical columns for feature importance'}
            
            importance_results = {}
            
            # Variance-based feature importance
            try:
                variances = self.data[numerical_cols].var()
                importance_results['variance_based'] = {
                    col: float(var) for col, var in variances.items()
                }
            except Exception as e:
                importance_results['variance_based'] = {'error': str(e)}
            
            # Correlation-based feature importance (using average absolute correlation)
            try:
                corr_matrix = self.data[numerical_cols].corr().abs()
                avg_correlations = corr_matrix.mean()
                importance_results['correlation_based'] = {
                    col: float(corr) for col, corr in avg_correlations.items()
                }
            except Exception as e:
                importance_results['correlation_based'] = {'error': str(e)}
            
            # PCA-based feature importance
            try:
                if len(numerical_cols) > 2:
                    pca_data = self.data[numerical_cols].dropna()
                    if len(pca_data) > 10:
                        pca = PCA(n_components=min(5, len(numerical_cols)))
                        pca.fit(StandardScaler().fit_transform(pca_data))
                        
                        # Feature importance based on loadings
                        feature_importance = np.abs(pca.components_).mean(axis=0)
                        importance_results['pca_based'] = {
                            col: float(imp) for col, imp in zip(numerical_cols, feature_importance)
                        }
            except Exception as e:
                importance_results['pca_based'] = {'error': str(e)}
            
            return importance_results
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def _generate_insights(self) -> List[str]:
        """Generate comprehensive AI-like insights about the data"""
        try:
            insights = []
            
            # Dataset size insights
            rows, cols = self.data.shape
            if rows > 100000:
                insights.append(f"Large dataset detected ({rows:,} rows). Consider sampling for exploratory analysis.")
            elif rows < 100:
                insights.append(f"Small dataset detected ({rows} rows). Results may have limited statistical power.")
            
            if cols > 50:
                insights.append(f"High-dimensional dataset ({cols} features). Consider dimensionality reduction techniques.")
            
            # Data quality insights
            quality_score = self.metadata.get('data_quality_score', 0)
            if quality_score < 70:
                insights.append(f"Low data quality score ({quality_score:.1f}). Prioritize data cleaning.")
            elif quality_score > 90:
                insights.append(f"High data quality score ({quality_score:.1f}). Data appears clean and ready for analysis.")
            
            # Missing data insights
            total_missing = self.data.isnull().sum().sum()
            missing_percentage = total_missing / (rows * cols) * 100
            if missing_percentage > 20:
                insights.append(f"High missing data rate ({missing_percentage:.1f}%). Investigate missing data patterns.")
            elif missing_percentage < 1:
                insights.append("Very low missing data rate. Excellent data completeness.")
            
            # Duplicate insights
            duplicates = self.data.duplicated().sum()
            if duplicates > 0:
                duplicate_pct = duplicates / rows * 100
                if duplicate_pct > 10:
                    insights.append(f"High duplication rate ({duplicate_pct:.1f}%). Review data collection process.")
                else:
                    insights.append(f"Moderate duplication detected ({duplicates} rows). Consider deduplication.")
            
            # Column type insights
            column_types = self.metadata.get('column_types', {})
            numerical_count = len(column_types.get('numerical', []))
            categorical_count = len(column_types.get('categorical', []))
            
            if numerical_count / cols > 0.8:
                insights.append("Dataset is predominantly numerical. Well-suited for statistical modeling.")
            elif categorical_count / cols > 0.8:
                insights.append("Dataset is predominantly categorical. Consider encoding strategies for ML models.")
            else:
                insights.append("Mixed data types detected. Good for comprehensive analysis.")
            
            # High cardinality insights
            high_cardinality = column_types.get('high_cardinality', [])
            if high_cardinality:
                insights.append(f"High cardinality columns detected: {', '.join(high_cardinality[:3])}. May need special handling.")
            
            # Constant columns
            constant_cols = column_types.get('constant', [])
            if constant_cols:
                insights.append(f"Constant columns found: {', '.join(constant_cols)}. Consider removing for analysis.")
            
            # ID-like columns
            id_cols = column_types.get('id_like', [])
            if id_cols:
                insights.append(f"Potential identifier columns: {', '.join(id_cols)}. Exclude from predictive modeling.")
            
            # Datetime insights
            datetime_cols = column_types.get('datetime', [])
            if datetime_cols:
                insights.append(f"Temporal data available ({len(datetime_cols)} columns). Consider time series analysis.")
            
            # Memory usage insights
            memory_mb = self.metadata.get('memory_usage', 0) / (1024 * 1024)
            if memory_mb > 100:
                insights.append(f"Large memory footprint ({memory_mb:.1f} MB). Consider optimization strategies.")
            
            # Numerical data insights
            if numerical_count > 0:
                # Check for skewed distributions
                skewness = self.metadata.get('skewness', {})
                highly_skewed = [col for col, skew in skewness.items() if abs(skew) > 2]
                if highly_skewed:
                    insights.append(f"Highly skewed distributions detected: {', '.join(highly_skewed[:3])}. Consider transformations.")
                
                # Check for potential outliers
                numerical_stats = self.metadata.get('numerical_stats', {})
                potential_outlier_cols = []
                for col, stats_dict in numerical_stats.items():
                    if 'std' in stats_dict and 'mean' in stats_dict:
                        cv = abs(stats_dict['std'] / stats_dict['mean']) if stats_dict['mean'] != 0 else 0
                        if cv > 2:  # High coefficient of variation
                            potential_outlier_cols.append(col)
                
                if potential_outlier_cols:
                    insights.append(f"High variability in: {', '.join(potential_outlier_cols[:3])}. Check for outliers.")
            
            # Categorical data insights
            if categorical_count > 0:
                cat_stats = self.metadata.get('categorical_stats', {})
                low_entropy_cols = []
                for col, stats_dict in cat_stats.items():
                    if isinstance(stats_dict, dict) and 'cardinality_ratio' in stats_dict:
                        if stats_dict['cardinality_ratio'] < 0.05:
                            low_entropy_cols.append(col)
                
                if low_entropy_cols:
                    insights.append(f"Low diversity categorical columns: {', '.join(low_entropy_cols[:3])}. May have limited predictive power.")
            
            # Correlation insights (if correlation analysis was performed)
            try:
                numerical_cols = self.data.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 1:
                    corr_matrix = self.data[numerical_cols].corr()
                    high_corr_pairs = 0
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.8:
                                high_corr_pairs += 1
                    
                    if high_corr_pairs > 0:
                        insights.append(f"Found {high_corr_pairs} highly correlated feature pairs. Consider feature selection.")
            except:
                pass
            
            # General recommendations
            if len(insights) == 0:
                insights.append("Data appears well-structured. Ready for advanced analytics.")
            
            # Add methodology recommendations
            if numerical_count > categorical_count:
                insights.append("Numerical data dominance suggests regression or clustering approaches may be effective.")
            elif categorical_count > numerical_count:
                insights.append("Categorical data dominance suggests classification or association analysis may be effective.")
            
            if rows > 1000 and cols > 10:
                insights.append("Dataset size supports machine learning approaches. Consider train/validation/test splits.")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return ["Unable to generate insights due to analysis error."]

class StatisticalTester:
    """Comprehensive statistical testing class with 40+ tests"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.test_results = []
    
    def run_comprehensive_tests(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive statistical tests based on configuration"""
        try:
            results = {}
            
            # Normality tests
            if test_config.get('normality_tests'):
                results['normality'] = self.run_normality_tests(
                    test_config.get('variables', []),
                    test_config.get('alpha', 0.05)
                )
            
            # Hypothesis tests
            if test_config.get('hypothesis_tests'):
                results['hypothesis'] = self.run_hypothesis_tests(
                    test_config.get('test_type', 'two_sample_t'),
                    test_config.get('var1'),
                    test_config.get('var2'),
                    test_config.get('alpha', 0.05),
                    test_config.get('alternative', 'two-sided')
                )
            
            # Correlation tests
            if test_config.get('correlation_tests'):
                results['correlation'] = self.run_correlation_tests(
                    test_config.get('var1'),
                    test_config.get('var2'),
                    test_config.get('alpha', 0.05)
                )
            
            # Independence tests
            if test_config.get('independence_tests'):
                results['independence'] = self.run_independence_tests(
                    test_config.get('var1'),
                    test_config.get('var2'),
                    test_config.get('alpha', 0.05)
                )
            
            # Variance tests
            if test_config.get('variance_tests'):
                results['variance'] = self.run_variance_tests(
                    test_config.get('variables', []),
                    test_config.get('alpha', 0.05)
                )
            
            # Non-parametric tests
            if test_config.get('nonparametric_tests'):
                results['nonparametric'] = self.run_nonparametric_tests(
                    test_config.get('test_type'),
                    test_config.get('variables', []),
                    test_config.get('alpha', 0.05)
                )
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            logger.error(f"Error running comprehensive tests: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_normality_tests(self, columns: List[str], alpha: float = 0.05) -> Dict[str, Any]:
        """Run comprehensive normality tests"""
        try:
            results = {}
            
            for col in columns:
                if col not in self.data.columns:
                    continue
                    
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    continue
                    
                clean_data = self.data[col].dropna()
                if len(clean_data) < 3:
                    continue
                
                # Sample for large datasets
                if len(clean_data) > 5000:
                    clean_data = clean_data.sample(5000, random_state=42)
                
                col_results = {}
                
                # Shapiro-Wilk test
                try:
                    if len(clean_data) <= 5000:
                        stat, p_value = stats.shapiro(clean_data)
                        col_results['shapiro_wilk'] = self._format_test_result(
                            'Shapiro-Wilk Test',
                            stat, p_value, alpha,
                            'Data follows normal distribution',
                            'Data does not follow normal distribution'
                        )
                except Exception as e:
                    col_results['shapiro_wilk'] = {'error': str(e)}
                
                # D'Agostino-Pearson test
                try:
                    if len(clean_data) >= 8:
                        stat, p_value = stats.normaltest(clean_data)
                        col_results['dagostino_pearson'] = self._format_test_result(
                            "D'Agostino-Pearson Test",
                            stat, p_value, alpha,
                            'Data follows normal distribution',
                            'Data does not follow normal distribution'
                        )
                except Exception as e:
                    col_results['dagostino_pearson'] = {'error': str(e)}
                
                # Kolmogorov-Smirnov test
                try:
                    # Standardize data for KS test
                    standardized = (clean_data - clean_data.mean()) / clean_data.std()
                    stat, p_value = stats.kstest(standardized, 'norm')
                    col_results['kolmogorov_smirnov'] = self._format_test_result(
                        'Kolmogorov-Smirnov Test',
                        stat, p_value, alpha,
                        'Data follows normal distribution',
                        'Data does not follow normal distribution'
                    )
                except Exception as e:
                    col_results['kolmogorov_smirnov'] = {'error': str(e)}
                
                # Anderson-Darling test
                try:
                    result = stats.anderson(clean_data, dist='norm')
                    # Use 5% significance level
                    critical_value = result.critical_values[2]  # 5% level
                    is_normal = result.statistic < critical_value
                    
                    col_results['anderson_darling'] = {
                        'test_name': 'Anderson-Darling Test',
                        'statistic': float(result.statistic),
                        'critical_value': float(critical_value),
                        'significant': not is_normal,
                        'interpretation': f"Data {'follows' if is_normal else 'does not follow'} normal distribution"
                    }
                except Exception as e:
                    col_results['anderson_darling'] = {'error': str(e)}
                
                # Jarque-Bera test
                try:
                    stat, p_value = stats.jarque_bera(clean_data)
                    col_results['jarque_bera'] = self._format_test_result(
                        'Jarque-Bera Test',
                        stat, p_value, alpha,
                        'Data follows normal distribution',
                        'Data does not follow normal distribution'
                    )
                except Exception as e:
                    col_results['jarque_bera'] = {'error': str(e)}
                
                # Lilliefors test (if available)
                try:
                    # Simplified Lilliefors test using KS test on standardized data
                    standardized = (clean_data - clean_data.mean()) / clean_data.std()
                    stat, p_value = stats.kstest(standardized, 'norm')
                    col_results['lilliefors'] = self._format_test_result(
                        'Lilliefors Test',
                        stat, p_value, alpha,
                        'Data follows normal distribution',
                        'Data does not follow normal distribution'
                    )
                except Exception as e:
                    col_results['lilliefors'] = {'error': str(e)}
                
                results[col] = col_results
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            logger.error(f"Error in normality tests: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _format_test_result(self, test_name: str, statistic: float, p_value: float, 
                          alpha: float, null_hyp: str, alt_hyp: str) -> Dict[str, Any]:
        """Format statistical test result"""
        return {
            'test_name': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'alpha': alpha,
            'null_hypothesis': null_hyp,
            'alternative_hypothesis': alt_hyp,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis (p = {p_value:.4f})"
        }
    
    def run_hypothesis_tests(self, test_type: str, var1: str, var2: str = None, 
                           alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """Run various hypothesis tests"""
        try:
            if test_type == 'one_sample_t':
                return self._one_sample_t_test(var1, alpha, alternative)
            elif test_type == 'two_sample_t':
                return self._two_sample_t_test(var1, var2, alpha, alternative)
            elif test_type == 'paired_t':
                return self._paired_t_test(var1, var2, alpha, alternative)
            elif test_type == 'welch_t':
                return self._welch_t_test(var1, var2, alpha, alternative)
            elif test_type == 'one_way_anova':
                return self._one_way_anova(var1, var2, alpha)
            elif test_type == 'two_way_anova':
                return self._two_way_anova(var1, var2, alpha)
            else:
                return {'success': False, 'error': f'Unknown test type: {test_type}'}
                
        except Exception as e:
            logger.error(f"Error in hypothesis tests: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _one_sample_t_test(self, variable: str, alpha: float, alternative: str, mu: float = 0) -> Dict[str, Any]:
        """Perform one-sample t-test"""
        try:
            data = self.data[variable].dropna()
            if len(data) < 2:
                return {'success': False, 'error': 'Insufficient data points'}
            
            # Map alternative hypothesis
            alt_map = {'two-sided': 'two-sided', 'greater': 'greater', 'less': 'less'}
            alt = alt_map.get(alternative, 'two-sided')
            
            statistic, p_value = stats.ttest_1samp(data, mu, alternative=alt)
            
            result = self._format_test_result(
                'One-Sample t-Test',
                statistic, p_value, alpha,
                f'Population mean equals {mu}',
                f'Population mean {alternative} {mu}'
            )
            
            result.update({
                'degrees_freedom': len(data) - 1,
                'sample_mean': float(data.mean()),
                'sample_std': float(data.std()),
                'sample_size': len(data),
                'effect_size': float((data.mean() - mu) / data.std()) if data.std() > 0 else 0
            })
            
            return {'success': True, 'results': result}
            
        except Exception as e:
            logger.error(f"Error in one-sample t-test: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _two_sample_t_test(self, var1: str, var2: str, alpha: float, alternative: str) -> Dict[str, Any]:
        """Perform two-sample t-test"""
        try:
            data1 = self.data[var1].dropna()
            data2 = self.data[var2].dropna()
            
            if len(data1) < 2 or len(data2) < 2:
                return {'success': False, 'error': 'Insufficient data points'}
            
            alt_map = {'two-sided': 'two-sided', 'greater': 'greater', 'less': 'less'}
            alt = alt_map.get(alternative, 'two-sided')
            
            statistic, p_value = stats.ttest_ind(data1, data2, alternative=alt)
            
            result = self._format_test_result(
                'Two-Sample t-Test',
                statistic, p_value, alpha,
                f'Mean of {var1} equals mean of {var2}',
                f'Mean of {var1} {alternative} mean of {var2}'
            )
            
            # Calculate pooled standard deviation
            pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + (len(data2) - 1) * data2.var()) / 
                                (len(data1) + len(data2) - 2))
            
            result.update({
                'degrees_freedom': len(data1) + len(data2) - 2,
                'sample_stats': {
                    var1: {'mean': float(data1.mean()), 'std': float(data1.std()), 'n': len(data1)},
                    var2: {'mean': float(data2.mean()), 'std': float(data2.std()), 'n': len(data2)}
                },
                'pooled_std': float(pooled_std),
                'cohens_d': float((data1.mean() - data2.mean()) / pooled_std) if pooled_std > 0 else 0
            })
            
            return {'success': True, 'results': result}
            
        except Exception as e:
            logger.error(f"Error in two-sample t-test: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _paired_t_test(self, var1: str, var2: str, alpha: float, alternative: str) -> Dict[str, Any]:
        """Perform paired t-test"""
        try:
            paired_data = self.data[[var1, var2]].dropna()
            if len(paired_data) < 2:
                return {'success': False, 'error': 'Insufficient paired data points'}
            
            data1 = paired_data[var1]
            data2 = paired_data[var2]
            differences = data1 - data2
            
            alt_map = {'two-sided': 'two-sided', 'greater': 'greater', 'less': 'less'}
            alt = alt_map.get(alternative, 'two-sided')
            
            statistic, p_value = stats.ttest_rel(data1, data2, alternative=alt)
            
            result = self._format_test_result(
                'Paired t-Test',
                statistic, p_value, alpha,
                f'Mean difference between {var1} and {var2} equals 0',
                f'Mean difference between {var1} and {var2} {alternative} 0'
            )
            
            result.update({
                'degrees_freedom': len(paired_data) - 1,
                'difference_stats': {
                    'mean_difference': float(differences.mean()),
                    'std_difference': float(differences.std()),
                    'n_pairs': len(paired_data)
                },
                'effect_size': float(differences.mean() / differences.std()) if differences.std() > 0 else 0
            })
            
            return {'success': True, 'results': result}
            
        except Exception as e:
            logger.error(f"Error in paired t-test: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _welch_t_test(self, var1: str, var2: str, alpha: float, alternative: str) -> Dict[str, Any]:
        """Perform Welch's t-test (unequal variances)"""
        try:
            data1 = self.data[var1].dropna()
            data2 = self.data[var2].dropna()
            
            if len(data1) < 2 or len(data2) < 2:
                return {'success': False, 'error': 'Insufficient data points'}
            
            alt_map = {'two-sided': 'two-sided', 'greater': 'greater', 'less': 'less'}
            alt = alt_map.get(alternative, 'two-sided')
            
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False, alternative=alt)
            
            # Calculate Welch's degrees of freedom
            s1_sq, s2_sq = data1.var(ddof=1), data2.var(ddof=1)
            n1, n2 = len(data1), len(data2)
            
            numerator = (s1_sq/n1 + s2_sq/n2)**2
            denominator = (s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1)
            welch_df = numerator / denominator if denominator > 0 else n1 + n2 - 2
            
            result = self._format_test_result(
                "Welch's t-Test",
                statistic, p_value, alpha,
                f'Mean of {var1} equals mean of {var2}',
                f'Mean of {var1} {alternative} mean of {var2}'
            )
            
            result.update({
                'degrees_freedom': float(welch_df),
                'sample_stats': {
                    var1: {'mean': float(data1.mean()), 'std': float(data1.std()), 'n': len(data1)},
                    var2: {'mean': float(data2.mean()), 'std': float(data2.std()), 'n': len(data2)}
                },
                'variance_ratio': float(s1_sq / s2_sq) if s2_sq > 0 else float('inf')
            })
            
            return {'success': True, 'results': result}
            
        except Exception as e:
            logger.error(f"Error in Welch's t-test: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _one_way_anova(self, dependent_var: str, grouping_var: str, alpha: float) -> Dict[str, Any]:
        """Perform one-way ANOVA"""
        try:
            # Get groups
            groups_data = []
            group_names = []
            group_stats = {}
            
            for name, group in self.data.groupby(grouping_var):
                group_data = group[dependent_var].dropna()
                if len(group_data) > 0:
                    groups_data.append(group_data)
                    group_names.append(str(name))
                    group_stats[str(name)] = {
                        'mean': float(group_data.mean()),
                        'std': float(group_data.std()),
                        'n': len(group_data),
                        'min': float(group_data.min()),
                        'max': float(group_data.max())
                    }
            
            if len(groups_data) < 2:
                return {'success': False, 'error': 'Need at least 2 groups for ANOVA'}
            
            # Perform ANOVA
            statistic, p_value = stats.f_oneway(*groups_data)
            
            # Calculate degrees of freedom
            df_between = len(groups_data) - 1
            df_within = sum(len(group) for group in groups_data) - len(groups_data)
            
            # Calculate effect size (eta-squared)
            ss_total = sum((group - self.data[dependent_var].mean())**2 for group in groups_data for val in group)
            ss_between = sum(len(group) * (group.mean() - self.data[dependent_var].mean())**2 for group in groups_data)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            result = self._format_test_result(
                'One-Way ANOVA',
                statistic, p_value, alpha,
                'All group means are equal',
                'At least one group mean is different'
            )
            
            result.update({
                'f_statistic': float(statistic),
                'df_between': df_between,
                'df_within': df_within,
                'eta_squared': float(eta_squared),
                'group_statistics': group_stats,
                'number_of_groups': len(groups_data)
            })
            
            return {'success': True, 'results': result}
            
        except Exception as e:
            logger.error(f"Error in one-way ANOVA: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _two_way_anova(self, dependent_var: str, factor1: str, factor2: str, alpha: float) -> Dict[str, Any]:
        """Perform two-way ANOVA (simplified)"""
        try:
            # This is a simplified implementation
            # For full two-way ANOVA, would typically use statsmodels
            
            # Get data with both factors
            data_subset = self.data[[dependent_var, factor1, factor2]].dropna()
            
            if len(data_subset) < 10:
                return {'success': False, 'error': 'Insufficient data for two-way ANOVA'}
            
            # Perform separate one-way ANOVAs for each factor
            groups_f1 = [group[dependent_var] for name, group in data_subset.groupby(factor1) if len(group) > 0]
            groups_f2 = [group[dependent_var] for name, group in data_subset.groupby(factor2) if len(group) > 0]
            
            if len(groups_f1) < 2 or len(groups_f2) < 2:
                return {'success': False, 'error': 'Insufficient groups for two-way ANOVA'}
            
            f_stat1, p_val1 = stats.f_oneway(*groups_f1)
            f_stat2, p_val2 = stats.f_oneway(*groups_f2)
            
            result = {
                'test_name': 'Two-Way ANOVA (Simplified)',
                'factor1_results': {
                    'factor': factor1,
                    'f_statistic': float(f_stat1),
                    'p_value': float(p_val1),
                    'significant': p_val1 < alpha
                },
                'factor2_results': {
                    'factor': factor2,
                    'f_statistic': float(f_stat2),
                    'p_value': float(p_val2),
                    'significant': p_val2 < alpha
                },
                'interpretation': f"Factor {factor1}: {'significant' if p_val1 < alpha else 'not significant'}, "
                               f"Factor {factor2}: {'significant' if p_val2 < alpha else 'not significant'}"
            }
            
            return {'success': True, 'results': result}
            
        except Exception as e:
            logger.error(f"Error in two-way ANOVA: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_correlation_tests(self, var1: str, var2: str, alpha: float = 0.05) -> Dict[str, Any]:
        """Run various correlation tests"""
        try:
            # Get paired data
            paired_data = self.data[[var1, var2]].dropna()
            
            if len(paired_data) < 3:
                return {'success': False, 'error': 'Insufficient paired data points'}
            
            data1 = paired_data[var1]
            data2 = paired_data[var2]
            
            results = {}
            
            # Pearson correlation
            try:
                r, p_value = stats.pearsonr(data1, data2)
                results['pearson'] = {
                    'test_name': 'Pearson Correlation',
                    'correlation': float(r),
                    'p_value': float(p_value),
                    'significant': p_value < alpha,
                    'interpretation': self._interpret_correlation_result(r, p_value, alpha, 'Pearson'),
                    'confidence_interval': self._correlation_confidence_interval(r, len(paired_data))
                }
            except Exception as e:
                results['pearson'] = {'error': str(e)}
            
            # Spearman correlation
            try:
                rho, p_value = stats.spearmanr(data1, data2)
                results['spearman'] = {
                    'test_name': 'Spearman Correlation',
                    'correlation': float(rho),
                    'p_value': float(p_value),
                    'significant': p_value < alpha,
                    'interpretation': self._interpret_correlation_result(rho, p_value, alpha, 'Spearman')
                }
            except Exception as e:
                results['spearman'] = {'error': str(e)}
            
            # Kendall's tau
            try:
                tau, p_value = stats.kendalltau(data1, data2)
                results['kendall'] = {
                    'test_name': "Kendall's Tau",
                    'correlation': float(tau),
                    'p_value': float(p_value),
                    'significant': p_value < alpha,
                    'interpretation': self._interpret_correlation_result(tau, p_value, alpha, "Kendall's tau")
                }
            except Exception as e:
                results['kendall'] = {'error': str(e)}
            
            # Point-biserial correlation (if one variable is binary)
            try:
                if data1.nunique() == 2 or data2.nunique() == 2:
                    # Convert binary to 0/1
                    binary_var = data1 if data1.nunique() == 2 else data2
                    continuous_var = data2 if data1.nunique() == 2 else data1
                    
                    binary_encoded = (binary_var == binary_var.unique()[1]).astype(int)
                    r_pb, p_value = stats.pearsonr(binary_encoded, continuous_var)
                    
                    results['point_biserial'] = {
                        'test_name': 'Point-Biserial Correlation',
                        'correlation': float(r_pb),
                        'p_value': float(p_value),
                        'significant': p_value < alpha,
                        'interpretation': self._interpret_correlation_result(r_pb, p_value, alpha, 'Point-biserial')
                    }
            except Exception as e:
                results['point_biserial'] = {'error': str(e)}
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            logger.error(f"Error in correlation tests: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _interpret_correlation_result(self, correlation: float, p_value: float, alpha: float, test_name: str) -> str:
        """Interpret correlation test results"""
        strength = self._get_correlation_strength(abs(correlation))
        direction = "positive" if correlation > 0 else "negative" if correlation < 0 else "no"
        significance = "significant" if p_value < alpha else "not significant"
        
        interpretation = f"{test_name} correlation shows {strength} {direction} association "
        interpretation += f"(r = {correlation:.3f}, p = {p_value:.4f}). "
        interpretation += f"Result is {significance} at α = {alpha}."
        
        return interpretation
    
    def _get_correlation_strength(self, abs_correlation: float) -> str:
        """Interpret correlation strength"""
        if abs_correlation >= 0.9:
            return "very strong"
        elif abs_correlation >= 0.7:
            return "strong"
        elif abs_correlation >= 0.5:
            return "moderate"
        elif abs_correlation >= 0.3:
            return "weak"
        else:
            return "very weak"
    
    def _correlation_confidence_interval(self, r: float, n: int, confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for correlation coefficient"""
        try:
            # Fisher's z-transformation
            z = 0.5 * np.log((1 + r) / (1 - r))
            
            # Standard error
            se = 1 / np.sqrt(n - 3)
            
            # Critical value
            z_critical = stats.norm.ppf((1 + confidence) / 2)
            
            # Confidence interval for z
            z_lower = z - z_critical * se
            z_upper = z + z_critical * se
            
            # Transform back to correlation scale
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            return {
                'lower': float(r_lower),
                'upper': float(r_upper),
                'confidence_level': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error calculating correlation CI: {str(e)}")
            return {'error': str(e)}
    
    def run_independence_tests(self, var1: str, var2: str, alpha: float = 0.05) -> Dict[str, Any]:
        """Run tests for independence between categorical variables"""
        try:
            # Create contingency table
            contingency_table = pd.crosstab(self.data[var1], self.data[var2])
            
            if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                return {'success': False, 'error': 'Insufficient data for independence tests'}
            
            results = {}
            
            # Chi-square test
            try:
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                # Calculate effect sizes
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                
                # Phi coefficient (for 2x2 tables)
                phi = np.sqrt(chi2_stat / n) if contingency_table.shape == (2, 2) else None
                
                results['chi_square'] = {
                    'test_name': 'Chi-Square Test of Independence',
                    'chi2_statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'degrees_freedom': int(dof),
                    'significant': p_value < alpha,
                    'cramers_v': float(cramers_v),
                    'effect_size_interpretation': self._interpret_cramers_v(cramers_v),
                    'interpretation': f"Variables are {'dependent' if p_value < alpha else 'independent'} "
                                    f"(χ² = {chi2_stat:.3f}, p = {p_value:.4f})"
                }
                
                if phi is not None:
                    results['chi_square']['phi_coefficient'] = float(phi)
                
            except Exception as e:
                results['chi_square'] = {'error': str(e)}
            
            # Fisher's exact test (for 2x2 tables)
            if contingency_table.shape == (2, 2):
                try:
                    oddsratio, p_value = stats.fisher_exact(contingency_table)
                    results['fisher_exact'] = {
                        'test_name': "Fisher's Exact Test",
                        'odds_ratio': float(oddsratio),
                        'p_value': float(p_value),
                        'significant': p_value < alpha,
                        'interpretation': f"Association is {'significant' if p_value < alpha else 'not significant'} "
                                       f"(OR = {oddsratio:.3f}, p = {p_value:.4f})"
                    }
                except Exception as e:
                    results['fisher_exact'] = {'error': str(e)}
            
            # G-test (likelihood ratio)
            try:
                observed = contingency_table.values
                row_totals = observed.sum(axis=1)
                col_totals = observed.sum(axis=0)
                total = observed.sum()
                
                expected = np.outer(row_totals, col_totals) / total
                
                # Calculate G statistic
                mask = observed > 0
                g_stat = 2 * np.sum(observed[mask] * np.log(observed[mask] / expected[mask]))
                
                p_value = 1 - stats.chi2.cdf(g_stat, dof)
                
                results['g_test'] = {
                    'test_name': 'G-Test (Likelihood Ratio)',
                    'g_statistic': float(g_stat),
                    'p_value': float(p_value),
                    'degrees_freedom': int(dof),
                    'significant': p_value < alpha,
                    'interpretation': f"Variables are {'dependent' if p_value < alpha else 'independent'} "
                                   f"(G = {g_stat:.3f}, p = {p_value:.4f})"
                }
                
            except Exception as e:
                results['g_test'] = {'error': str(e)}
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            logger.error(f"Error in independence tests: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpret Cramér's V effect size"""
        if cramers_v >= 0.5:
            return "large effect"
        elif cramers_v >= 0.3:
            return "medium effect"
        elif cramers_v >= 0.1:
            return "small effect"
        else:
            return "negligible effect"
    
    def run_variance_tests(self, groups: List[str], alpha: float = 0.05) -> Dict[str, Any]:
        """Run tests for equality of variances"""
        try:
            # Get data for each group
            group_data = []
            for group in groups:
                if group in self.data.columns:
                    data = self.data[group].dropna()
                    if len(data) > 1:
                        group_data.append(data)
            
            if len(group_data) < 2:
                return {'success': False, 'error': 'Need at least 2 groups with sufficient data'}
            
            results = {}
            
            # Levene's test
            try:
                statistic, p_value = stats.levene(*group_data)
                results['levene'] = self._format_test_result(
                    "Levene's Test",
                    statistic, p_value, alpha,
                    "Variances are equal",
                    "Variances are unequal"
                )
            except Exception as e:
                results['levene'] = {'error': str(e)}
            
            # Bartlett's test
            try:
                statistic, p_value = stats.bartlett(*group_data)
                results['bartlett'] = self._format_test_result(
                    "Bartlett's Test",
                    statistic, p_value, alpha,
                    "Variances are equal",
                    "Variances are unequal"
                )
            except Exception as e:
                results['bartlett'] = {'error': str(e)}
            
            # Fligner-Killeen test
            try:
                statistic, p_value = stats.fligner(*group_data)
                results['fligner'] = self._format_test_result(
                    'Fligner-Killeen Test',
                    statistic, p_value, alpha,
                    "Variances are equal",
                    "Variances are unequal"
                )
            except Exception as e:
                results['fligner'] = {'error': str(e)}
            
            # F-test for two groups
            if len(group_data) == 2:
                try:
                    var1, var2 = group_data[0].var(), group_data[1].var()
                    f_stat = var1 / var2 if var2 > 0 else float('inf')
                    df1, df2 = len(group_data[0]) - 1, len(group_data[1]) - 1
                    
                    # Two-tailed test
                    p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
                    
                    results['f_test'] = self._format_test_result(
                        'F-Test for Equal Variances',
                        f_stat, p_value, alpha,
                        "Variances are equal",
                        "Variances are unequal"
                    )
                    results['f_test']['degrees_freedom'] = {'numerator': df1, 'denominator': df2}
                    
                except Exception as e:
                    results['f_test'] = {'error': str(e)}
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            logger.error(f"Error in variance tests: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_nonparametric_tests(self, test_type: str, groups: List[str], alpha: float = 0.05) -> Dict[str, Any]:
        """Run various non-parametric tests"""
        try:
            if test_type == 'mann_whitney':
                return self._mann_whitney_test(groups[0], groups[1], alpha)
            elif test_type == 'wilcoxon_signed_rank':
                return self._wilcoxon_signed_rank_test(groups[0], groups[1], alpha)
            elif test_type == 'kruskal_wallis':
                return self._kruskal_wallis_test(groups, alpha)
            elif test_type == 'friedman':
                return self._friedman_test(groups, alpha)
            elif test_type == 'mood_median':
                return self._mood_median_test(groups, alpha)
            else:
                return {'success': False, 'error': f'Unknown non-parametric test: {test_type}'}
                
        except Exception as e:
            logger.error(f"Error in non-parametric tests: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _mann_whitney_test(self, var1: str, var2: str, alpha: float) -> Dict[str, Any]:
        """Perform Mann-Whitney U test"""
        try:
            data1 = self.data[var1].dropna()
            data2 = self.data[var2].dropna()
            
            if len(data1) < 1 or len(data2) < 1:
                return {'success': False, 'error': 'Insufficient data points'}
            
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(data1), len(data2)
            r = 1 - (2 * statistic) / (n1 * n2)
            
            result = self._format_test_result(
                'Mann-Whitney U Test',
                statistic, p_value, alpha,
                f'Distributions of {var1} and {var2} are identical',
                f'Distributions of {var1} and {var2} are different'
            )
            
            result.update({
                'u_statistic': float(statistic),
                'sample_sizes': {'group1': n1, 'group2': n2},
                'effect_size_r': float(r),
                'median_difference': float(data1.median() - data2.median())
            })
            
            return {'success': True, 'results': result}
            
        except Exception as e:
            logger.error(f"Error in Mann-Whitney test: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _wilcoxon_signed_rank_test(self, var1: str, var2: str, alpha: float) -> Dict[str, Any]:
        """Perform Wilcoxon signed-rank test"""
        try:
            paired_data = self.data[[var1, var2]].dropna()
            
            if len(paired_data) < 2:
                return {'success': False, 'error': 'Insufficient paired data points'}
            
            data1 = paired_data[var1]
            data2 = paired_data[var2]
            
            statistic, p_value = stats.wilcoxon(data1, data2)
            
            differences = data1 - data2
            result = self._format_test_result(
                'Wilcoxon Signed-Rank Test',
                statistic, p_value, alpha,
                f'Median difference between {var1} and {var2} is zero',
                f'Median difference between {var1} and {var2} is not zero'
            )
            
            result.update({
                'w_statistic': float(statistic),
                'sample_size': len(paired_data),
                'median_difference': float(differences.median()),
                'mean_difference': float(differences.mean())
            })
            
            return {'success': True, 'results': result}
            
        except Exception as e:
            logger.error(f"Error in Wilcoxon signed-rank test: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _kruskal_wallis_test(self, groups: List[str], alpha: float) -> Dict[str, Any]:
        """Perform Kruskal-Wallis test"""
        try:
            group_data = []
            group_stats = {}
            
            for group in groups:
                if group in self.data.columns:
                    data = self.data[group].dropna()
                    if len(data) > 0:
                        group_data.append(data)
                        group_stats[group] = {
                            'median': float(data.median()),
                            'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
                            'n': len(data),
                            'mean_rank': 0  # Will be calculated after test
                        }
            
            if len(group_data) < 2:
                return {'success': False, 'error': 'Need at least 2 groups for Kruskal-Wallis test'}
            
            statistic, p_value = stats.kruskal(*group_data)
            
            # Calculate mean ranks
            all_data = np.concatenate(group_data)
            ranks = stats.rankdata(all_data)
            
            start_idx = 0
            for i, (group, data) in enumerate(zip(groups, group_data)):
                end_idx = start_idx + len(data)
                group_ranks = ranks[start_idx:end_idx]
                if group in group_stats:
                    group_stats[group]['mean_rank'] = float(np.mean(group_ranks))
                start_idx = end_idx
            
            result = self._format_test_result(
                'Kruskal-Wallis Test',
                statistic, p_value, alpha,
                'All groups have identical distributions',
                'At least one group has a different distribution'
            )
            
            result.update({
                'h_statistic': float(statistic),
                'degrees_freedom': len(group_data) - 1,
                'group_statistics': group_stats,
                'number_of_groups': len(group_data)
            })
            
            return {'success': True, 'results': result}
            
        except Exception as e:
            logger.error(f"Error in Kruskal-Wallis test: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _friedman_test(self, groups: List[str], alpha: float) -> Dict[str, Any]:
        """Perform Friedman test"""
        try:
            # Get complete cases for all groups
            complete_data = self.data[groups].dropna()
            
            if len(complete_data) < 2:
                return {'success': False, 'error': 'Insufficient complete cases for Friedman test'}
            
            group_arrays = [complete_data[group].values for group in groups]
            statistic, p_value = stats.friedmanchisquare(*group_arrays)
            
            # Calculate group statistics
            group_stats = {}
            for group in groups:
                data = complete_data[group]
                group_stats[group] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'n': len(data)
                }
            
            result = self._format_test_result(
                'Friedman Test',
                statistic, p_value, alpha,
                'All repeated measures have identical effects',
                'At least one repeated measure has a different effect'
            )
            
            result.update({
                'chi2_statistic': float(statistic),
                'degrees_freedom': len(groups) - 1,
                'group_statistics': group_stats,
                'number_of_subjects': len(complete_data)
            })
            
            return {'success': True, 'results': result}
            
        except Exception as e:
            logger.error(f"Error in Friedman test: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _mood_median_test(self, groups: List[str], alpha: float) -> Dict[str, Any]:
        """Perform Mood's median test"""
        try:
            group_data = []
            for group in groups:
                if group in self.data.columns:
                    data = self.data[group].dropna()
                    if len(data) > 0:
                        group_data.append(data)
            
            if len(group_data) < 2:
                return {'success': False, 'error': 'Need at least 2 groups for Mood median test'}
            
            # Calculate grand median
            all_data = np.concatenate(group_data)
            grand_median = np.median(all_data)
            
            # Create contingency table
            above_counts = []
            below_counts = []
            
            for data in group_data:
                above = np.sum(data > grand_median)
                below = np.sum(data <= grand_median)
                above_counts.append(above)
                below_counts.append(below)
            
            contingency = np.array([above_counts, below_counts])
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            result = self._format_test_result(
                "Mood's Median Test",
                chi2_stat, p_value, alpha,
                'All groups have the same median',
                'At least one group has a different median'
            )
            
            result.update({
                'grand_median': float(grand_median),
                'contingency_table': {
                    'above_median': above_counts,
                    'below_median': below_counts
                },
                'degrees_freedom': dof
            })
            
            return {'success': True, 'results': result}
            
        except Exception as e:
            logger.error(f"Error in Mood median test: {str(e)}")
            return {'success': False, 'error': str(e)}

class FeatureEngineer:
    """Advanced feature engineering class with comprehensive transformations"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.original_data = data.copy()
        self.transformers = {}
        self.feature_history = []
        self.scaler_fitted = False
    
    def apply_comprehensive_transformations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply comprehensive feature engineering based on configuration"""
        try:
            results = {}
            
            # Scaling transformations
            if config.get('scaling'):
                scaling_config = config['scaling']
                columns = scaling_config.get('columns', [])
                method = scaling_config.get('method', 'standard')
                results['scaling'] = self.apply_scaling(columns, method)
            
            # Encoding transformations
            if config.get('encoding'):
                encoding_config = config['encoding']
                columns = encoding_config.get('columns', [])
                method = encoding_config.get('method', 'onehot')
                results['encoding'] = self.apply_encoding(columns, method)
            
            # Mathematical transformations
            if config.get('mathematical'):
                math_config = config['mathematical']
                columns = math_config.get('columns', [])
                method = math_config.get('method', 'log')
                results['mathematical'] = self.apply_mathematical_transformations(columns, method)
            
            # Binning
            if config.get('binning'):
                binning_config = config['binning']
                columns = binning_config.get('columns', [])
                method = binning_config.get('method', 'equal_width')
                n_bins = binning_config.get('n_bins', 5)
                results['binning'] = self.create_binning_features(columns, method, n_bins)
            
            # DateTime features
            if config.get('datetime'):
                datetime_config = config['datetime']
                columns = datetime_config.get('columns', [])
                results['datetime'] = self.create_datetime_features(columns)
            
            # Text features
            if config.get('text'):
                text_config = config['text']
                columns = text_config.get('columns', [])
                results['text'] = self.create_text_features(columns)
            
            # Polynomial features
            if config.get('polynomial'):
                poly_config = config['polynomial']
                columns = poly_config.get('columns', [])
                degree = poly_config.get('degree', 2)
                results['polynomial'] = self.create_polynomial_features(columns, degree)
            
            # Interaction features
            if config.get('interactions'):
                interaction_config = config['interactions']
                columns = interaction_config.get('columns', [])
                results['interactions'] = self.create_interaction_features(columns)
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            logger.error(f"Error in comprehensive transformations: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def apply_scaling(self, columns: List[str], method: str = 'standard') -> Dict[str, Any]:
        """Apply various scaling methods"""
        try:
            if not columns:
                return {'success': False, 'error': 'No columns provided for scaling'}
            
            # Validate columns exist and are numeric
            valid_columns = []
            for col in columns:
                if col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col]):
                    valid_columns.append(col)
            
            if not valid_columns:
                return {'success': False, 'error': 'No valid numeric columns found'}
            
            # Select scaler
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'quantile_uniform':
                scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
            elif method == 'quantile_normal':
                scaler = QuantileTransformer(output_distribution='normal', random_state=42)
            elif method == 'maxabs':
                scaler = MaxAbsScaler()
            elif method == 'normalizer':
                scaler = Normalizer()
            else:
                return {'success': False, 'error': f'Unknown scaling method: {method}'}
            
            # Prepare data (handle missing values)
            data_to_scale = self.data[valid_columns].copy()
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data_to_scale),
                columns=valid_columns,
                index=data_to_scale.index
            )
            
            # Fit and transform
            scaled_data = scaler.fit_transform(data_imputed)
            
            # Create new column names
            new_columns = [f"{col}_{method}_scaled" for col in valid_columns]
            
            # Add scaled columns to dataframe
            scaled_df = pd.DataFrame(scaled_data, columns=new_columns, index=self.data.index)
            self.data = pd.concat([self.data, scaled_df], axis=1)
            
            # Store transformer
            transformer_key = f'{method}_scaler_{len(self.transformers)}'
            self.transformers[transformer_key] = {
                'transformer': scaler,
                'imputer': imputer,
                'original_columns': valid_columns,
                'new_columns': new_columns,
                'method': method
            }
            
            # Add to history
            self.feature_history.append({
                'operation': f'{method}_scaling',
                'columns': valid_columns,
                'new_columns': new_columns,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'new_columns': new_columns,
                'original_columns': valid_columns,
                'transformation_info': {
                    'method': method,
                    'columns_processed': len(valid_columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in scaling: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def apply_encoding(self, columns: List[str], method: str = 'onehot') -> Dict[str, Any]:
        """Apply various encoding methods for categorical variables"""
        try:
            if not columns:
                return {'success': False, 'error': 'No columns provided for encoding'}
            
            # Validate columns exist
            valid_columns = [col for col in columns if col in self.data.columns]
            if not valid_columns:
                return {'success': False, 'error': 'No valid columns found'}
            
            new_columns = []
            
            if method == 'onehot':
                for col in valid_columns:
                    try:
                        # Handle missing values
                        series_clean = self.data[col].fillna('missing')
                        
                        # Limit categories to prevent explosion
                        value_counts = series_clean.value_counts()
                        if len(value_counts) > 20:
                            top_categories = value_counts.head(19).index.tolist()
                            series_clean = series_clean.where(
                                series_clean.isin(top_categories), 'other'
                            )
                        
                        # Apply one-hot encoding
                        encoded_df = pd.get_dummies(
                            series_clean, 
                            prefix=col, 
                            dummy_na=False,
                            drop_first=True
                        )
                        
                        # Add to main dataframe
                        self.data = pd.concat([self.data, encoded_df], axis=1)
                        new_columns.extend(encoded_df.columns.tolist())
                        
                        # Store information
                        self.transformers[f'onehot_{col}'] = {
                            'method': 'onehot',
                            'original_column': col,
                            'new_columns': encoded_df.columns.tolist(),
                            'categories': series_clean.unique().tolist()
                        }
                        
                    except Exception as e:
                        logger.warning(f"Error encoding column {col}: {str(e)}")
            
            elif method == 'label':
                for col in valid_columns:
                    try:
                        # Handle missing values
                        series = self.data[col].copy()
                        
                        # Create label encoder
                        encoder = LabelEncoder()
                        
                        # Fit on non-null values
                        non_null_mask = series.notna()
                        if non_null_mask.any():
                            unique_values = series[non_null_mask].unique()
                            encoder.fit(unique_values)
                            
                            # Create new column
                            new_col_name = f"{col}_label_encoded"
                            encoded_series = pd.Series(index=series.index, dtype='float64')
                            encoded_series[non_null_mask] = encoder.transform(series[non_null_mask])
                            
                            self.data[new_col_name] = encoded_series
                            new_columns.append(new_col_name)
                            
                            # Store transformer
                            self.transformers[f'label_{col}'] = {
                                'method': 'label',
                                'transformer': encoder,
                                'original_column': col,
                                'new_column': new_col_name,
                                'classes': encoder.classes_.tolist()
                            }
                            
                    except Exception as e:
                        logger.warning(f"Error label encoding column {col}: {str(e)}")
            
            elif method == 'frequency':
                for col in valid_columns:
                    try:
                        # Calculate frequency encoding
                        freq_map = self.data[col].value_counts(normalize=True).to_dict()
                        new_col_name = f"{col}_frequency_encoded"
                        self.data[new_col_name] = self.data[col].map(freq_map).fillna(0)
                        new_columns.append(new_col_name)
                        
                        # Store transformer
                        self.transformers[f'frequency_{col}'] = {
                            'method': 'frequency',
                            'frequency_map': freq_map,
                            'original_column': col,
                            'new_column': new_col_name
                        }
                        
                    except Exception as e:
                        logger.warning(f"Error frequency encoding column {col}: {str(e)}")
            
            elif method == 'binary':
                for col in valid_columns:
                    try:
                        # Simple binary encoding
                        unique_values = self.data[col].dropna().unique()
                        if len(unique_values) <= 20:  # Reasonable limit
                            # Create binary representation
                            n_bits = int(np.ceil(np.log2(len(unique_values))))
                            
                            # Create mapping
                            value_to_binary = {}
                            for i, val in enumerate(unique_values):
                                binary_rep = format(i, f'0{n_bits}b')
                                value_to_binary[val] = binary_rep
                            
                            # Create binary columns
                            for bit_pos in range(n_bits):
                                new_col_name = f"{col}_binary_{bit_pos}"
                                bit_values = self.data[col].map(
                                    lambda x: int(value_to_binary.get(x, '0' * n_bits)[bit_pos]) 
                                    if pd.notna(x) else 0
                                )
                                self.data[new_col_name] = bit_values
                                new_columns.append(new_col_name)
                            
                            # Store transformer
                            self.transformers[f'binary_{col}'] = {
                                'method': 'binary',
                                'mapping': value_to_binary,
                                'original_column': col,
                                'new_columns': [f"{col}_binary_{i}" for i in range(n_bits)]
                            }
                            
                    except Exception as e:
                        logger.warning(f"Error binary encoding column {col}: {str(e)}")
            
            elif method == 'target':
                # Note: This would require a target variable
                return {'success': False, 'error': 'Target encoding requires target variable specification'}
            
            else:
                return {'success': False, 'error': f'Unknown encoding method: {method}'}
            
            # Add to history
            self.feature_history.append({
                'operation': f'{method}_encoding',
                'columns': valid_columns,
                'new_columns': new_columns,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'new_columns': new_columns,
                'original_columns': valid_columns,
                'transformation_info': {
                    'method': method,
                    'columns_processed': len(valid_columns),
                    'features_created': len(new_columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in encoding: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def apply_mathematical_transformations(self, columns: List[str], method: str = 'log') -> Dict[str, Any]:
        """Apply mathematical transformations"""
        try:
            if not columns:
                return {'success': False, 'error': 'No columns provided'}
            
            # Validate columns are numeric
            numeric_columns = []
            for col in columns:
                if col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_columns.append(col)
            
            if not numeric_columns:
                return {'success': False, 'error': 'No valid numeric columns found'}
            
            new_columns = []
            
            for col in numeric_columns:
                try:
                    series = self.data[col].copy()
                    
                    if method == 'log':
                        # Log transformation (handle zeros and negatives)
                        new_col_name = f"{col}_log"
                        # Add small constant to handle zeros, absolute value for negatives
                        transformed = np.log1p(np.abs(series))
                        self.data[new_col_name] = transformed
                        
                    elif method == 'sqrt':
                        # Square root transformation
                        new_col_name = f"{col}_sqrt"
                        transformed = np.sqrt(np.abs(series))
                        self.data[new_col_name] = transformed
                        
                    elif method == 'square':
                        # Square transformation
                        new_col_name = f"{col}_squared"
                        self.data[new_col_name] = series ** 2
                        
                    elif method == 'reciprocal':
                        # Reciprocal transformation
                        new_col_name = f"{col}_reciprocal"
                        # Avoid division by zero
                        self.data[new_col_name] = 1 / (series + 1e-8)
                        
                    elif method == 'boxcox':
                        # Box-Cox transformation
                        new_col_name = f"{col}_boxcox"
                        # Ensure positive values
                        positive_series = series + abs(series.min()) + 1
                        clean_series = positive_series.dropna()
                        
                        if len(clean_series) > 0:
                            transformed_data, lambda_param = stats.boxcox(clean_series)
                            
                            # Apply to full series
                            full_transformed = pd.Series(index=series.index, dtype=float)
                            full_transformed.loc[clean_series.index] = transformed_data
                            self.data[new_col_name] = full_transformed
                            
                            # Store lambda parameter
                            self.transformers[f'boxcox_{col}'] = {
                                'lambda': lambda_param,
                                'shift': abs(series.min()) + 1,
                                'original_column': col,
                                'new_column': new_col_name,
                                'method': 'boxcox'
                            }
                    
                    elif method == 'yeo_johnson':
                        # Yeo-Johnson transformation (can handle negative values)
                        new_col_name = f"{col}_yeo_johnson"
                        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                        
                        clean_data = series.dropna().values.reshape(-1, 1)
                        if len(clean_data) > 0:
                            transformed_data = transformer.fit_transform(clean_data)
                            
                            # Apply to full series
                            full_transformed = pd.Series(index=series.index, dtype=float)
                            full_transformed.loc[series.dropna().index] = transformed_data.flatten()
                            self.data[new_col_name] = full_transformed
                            
                            self.transformers[f'yeo_johnson_{col}'] = {
                                'transformer': transformer,
                                'original_column': col,
                                'new_column': new_col_name,
                                'method': 'yeo_johnson'
                            }
                    
                    elif method == 'inverse':
                        # Inverse transformation
                        new_col_name = f"{col}_inverse"
                        self.data[new_col_name] = -series
                        
                    elif method == 'exp':
                        # Exponential transformation (be careful with large values)
                        new_col_name = f"{col}_exp"
                        # Clip to prevent overflow
                        clipped_series = np.clip(series, -50, 50)
                        self.data[new_col_name] = np.exp(clipped_series)
                    
                    else:
                        continue
                    
                    new_columns.append(new_col_name)
                    
                except Exception as e:
                    logger.warning(f"Error transforming column {col} with {method}: {str(e)}")
            
            # Add to history
            self.feature_history.append({
                'operation': f'{method}_transformation',
                'columns': numeric_columns,
                'new_columns': new_columns,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'new_columns': new_columns,
                'original_columns': numeric_columns,
                'transformation_info': {
                    'method': method,
                    'columns_processed': len(numeric_columns),
                    'features_created': len(new_columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in mathematical transformations: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_binning_features(self, columns: List[str], method: str = 'equal_width', n_bins: int = 5) -> Dict[str, Any]:
        """Create binning features"""
        try:
            if not columns:
                return {'success': False, 'error': 'No columns provided'}
            
            numeric_columns = []
            for col in columns:
                if col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_columns.append(col)
            
            if not numeric_columns:
                return {'success': False, 'error': 'No valid numeric columns found'}
            
            new_columns = []
            
            for col in numeric_columns:
                try:
                    series = self.data[col].dropna()
                    if len(series.unique()) < 2:
                        continue  # Skip constant columns
                    
                    if method == 'equal_width':
                        new_col_name = f"{col}_binned_width_{n_bins}"
                        binned_data = pd.cut(self.data[col], bins=n_bins, labels=False, duplicates='drop')
                        self.data[new_col_name] = binned_data
                        
                        # Store bin edges
                        bin_edges = pd.cut(series, bins=n_bins, retbins=True, duplicates='drop')[1]
                        
                        self.transformers[f'binning_width_{col}'] = {
                            'method': 'equal_width',
                            'n_bins': n_bins,
                            'bin_edges': bin_edges.tolist(),
                            'original_column': col,
                            'new_column': new_col_name
                        }
                        
                    elif method == 'equal_frequency':
                        new_col_name = f"{col}_binned_freq_{n_bins}"
                        try:
                            binned_data = pd.qcut(self.data[col], q=n_bins, labels=False, duplicates='drop')
                            self.data[new_col_name] = binned_data
                            
                            # Store quantiles
                            quantiles = self.data[col].quantile([i/n_bins for i in range(n_bins+1)]).tolist()
                            
                            self.transformers[f'binning_freq_{col}'] = {
                                'method': 'equal_frequency',
                                'n_bins': n_bins,
                                'quantiles': quantiles,
                                'original_column': col,
                                'new_column': new_col_name
                            }
                            
                        except ValueError as e:
                            # Handle case where quantiles are not unique
                            logger.warning(f"Could not create equal frequency bins for {col}: {str(e)}")
                            continue
                    
                    elif method == 'kmeans':
                        # K-means based binning
                        new_col_name = f"{col}_binned_kmeans_{n_bins}"
                        
                        clean_data = series.values.reshape(-1, 1)
                        kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(clean_data)
                        
                        # Map back to full series
                        full_labels = pd.Series(index=self.data.index, dtype='float64')
                        full_labels.loc[series.index] = cluster_labels
                        self.data[new_col_name] = full_labels
                        
                        self.transformers[f'binning_kmeans_{col}'] = {
                            'method': 'kmeans',
                            'n_bins': n_bins,
                            'kmeans_model': kmeans,
                            'original_column': col,
                            'new_column': new_col_name
                        }
                    
                    else:
                        continue
                    
                    new_columns.append(new_col_name)
                    
                except Exception as e:
                    logger.warning(f"Error binning column {col}: {str(e)}")
            
            # Add to history
            self.feature_history.append({
                'operation': f'{method}_binning',
                'columns': numeric_columns,
                'new_columns': new_columns,
                'n_bins': n_bins,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'new_columns': new_columns,
                'original_columns': numeric_columns,
                'transformation_info': {
                    'method': method,
                    'n_bins': n_bins,
                    'columns_processed': len(numeric_columns),
                    'features_created': len(new_columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in binning: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_datetime_features(self, columns: List[str]) -> Dict[str, Any]:
        """Extract comprehensive datetime features"""
        try:
            if not columns:
                return {'success': False, 'error': 'No columns provided'}
            
            datetime_columns = []
            for col in columns:
                if col in self.data.columns:
                    # Try to convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(self.data[col]):
                        try:
                            self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                        except:
                            continue
                    
                    if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                        datetime_columns.append(col)
            
            if not datetime_columns:
                return {'success': False, 'error': 'No valid datetime columns found'}
            
            new_columns = []
            
            for col in datetime_columns:
                try:
                    dt_series = self.data[col]
                    col_new_columns = []
                    
                    # Basic datetime components
                    self.data[f"{col}_year"] = dt_series.dt.year
                    self.data[f"{col}_month"] = dt_series.dt.month
                    self.data[f"{col}_day"] = dt_series.dt.day
                    self.data[f"{col}_weekday"] = dt_series.dt.weekday
                    self.data[f"{col}_hour"] = dt_series.dt.hour
                    self.data[f"{col}_minute"] = dt_series.dt.minute
                    self.data[f"{col}_second"] = dt_series.dt.second
                    
                    col_new_columns.extend([
                        f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_weekday",
                        f"{col}_hour", f"{col}_minute", f"{col}_second"
                    ])
                    
                    # Advanced datetime features
                    self.data[f"{col}_quarter"] = dt_series.dt.quarter
                    self.data[f"{col}_day_of_year"] = dt_series.dt.dayofyear
                    self.data[f"{col}_week_of_year"] = dt_series.dt.isocalendar().week
                    self.data[f"{col}_days_in_month"] = dt_series.dt.days_in_month
                    
                    col_new_columns.extend([
                        f"{col}_quarter", f"{col}_day_of_year", f"{col}_week_of_year", f"{col}_days_in_month"
                    ])
                    
                    # Boolean features
                    self.data[f"{col}_is_weekend"] = (dt_series.dt.weekday >= 5).astype(int)
                    self.data[f"{col}_is_month_start"] = dt_series.dt.is_month_start.astype(int)
                    self.data[f"{col}_is_month_end"] = dt_series.dt.is_month_end.astype(int)
                    self.data[f"{col}_is_quarter_start"] = dt_series.dt.is_quarter_start.astype(int)
                    self.data[f"{col}_is_quarter_end"] = dt_series.dt.is_quarter_end.astype(int)
                    self.data[f"{col}_is_year_start"] = dt_series.dt.is_year_start.astype(int)
                    self.data[f"{col}_is_year_end"] = dt_series.dt.is_year_end.astype(int)
                    
                    col_new_columns.extend([
                        f"{col}_is_weekend", f"{col}_is_month_start", f"{col}_is_month_end",
                        f"{col}_is_quarter_start", f"{col}_is_quarter_end", 
                        f"{col}_is_year_start", f"{col}_is_year_end"
                    ])
                    
                    # Cyclical features (sine and cosine encoding)
                    self.data[f"{col}_month_sin"] = np.sin(2 * np.pi * dt_series.dt.month / 12)
                    self.data[f"{col}_month_cos"] = np.cos(2 * np.pi * dt_series.dt.month / 12)
                    self.data[f"{col}_day_sin"] = np.sin(2 * np.pi * dt_series.dt.day / 31)
                    self.data[f"{col}_day_cos"] = np.cos(2 * np.pi * dt_series.dt.day / 31)
                    self.data[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
                    self.data[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
                    self.data[f"{col}_weekday_sin"] = np.sin(2 * np.pi * dt_series.dt.weekday / 7)
                    self.data[f"{col}_weekday_cos"] = np.cos(2 * np.pi * dt_series.dt.weekday / 7)
                    
                    col_new_columns.extend([
                        f"{col}_month_sin", f"{col}_month_cos", f"{col}_day_sin", f"{col}_day_cos",
                        f"{col}_hour_sin", f"{col}_hour_cos", f"{col}_weekday_sin", f"{col}_weekday_cos"
                    ])
                    
                    # Time since reference point
                    reference_date = dt_series.min()
                    if pd.notna(reference_date):
                        time_diff = dt_series - reference_date
                        self.data[f"{col}_days_since_start"] = time_diff.dt.days
                        self.data[f"{col}_hours_since_start"] = time_diff.dt.total_seconds() / 3600
                        
                        col_new_columns.extend([
                            f"{col}_days_since_start", f"{col}_hours_since_start"
                        ])
                    
                    # Season feature
                    def get_season(month):
                        if month in [12, 1, 2]:
                            return 0  # Winter
                        elif month in [3, 4, 5]:
                            return 1  # Spring
                        elif month in [6, 7, 8]:
                            return 2  # Summer
                        else:
                            return 3  # Fall
                    
                    self.data[f"{col}_season"] = dt_series.dt.month.apply(get_season)
                    col_new_columns.append(f"{col}_season")
                    
                    # Time of day categories
                    hour = dt_series.dt.hour
                    self.data[f"{col}_time_of_day"] = pd.cut(
                        hour, 
                        bins=[0, 6, 12, 18, 24], 
                        labels=[0, 1, 2, 3],  # Night, Morning, Afternoon, Evening
                        include_lowest=True
                    ).astype(float)
                    col_new_columns.append(f"{col}_time_of_day")
                    
                    new_columns.extend(col_new_columns)
                    
                    # Store datetime feature information
                    self.transformers[f'datetime_{col}'] = {
                        'original_column': col,
                        'new_columns': col_new_columns,
                        'method': 'datetime_extraction',
                        'reference_date': str(reference_date) if pd.notna(reference_date) else None
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing datetime column {col}: {str(e)}")
            
            # Add to history
            self.feature_history.append({
                'operation': 'datetime_feature_extraction',
                'columns': datetime_columns,
                'new_columns': new_columns,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'new_columns': new_columns,
                'original_columns': datetime_columns,
                'transformation_info': {
                    'datetime_columns_processed': len(datetime_columns),
                    'features_created': len(new_columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in datetime feature extraction: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_text_features(self, columns: List[str]) -> Dict[str, Any]:
        """Extract comprehensive text features"""
        try:
            if not columns:
                return {'success': False, 'error': 'No columns provided'}
            
            text_columns = [col for col in columns if col in self.data.columns]
            if not text_columns:
                return {'success': False, 'error': 'No valid columns found'}
            
            new_columns = []
            
            for col in text_columns:
                try:
                    text_series = self.data[col].astype(str).fillna('')
                    col_new_columns = []
                    
                    # Basic text features
                    self.data[f"{col}_char_count"] = text_series.str.len()
                    self.data[f"{col}_word_count"] = text_series.str.split().str.len().fillna(0)
                    self.data[f"{col}_sentence_count"] = text_series.str.count(r'[.!?]+') + 1
                    
                    # Calculate average word length safely
                    def safe_avg_word_length(text):
                        if not text or pd.isna(text):
                            return 0
                        words = str(text).split()
                        return np.mean([len(word) for word in words]) if words else 0
                    
                    self.data[f"{col}_avg_word_length"] = text_series.apply(safe_avg_word_length)
                    
                    col_new_columns.extend([
                        f"{col}_char_count", f"{col}_word_count", 
                        f"{col}_sentence_count", f"{col}_avg_word_length"
                    ])
                    
                    # Character-based features
                    self.data[f"{col}_uppercase_count"] = text_series.str.count(r'[A-Z]')
                    self.data[f"{col}_lowercase_count"] = text_series.str.count(r'[a-z]')
                    self.data[f"{col}_digit_count"] = text_series.str.count(r'\d')
                    self.data[f"{col}_special_char_count"] = text_series.str.count(r'[^a-zA-Z0-9\s]')
                    self.data[f"{col}_whitespace_count"] = text_series.str.count(r'\s')
                    self.data[f"{col}_punctuation_count"] = text_series.str.count(r'[^\w\s]')
                    
                    col_new_columns.extend([
                        f"{col}_uppercase_count", f"{col}_lowercase_count", f"{col}_digit_count",
                        f"{col}_special_char_count", f"{col}_whitespace_count", f"{col}_punctuation_count"
                    ])
                    
                    # Linguistic features
                    self.data[f"{col}_unique_words"] = text_series.apply(
                        lambda x: len(set(str(x).lower().split())) if x else 0
                    )
                    
                    def lexical_diversity(text):
                        if not text:
                            return 0
                        words = str(text).lower().split()
                        return len(set(words)) / len(words) if words else 0
                    
                    self.data[f"{col}_lexical_diversity"] = text_series.apply(lexical_diversity)
                    
                    col_new_columns.extend([f"{col}_unique_words", f"{col}_lexical_diversity"])
                    
                    # Readability features
                    def avg_sentence_length(text):
                        if not text:
                            return 0
                        sentences = str(text).split('.')
                        word_counts = [len(sentence.split()) for sentence in sentences if sentence.strip()]
                        return np.mean(word_counts) if word_counts else 0
                    
                    self.data[f"{col}_avg_sentence_length"] = text_series.apply(avg_sentence_length)
                    col_new_columns.append(f"{col}_avg_sentence_length")
                    
                    # Sentiment analysis (if TextBlob is available)
                    if TEXTBLOB_AVAILABLE:
                        try:
                            def safe_sentiment(text):
                                try:
                                    if not text or pd.isna(text):
                                        return 0.0
                                    return TextBlob(str(text)).sentiment.polarity
                                except:
                                    return 0.0
                            
                            def safe_subjectivity(text):
                                try:
                                    if not text or pd.isna(text):
                                        return 0.0
                                    return TextBlob(str(text)).sentiment.subjectivity
                                except:
                                    return 0.0
                            
                            self.data[f"{col}_sentiment_polarity"] = text_series.apply(safe_sentiment)
                            self.data[f"{col}_sentiment_subjectivity"] = text_series.apply(safe_subjectivity)
                            
                            col_new_columns.extend([
                                f"{col}_sentiment_polarity", f"{col}_sentiment_subjectivity"
                            ])
                            
                        except Exception as e:
                            logger.warning(f"Sentiment analysis failed for {col}: {str(e)}")
                    
                    # Pattern-based features
                    self.data[f"{col}_contains_email"] = text_series.str.contains(
                        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                        regex=True, na=False
                    ).astype(int)
                    
                    self.data[f"{col}_contains_url"] = text_series.str.contains(
                        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                        regex=True, na=False
                    ).astype(int)
                    
                    self.data[f"{col}_contains_phone"] = text_series.str.contains(
                        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                        regex=True, na=False
                    ).astype(int)
                    
                    col_new_columns.extend([
                        f"{col}_contains_email", f"{col}_contains_url", f"{col}_contains_phone"
                    ])
                    
                    # Language detection features (basic)
                    self.data[f"{col}_mostly_caps"] = (
                        text_series.str.count(r'[A-Z]') > text_series.str.count(r'[a-z]')
                    ).astype(int)
                    
                    self.data[f"{col}_has_numbers"] = text_series.str.contains(r'\d', na=False).astype(int)
                    
                    col_new_columns.extend([f"{col}_mostly_caps", f"{col}_has_numbers"])
                    
                    new_columns.extend(col_new_columns)
                    
                    # Store text feature information
                    self.transformers[f'text_{col}'] = {
                        'original_column': col,
                        'new_columns': col_new_columns,
                        'method': 'text_feature_extraction'
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing text column {col}: {str(e)}")
            
            # Add to history
            self.feature_history.append({
                'operation': 'text_feature_extraction',
                'columns': text_columns,
                'new_columns': new_columns,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'new_columns': new_columns,
                'original_columns': text_columns,
                'transformation_info': {
                    'text_columns_processed': len(text_columns),
                    'features_created': len(new_columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in text feature extraction: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_interaction_features(self, columns: List[str], 
                                   interaction_types: List[str] = ['multiply', 'divide', 'add', 'subtract']) -> Dict[str, Any]:
        """Create interaction features between numerical columns"""
        try:
            if not columns:
                return {'success': False, 'error': 'No columns provided'}
            
            numerical_cols = []
            for col in columns:
                if col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col]):
                    numerical_cols.append(col)
            
            if len(numerical_cols) < 2:
                return {'success': False, 'error': 'Need at least 2 numerical columns for interactions'}
            
            new_columns = []
            
            # Limit combinations to prevent explosion
            max_combinations = 50
            combinations = 0
            
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    if combinations >= max_combinations:
                        break
                    
                    try:
                        # Multiplication
                        if 'multiply' in interaction_types:
                            new_col_name = f"{col1}_{col2}_multiply"
                            self.data[new_col_name] = self.data[col1] * self.data[col2]
                            new_columns.append(new_col_name)
                        
                        # Division
                        if 'divide' in interaction_types:
                            new_col_name = f"{col1}_{col2}_divide"
                            # Avoid division by zero
                            denominator = self.data[col2] + 1e-8
                            self.data[new_col_name] = self.data[col1] / denominator
                            new_columns.append(new_col_name)
                        
                        # Addition
                        if 'add' in interaction_types:
                            new_col_name = f"{col1}_{col2}_add"
                            self.data[new_col_name] = self.data[col1] + self.data[col2]
                            new_columns.append(new_col_name)
                        
                        # Subtraction
                        if 'subtract' in interaction_types:
                            new_col_name = f"{col1}_{col2}_subtract"
                            self.data[new_col_name] = self.data[col1] - self.data[col2]
                            new_columns.append(new_col_name)
                        
                        # Statistical interactions
                        if 'max' in interaction_types:
                            new_col_name = f"{col1}_{col2}_max"
                            self.data[new_col_name] = np.maximum(self.data[col1], self.data[col2])
                            new_columns.append(new_col_name)
                        
                        if 'min' in interaction_types:
                            new_col_name = f"{col1}_{col2}_min"
                            self.data[new_col_name] = np.minimum(self.data[col1], self.data[col2])
                            new_columns.append(new_col_name)
                        
                        # Ratio features
                        if 'ratio' in interaction_types:
                            new_col_name = f"{col1}_{col2}_ratio"
                            # Handle zero denominators
                            ratio = np.where(
                                np.abs(self.data[col2]) > 1e-8,
                                self.data[col1] / self.data[col2],
                                0
                            )
                            self.data[new_col_name] = ratio
                            new_columns.append(new_col_name)
                        
                        combinations += 1
                        
                    except Exception as e:
                        logger.warning(f"Error creating interaction between {col1} and {col2}: {str(e)}")
                
                if combinations >= max_combinations:
                    break
            
            # Store interaction information
            self.transformers['interactions'] = {
                'original_columns': numerical_cols,
                'new_columns': new_columns,
                'interaction_types': interaction_types,
                'method': 'interaction_features'
            }
            
            # Add to history
            self.feature_history.append({
                'operation': 'interaction_features',
                'columns': numerical_cols,
                'new_columns': new_columns,
                'interaction_types': interaction_types,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'new_columns': new_columns,
                'original_columns': numerical_cols,
                'transformation_info': {
                    'interaction_types': interaction_types,
                    'features_created': len(new_columns),
                    'combinations_created': combinations
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_polynomial_features(self, columns: List[str], degree: int = 2, 
                                  include_bias: bool = False, interaction_only: bool = False) -> Dict[str, Any]:
        """Create polynomial features"""
        try:
            if not columns:
                return {'success': False, 'error': 'No columns provided'}
            
            numerical_cols = []
            for col in columns:
                if col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col]):
                    numerical_cols.append(col)
            
            if not numerical_cols:
                return {'success': False, 'error': 'No numerical columns found for polynomial features'}
            
            # Limit number of features to prevent explosion
            max_features = 20
            if len(numerical_cols) > max_features:
                numerical_cols = numerical_cols[:max_features]
                logger.warning(f"Limited to first {max_features} columns to prevent feature explosion")
            
            # Prepare data (handle missing values)
            data_subset = self.data[numerical_cols].copy()
            
            # Fill missing values with median
            imputer = SimpleImputer(strategy='median')
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data_subset),
                columns=numerical_cols,
                index=data_subset.index
            )
            
            # Create polynomial features
            poly = PolynomialFeatures(
                degree=degree, 
                include_bias=include_bias, 
                interaction_only=interaction_only
            )
            poly_features = poly.fit_transform(data_imputed)
            
            # Get feature names
            feature_names = poly.get_feature_names_out(numerical_cols)
            
            # Identify new features (exclude original ones)
            if not include_bias:
                # Skip original features (first len(numerical_cols) features)
                new_feature_names = feature_names[len(numerical_cols):]
                new_features = poly_features[:, len(numerical_cols):]
            else:
                # Skip bias and original features
                new_feature_names = feature_names[len(numerical_cols)+1:]
                new_features = poly_features[:, len(numerical_cols)+1:]
            
            # Create more readable names
            readable_names = []
            for name in new_feature_names:
                # Clean up the polynomial feature names
                clean_name = name.replace(' ', '_').replace('^', '_pow_')
                readable_name = f"poly_{clean_name}"
                readable_names.append(readable_name)
            
            # Add to dataframe
            if len(readable_names) > 0:
                poly_df = pd.DataFrame(
                    new_features, 
                    columns=readable_names, 
                    index=self.data.index
                )
                self.data = pd.concat([self.data, poly_df], axis=1)
            
            # Store polynomial information
            self.transformers['polynomial'] = {
                'transformer': poly,
                'imputer': imputer,
                'original_columns': numerical_cols,
                'new_columns': readable_names,
                'degree': degree,
                'include_bias': include_bias,
                'interaction_only': interaction_only,
                'method': 'polynomial_features'
            }
            
            # Add to history
            self.feature_history.append({
                'operation': 'polynomial_features',
                'columns': numerical_cols,
                'new_columns': readable_names,
                'degree': degree,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'new_columns': readable_names,
                'original_columns': numerical_cols,
                'transformation_info': {
                    'degree': degree,
                    'original_features': len(numerical_cols),
                    'polynomial_features_created': len(readable_names),
                    'include_bias': include_bias,
                    'interaction_only': interaction_only
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating polynomial features: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_feature_history(self) -> List[Dict[str, Any]]:
        """Get the history of feature engineering operations"""
        return self.feature_history
    
    def get_data(self) -> pd.DataFrame:
        """Get the current transformed data"""
        return self.data
    
    def get_transformer_info(self) -> Dict[str, Any]:
        """Get information about all transformers"""
        transformer_summary = {}
        for key, transformer in self.transformers.items():
            # Create a serializable summary
            transformer_summary[key] = {
                'method': transformer.get('method', 'unknown'),
                'original_columns': transformer.get('original_columns', []),
                'new_columns': transformer.get('new_columns', []),
                'parameters': {k: v for k, v in transformer.items() 
                             if k not in ['transformer', 'imputer'] and not callable(v)}
            }
        return transformer_summary
    
    def reset_data(self) -> Dict[str, Any]:
        """Reset data to original state"""
        try:
            self.data = self.original_data.copy()
            self.transformers = {}
            self.feature_history = []
            self.scaler_fitted = False
            
            return {
                'success': True,
                'message': 'Data reset to original state',
                'current_shape': self.data.shape
            }
            
        except Exception as e:
            logger.error(f"Error resetting data: {str(e)}")
            return {'success': False, 'error': str(e)}

class VisualizationEngine:
    """Advanced visualization engine with 60+ chart types"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.color_palettes = {
            'dark': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'light': ['#2E86C1', '#E67E22', '#28B463', '#E74C3C', '#8E44AD', '#D68910'],
            'colorful': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
            'minimal': ['#34495E', '#7F8C8D', '#BDC3C7', '#ECF0F1', '#95A5A6', '#AEB6BF']
        }
    
    def create_comprehensive_visualization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization based on comprehensive configuration"""
        try:
            chart_type = config.get('chart_type')
            
            # Route to appropriate visualization method
            if chart_type in ['histogram', 'hist']:
                return self._create_histogram(config)
            elif chart_type in ['boxplot', 'box']:
                return self._create_boxplot(config)
            elif chart_type in ['scatter', 'scatterplot']:
                return self._create_scatterplot(config)
            elif chart_type in ['line', 'lineplot']:
                return self._create_lineplot(config)
            elif chart_type in ['bar', 'barplot']:
                return self._create_barplot(config)
            elif chart_type in ['heatmap', 'correlation_heatmap']:
                return self._create_heatmap(config)
            elif chart_type in ['violin', 'violinplot']:
                return self._create_violin_plot(config)
            elif chart_type in ['density', 'kde']:
                return self._create_density_plot(config)
            elif chart_type == 'pairplot':
                return self._create_pairplot(config)
            elif chart_type == 'distribution':
                return self._create_distribution_plot(config)
            elif chart_type in ['3d_scatter', 'scatter3d']:
                return self._create_3d_scatter(config)
            elif chart_type == 'sunburst':
                return self._create_sunburst(config)
            elif chart_type == 'treemap':
                return self._create_treemap(config)
            elif chart_type == 'parallel_coordinates':
                return self._create_parallel_coordinates(config)
            elif chart_type == 'radar':
                return self._create_radar_chart(config)
            elif chart_type == 'pie':
                return self._create_pie_chart(config)
            elif chart_type == 'donut':
                return self._create_donut_chart(config)
            elif chart_type == 'waterfall':
                return self._create_waterfall_chart(config)
            elif chart_type == 'funnel':
                return self._create_funnel_chart(config)
            elif chart_type == 'gauge':
                return self._create_gauge_chart(config)
            elif chart_type == 'candlestick':
                return self._create_candlestick_chart(config)
            elif chart_type == 'ohlc':
                return self._create_ohlc_chart(config)
            elif chart_type == 'surface':
                return self._create_surface_plot(config)
            elif chart_type == 'contour':
                return self._create_contour_plot(config)
            elif chart_type == 'network':
                return self._create_network_plot(config)
            elif chart_type == 'sankey':
                return self._create_sankey_diagram(config)
            else:
                return {'success': False, 'error': f'Unknown chart type: {chart_type}'}
                
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _get_theme_template(self, theme: str) -> str:
        """Get Plotly template based on theme"""
        theme_mapping = {
            'dark': 'plotly_dark',
            'light': 'plotly_white',
            'colorful': 'plotly',
            'minimal': 'simple_white'
        }
        return theme_mapping.get(theme, 'plotly_dark')
    
    def _get_color_palette(self, theme: str) -> List[str]:
        """Get color palette based on theme"""
        return self.color_palettes.get(theme, self.color_palettes['dark'])
    
    def _create_histogram(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create advanced histogram with multiple options"""
        try:
            x_col = config.get('x_col')
            color_col = config.get('color_col')
            facet_col = config.get('facet_col')
            title = config.get('title', f'Distribution of {x_col}')
            theme = config.get('theme', 'dark')
            
            if not x_col or x_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing x column'}
            
            # Create histogram
            fig = px.histogram(
                self.data,
                x=x_col,
                color=color_col if color_col and color_col in self.data.columns else None,
                facet_col=facet_col if facet_col and facet_col in self.data.columns else None,
                title=title,
                template=self._get_theme_template(theme),
                color_discrete_sequence=self._get_color_palette(theme)
            )
            
            # Add statistical annotations
            if pd.api.types.is_numeric_dtype(self.data[x_col]):
                mean_val = self.data[x_col].mean()
                median_val = self.data[x_col].median()
                
                # Add vertical lines for mean and median
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                             annotation_text=f"Mean: {mean_val:.2f}")
                fig.add_vline(x=median_val, line_dash="dot", line_color="blue", 
                             annotation_text=f"Median: {median_val:.2f}")
            
            # Customize layout
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title='Count',
                showlegend=bool(color_col),
                hovermode='x unified'
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'histogram',
                'description': f'Histogram showing distribution of {x_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_boxplot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create advanced box plot"""
        try:
            x_col = config.get('x_col')
            y_col = config.get('y_col')
            color_col = config.get('color_col')
            title = config.get('title')
            theme = config.get('theme', 'dark')
            
            if not y_col or y_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing y column'}
            
            # Create box plot
            fig = px.box(
                self.data,
                x=x_col if x_col and x_col in self.data.columns else None,
                y=y_col,
                color=color_col if color_col and color_col in self.data.columns else None,
                title=title or f'Box Plot of {y_col}',
                template=self._get_theme_template(theme),
                color_discrete_sequence=self._get_color_palette(theme),
                points="outliers"  # Show outliers
            )
            
            # Add statistical annotations
            if pd.api.types.is_numeric_dtype(self.data[y_col]):
                q1 = self.data[y_col].quantile(0.25)
                q3 = self.data[y_col].quantile(0.75)
                iqr = q3 - q1
                
                # Add IQR annotation
                fig.add_annotation(
                    text=f"IQR: {iqr:.2f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=12)
                )
            
            fig.update_layout(
                xaxis_title=x_col if x_col else '',
                yaxis_title=y_col,
                showlegend=bool(color_col)
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'boxplot',
                'description': f'Box plot showing distribution of {y_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating box plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_scatterplot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create advanced scatter plot with regression line and confidence intervals"""
        try:
            x_col = config.get('x_col')
            y_col = config.get('y_col')
            color_col = config.get('color_col')
            size_col = config.get('size_col')
            title = config.get('title')
            theme = config.get('theme', 'dark')
            
            if not x_col or not y_col or x_col not in self.data.columns or y_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing x/y columns'}
            
            # Create scatter plot
            fig = px.scatter(
                self.data,
                x=x_col,
                y=y_col,
                color=color_col if color_col and color_col in self.data.columns else None,
                size=size_col if size_col and size_col in self.data.columns else None,
                title=title or f'{x_col} vs {y_col}',
                template=self._get_theme_template(theme),
                color_discrete_sequence=self._get_color_palette(theme),
                trendline="ols",  # Add regression line
                hover_data=[col for col in [color_col, size_col] if col and col in self.data.columns]
            )
            
            # Calculate correlation
            if pd.api.types.is_numeric_dtype(self.data[x_col]) and pd.api.types.is_numeric_dtype(self.data[y_col]):
                correlation = self.data[x_col].corr(self.data[y_col])
                
                # Add correlation annotation
                fig.add_annotation(
                    text=f"Correlation: {correlation:.3f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                
                # Add R² from trendline if available
                try:
                    # Extract R² from the trendline
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    
                    clean_data = self.data[[x_col, y_col]].dropna()
                    if len(clean_data) > 1:
                        X = clean_data[x_col].values.reshape(-1, 1)
                        y = clean_data[y_col].values
                        
                        lr = LinearRegression()
                        lr.fit(X, y)
                        y_pred = lr.predict(X)
                        r2 = r2_score(y, y_pred)
                        
                        fig.add_annotation(
                            text=f"R²: {r2:.3f}",
                            xref="paper", yref="paper",
                            x=0.02, y=0.92,
                            showarrow=False,
                            font=dict(size=12),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="black",
                            borderwidth=1
                        )
                except:
                    pass
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                showlegend=bool(color_col)
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'scatter',
                'description': f'Scatter plot of {x_col} vs {y_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_lineplot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create advanced line plot"""
        try:
            x_col = config.get('x_col')
            y_col = config.get('y_col')
            color_col = config.get('color_col')
            title = config.get('title')
            theme = config.get('theme', 'dark')
            
            if not x_col or not y_col or x_col not in self.data.columns or y_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing x/y columns'}
            
            # Sort data by x column for proper line plotting
            sorted_data = self.data.sort_values(x_col)
            
            fig = px.line(
                sorted_data,
                x=x_col,
                y=y_col,
                color=color_col if color_col and color_col in self.data.columns else None,
                title=title or f'{y_col} over {x_col}',
                template=self._get_theme_template(theme),
                color_discrete_sequence=self._get_color_palette(theme),
                markers=True
            )
            
            # Add trend analysis if numeric
            if pd.api.types.is_numeric_dtype(self.data[x_col]) and pd.api.types.is_numeric_dtype(self.data[y_col]):
                # Calculate trend
                clean_data = self.data[[x_col, y_col]].dropna()
                if len(clean_data) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[x_col], clean_data[y_col])
                    
                    # Add trend information
                    trend_text = f"Trend: {'↗' if slope > 0 else '↘'} (slope: {slope:.3f})"
                    fig.add_annotation(
                        text=trend_text,
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=12)
                    )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                showlegend=bool(color_col),
                hovermode='x unified'
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'line',
                'description': f'Line plot of {y_col} over {x_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating line plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_barplot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create advanced bar plot"""
        try:
            x_col = config.get('x_col')
            y_col = config.get('y_col')
            color_col = config.get('color_col')
            title = config.get('title')
            theme = config.get('theme', 'dark')
            orientation = config.get('orientation', 'vertical')
            
            if not x_col or x_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing x column'}
            
            # Prepare data
            if y_col and y_col in self.data.columns:
                # Aggregate data
                if pd.api.types.is_numeric_dtype(self.data[y_col]):
                    plot_data = self.data.groupby(x_col)[y_col].mean().reset_index()
                else:
                    plot_data = self.data
            else:
                # Count plot
                value_counts = self.data[x_col].value_counts().reset_index()
                value_counts.columns = [x_col, 'count']
                plot_data = value_counts
                y_col = 'count'
            
            # Create bar plot
            if orientation == 'horizontal':
                fig = px.bar(
                    plot_data,
                    x=y_col, y=x_col,
                    color=color_col if color_col and color_col in plot_data.columns else None,
                    title=title or f'{y_col} by {x_col}',
                    template=self._get_theme_template(theme),
                    color_discrete_sequence=self._get_color_palette(theme),
                    orientation='h'
                )
            else:
                fig = px.bar(
                    plot_data,
                    x=x_col, y=y_col,
                    color=color_col if color_col and color_col in plot_data.columns else None,
                    title=title or f'{y_col} by {x_col}',
                    template=self._get_theme_template(theme),
                    color_discrete_sequence=self._get_color_palette(theme)
                )
            
            # Add value labels on bars
            if len(plot_data) < 20:  # Only for reasonable number of bars
                fig.update_traces(texttemplate='%{y}', textposition='outside')
            
            fig.update_layout(
                xaxis_title=x_col if orientation == 'vertical' else y_col,
                yaxis_title=y_col if orientation == 'vertical' else x_col,
                showlegend=bool(color_col)
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'bar',
                'description': f'Bar plot of {y_col} by {x_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating bar plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_heatmap(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create advanced correlation heatmap"""
        try:
            title = config.get('title', 'Correlation Heatmap')
            theme = config.get('theme', 'dark')
            method = config.get('correlation_method', 'pearson')
            
            # Get numerical columns
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) < 2:
                return {'success': False, 'error': 'Need at least 2 numerical columns for heatmap'}
            
            # Calculate correlation matrix
            corr_matrix = self.data[numerical_cols].corr(method=method)
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                title=title,
                template=self._get_theme_template(theme),
                color_continuous_scale='RdBu_r',
                aspect='auto',
                text_auto=True
            )
            
            # Add correlation values as text
            fig.update_traces(
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10}
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Variables',
                yaxis_title='Variables',
                title_x=0.5
            )
            
            # Add interpretation
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_correlations.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")
            
            description = f'Correlation heatmap using {method} method.'
            if strong_correlations:
                description += f" Strong correlations found: {', '.join(strong_correlations[:3])}"
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'heatmap',
                'description': description
            }
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_violin_plot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create advanced violin plot"""
        try:
            x_col = config.get('x_col')
            y_col = config.get('y_col')
            color_col = config.get('color_col')
            title = config.get('title')
            theme = config.get('theme', 'dark')
            
            if not y_col or y_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing y column'}
            
            fig = px.violin(
                self.data,
                x=x_col if x_col and x_col in self.data.columns else None,
                y=y_col,
                color=color_col if color_col and color_col in self.data.columns else None,
                title=title or f'Violin Plot of {y_col}',
                template=self._get_theme_template(theme),
                color_discrete_sequence=self._get_color_palette(theme),
                box=True,  # Show box plot inside violin
                points="outliers"  # Show outliers
            )
            
            # Add statistical information
            if pd.api.types.is_numeric_dtype(self.data[y_col]):
                stats_text = f"Mean: {self.data[y_col].mean():.2f}, Std: {self.data[y_col].std():.2f}"
                fig.add_annotation(
                    text=stats_text,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=12)
                )
            
            fig.update_layout(
                xaxis_title=x_col if x_col else '',
                yaxis_title=y_col,
                showlegend=bool(color_col)
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'violin',
                'description': f'Violin plot showing distribution of {y_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating violin plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_density_plot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create density plot using Plotly"""
        try:
            x_col = config.get('x_col')
            color_col = config.get('color_col')
            title = config.get('title')
            theme = config.get('theme', 'dark')
            
            if not x_col or x_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing x column'}
            
            if not pd.api.types.is_numeric_dtype(self.data[x_col]):
                return {'success': False, 'error': 'X column must be numeric for density plot'}
            
            # Create density plot using histogram with high bins
            fig = px.histogram(
                self.data,
                x=x_col,
                color=color_col if color_col and color_col in self.data.columns else None,
                title=title or f'Density Plot of {x_col}',
                template=self._get_theme_template(theme),
                color_discrete_sequence=self._get_color_palette(theme),
                nbins=50,
                histnorm='probability density'
            )
            
            # Add KDE curve using histogram
            clean_data = self.data[x_col].dropna()
            if len(clean_data) > 0:
                from scipy.stats import gaussian_kde
                
                try:
                    kde = gaussian_kde(clean_data)
                    x_range = np.linspace(clean_data.min(), clean_data.max(), 100)
                    kde_values = kde(x_range)
                    
                    fig.add_scatter(
                        x=x_range,
                        y=kde_values,
                        mode='lines',
                        name='KDE',
                        line=dict(color='red', width=3)
                    )
                except:
                    pass
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title='Density',
                showlegend=bool(color_col or 'KDE' in [trace.name for trace in fig.data])
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'density',
                'description': f'Density plot of {x_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating density plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_3d_scatter(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create 3D scatter plot"""
        try:
            x_col = config.get('x_col')
            y_col = config.get('y_col')
            z_col = config.get('z_col')
            color_col = config.get('color_col')
            size_col = config.get('size_col')
            title = config.get('title')
            theme = config.get('theme', 'dark')
            
            required_cols = [x_col, y_col, z_col]
            missing_cols = [col for col in required_cols if not col or col not in self.data.columns]
            
            if missing_cols:
                return {'success': False, 'error': f'Missing required columns: {missing_cols}'}
            
            fig = px.scatter_3d(
                self.data,
                x=x_col, y=y_col, z=z_col,
                color=color_col if color_col and color_col in self.data.columns else None,
                size=size_col if size_col and size_col in self.data.columns else None,
                title=title or f'3D Scatter: {x_col}, {y_col}, {z_col}',
                template=self._get_theme_template(theme),
                color_discrete_sequence=self._get_color_palette(theme)
            )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col
                )
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': '3d_scatter',
                'description': f'3D scatter plot of {x_col}, {y_col}, and {z_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating 3D scatter plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_pie_chart(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create pie chart"""
        try:
            values_col = config.get('values_col') or config.get('x_col')
            names_col = config.get('names_col') or config.get('color_col')
            title = config.get('title')
            theme = config.get('theme', 'dark')
            
            if not values_col or values_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing values column'}
            
            # Prepare data
            if names_col and names_col in self.data.columns:
                # Group by names column
                if pd.api.types.is_numeric_dtype(self.data[values_col]):
                    pie_data = self.data.groupby(names_col)[values_col].sum().reset_index()
                else:
                    pie_data = self.data[names_col].value_counts().reset_index()
                    pie_data.columns = [names_col, 'count']
                    values_col = 'count'
            else:
                # Use value counts of the values column
                pie_data = self.data[values_col].value_counts().reset_index()
                pie_data.columns = [values_col, 'count']
                names_col = values_col
                values_col = 'count'
            
            # Limit to top categories
            if len(pie_data) > 10:
                pie_data = pie_data.head(10)
                # Add "Others" category
                others_sum = self.data[values_col].sum() - pie_data[values_col].sum()
                if others_sum > 0:
                    others_row = pd.DataFrame({names_col: ['Others'], values_col: [others_sum]})
                    pie_data = pd.concat([pie_data, others_row], ignore_index=True)
            
            fig = px.pie(
                pie_data,
                values=values_col,
                names=names_col,
                title=title or f'Distribution of {names_col}',
                template=self._get_theme_template(theme),
                color_discrete_sequence=self._get_color_palette(theme)
            )
            
            # Add percentage and values
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'pie',
                'description': f'Pie chart showing distribution of {names_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_donut_chart(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create donut chart"""
        try:
            # Create pie chart first
            result = self._create_pie_chart(config)
            if not result['success']:
                return result
            
            # Convert to donut by adding hole
            fig_dict = result['figure']
            
            # Update traces to add hole
            for trace in fig_dict['data']:
                if trace['type'] == 'pie':
                    trace['hole'] = 0.3
            
            # Update description
            result['chart_type'] = 'donut'
            result['description'] = result['description'].replace('Pie chart', 'Donut chart')
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating donut chart: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_sunburst(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create sunburst chart"""
        try:
            path_col = config.get('path_col') or config.get('x_col')
            values_col = config.get('values_col') or config.get('y_col')
            title = config.get('title')
            theme = config.get('theme', 'dark')
            
            if not path_col or path_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing path column'}
            
            # Prepare data for sunburst
            if values_col and values_col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[values_col]):
                sunburst_data = self.data.groupby(path_col)[values_col].sum().reset_index()
                values = sunburst_data[values_col]
            else:
                sunburst_data = self.data[path_col].value_counts().reset_index()
                sunburst_data.columns = [path_col, 'count']
                values = sunburst_data['count']
            
            fig = px.sunburst(
                sunburst_data,
                path=[path_col],
                values=values,
                title=title or f'Sunburst Chart of {path_col}',
                template=self._get_theme_template(theme),
                color_discrete_sequence=self._get_color_palette(theme)
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'sunburst',
                'description': f'Sunburst chart of {path_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating sunburst chart: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_treemap(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create treemap"""
        try:
            path_col = config.get('path_col') or config.get('x_col')
            values_col = config.get('values_col') or config.get('y_col')
            title = config.get('title')
            theme = config.get('theme', 'dark')
            
            if not path_col or path_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing path column'}
            
            # Prepare data
            if values_col and values_col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[values_col]):
                treemap_data = self.data.groupby(path_col)[values_col].sum().reset_index()
                values = treemap_data[values_col]
            else:
                treemap_data = self.data[path_col].value_counts().reset_index()
                treemap_data.columns = [path_col, 'count']
                values = treemap_data['count']
            
            fig = px.treemap(
                treemap_data,
                path=[path_col],
                values=values,
                title=title or f'Treemap of {path_col}',
                template=self._get_theme_template(theme),
                color_discrete_sequence=self._get_color_palette(theme)
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'treemap',
                'description': f'Treemap of {path_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating treemap: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_parallel_coordinates(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create parallel coordinates plot"""
        try:
            color_col = config.get('color_col')
            title = config.get('title', 'Parallel Coordinates Plot')
            theme = config.get('theme', 'dark')
            
            # Get numerical columns
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 3:
                return {'success': False, 'error': 'Need at least 3 numerical columns for parallel coordinates'}
            
            # Limit to manageable number of dimensions
            if len(numerical_cols) > 8:
                numerical_cols = numerical_cols[:8]
            
            fig = px.parallel_coordinates(
                self.data,
                dimensions=numerical_cols,
                color=color_col if color_col and color_col in self.data.columns else numerical_cols[0],
                title=title,
                template=self._get_theme_template(theme),
                color_continuous_scale=self._get_color_palette(theme)
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'parallel_coordinates',
                'description': f'Parallel coordinates plot of {len(numerical_cols)} numerical variables'
            }
            
        except Exception as e:
            logger.error(f"Error creating parallel coordinates plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_radar_chart(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create radar chart"""
        try:
            group_col = config.get('group_col') or config.get('color_col')
            title = config.get('title', 'Radar Chart')
            theme = config.get('theme', 'dark')
            
            if not group_col or group_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing grouping column'}
            
            # Get numerical columns
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 3:
                return {'success': False, 'error': 'Need at least 3 numerical columns for radar chart'}
            
            # Limit dimensions
            numerical_cols = numerical_cols[:6]
            
            # Calculate mean values for each group
            radar_data = self.data.groupby(group_col)[numerical_cols].mean()
            
            # Normalize data to 0-1 scale
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(radar_data.T).T
            
            fig = go.Figure()
            
            colors = self._get_color_palette(theme)
            
            for idx, (group_name, group_data) in enumerate(radar_data.iterrows()):
                color_idx = idx % len(colors)
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_data[idx].tolist() + [normalized_data[idx][0]],  # Close the polygon
                    theta=numerical_cols + [numerical_cols[0]],  # Close the polygon
                    fill='toself',
                    name=str(group_name),
                    line_color=colors[color_idx]
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title=title,
                template=self._get_theme_template(theme)
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'radar',
                'description': f'Radar chart showing patterns by {group_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_pairplot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create pair plot matrix"""
        try:
            color_col = config.get('color_col')
            title = config.get('title', 'Pair Plot Matrix')
            theme = config.get('theme', 'dark')
            
            # Get numerical columns
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 2:
                return {'success': False, 'error': 'Need at least 2 numerical columns for pair plot'}
            
            # Limit to 5 columns to avoid overcrowding
            if len(numerical_cols) > 5:
                numerical_cols = numerical_cols[:5]
            
            fig = px.scatter_matrix(
                self.data,
                dimensions=numerical_cols,
                color=color_col if color_col and color_col in self.data.columns else None,
                title=title,
                template=self._get_theme_template(theme),
                color_discrete_sequence=self._get_color_palette(theme)
            )
            
            fig.update_traces(diagonal_visible=False)
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'pairplot',
                'description': f'Pair plot matrix of {len(numerical_cols)} numerical variables'
            }
            
        except Exception as e:
            logger.error(f"Error creating pair plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_distribution_plot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive distribution analysis plot"""
        try:
            x_col = config.get('x_col')
            title = config.get('title')
            theme = config.get('theme', 'dark')
            
            if not x_col or x_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing x column'}
            
            if not pd.api.types.is_numeric_dtype(self.data[x_col]):
                return {'success': False, 'error': 'Column must be numeric for distribution analysis'}
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram with KDE', 'Box Plot', 'Q-Q Plot', 'Violin Plot'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            clean_data = self.data[x_col].dropna()
            
            if len(clean_data) < 2:
                return {'success': False, 'error': 'Insufficient data points'}
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=clean_data, name='Histogram', showlegend=False, opacity=0.7),
                row=1, col=1
            )
            
            # Add KDE to histogram
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(clean_data)
                x_range = np.linspace(clean_data.min(), clean_data.max(), 100)
                kde_values = kde(x_range)
                # Scale KDE to match histogram
                kde_scaled = kde_values * len(clean_data) * (clean_data.max() - clean_data.min()) / 50
                
                fig.add_trace(
                    go.Scatter(x=x_range, y=kde_scaled, mode='lines', name='KDE', 
                              line=dict(color='red', width=2), showlegend=False),
                    row=1, col=1
                )
            except:
                pass
            
            # Box plot
            fig.add_trace(
                go.Box(y=clean_data, name='Box Plot', showlegend=False),
                row=1, col=2
            )
            
            # Q-Q plot
            try:
                from scipy import stats as scipy_stats
                theoretical_quantiles, sample_quantiles = scipy_stats.probplot(clean_data, dist="norm")[:2]
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode='markers',
                        name='Q-Q Plot',
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                # Add diagonal line
                min_val = min(min(theoretical_quantiles), min(sample_quantiles))
                max_val = max(max(theoretical_quantiles), max(sample_quantiles))
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Normal Line',
                        line=dict(color='red', dash='dash'),
                        showlegend=False
                    ),
                    row=2, col=1
                )
            except:
                pass
            
            # Violin plot
            fig.add_trace(
                go.Violin(y=clean_data, name='Violin Plot', showlegend=False, box_visible=True),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=title or f'Distribution Analysis of {x_col}',
                template=self._get_theme_template(theme),
                showlegend=False
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'distribution_analysis',
                'description': f'Comprehensive distribution analysis of {x_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_waterfall_chart(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create waterfall chart"""
        try:
            x_col = config.get('x_col')
            y_col = config.get('y_col')
            title = config.get('title', 'Waterfall Chart')
            theme = config.get('theme', 'dark')
            
            if not x_col or not y_col or x_col not in self.data.columns or y_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing x/y columns'}
            
            if not pd.api.types.is_numeric_dtype(self.data[y_col]):
                return {'success': False, 'error': 'Y column must be numeric for waterfall chart'}
            
            # Prepare data
            waterfall_data = self.data[[x_col, y_col]].copy()
            waterfall_data = waterfall_data.groupby(x_col)[y_col].sum().reset_index()
            
            # Create waterfall chart
            fig = go.Figure(go.Waterfall(
                name="Waterfall",
                orientation="v",
                measure=["relative"] * len(waterfall_data),
                x=waterfall_data[x_col],
                y=waterfall_data[y_col],
                text=[f"{val:.2f}" for val in waterfall_data[y_col]],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title=title,
                template=self._get_theme_template(theme),
                showlegend=False
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'waterfall',
                'description': f'Waterfall chart of {y_col} by {x_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating waterfall chart: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_funnel_chart(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create funnel chart"""
        try:
            x_col = config.get('x_col')
            y_col = config.get('y_col')
            title = config.get('title', 'Funnel Chart')
            theme = config.get('theme', 'dark')
            
            if not x_col or not y_col or x_col not in self.data.columns or y_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing x/y columns'}
            
            if not pd.api.types.is_numeric_dtype(self.data[y_col]):
                return {'success': False, 'error': 'Y column must be numeric for funnel chart'}
            
            # Prepare data
            funnel_data = self.data[[x_col, y_col]].copy()
            funnel_data = funnel_data.groupby(x_col)[y_col].sum().reset_index()
            funnel_data = funnel_data.sort_values(y_col, ascending=False)
            
            fig = go.Figure(go.Funnel(
                y=funnel_data[x_col],
                x=funnel_data[y_col],
                textinfo="value+percent initial"
            ))
            
            fig.update_layout(
                title=title,
                template=self._get_theme_template(theme)
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'funnel',
                'description': f'Funnel chart of {y_col} by {x_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating funnel chart: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_gauge_chart(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create gauge chart"""
        try:
            value_col = config.get('value_col') or config.get('y_col')
            title = config.get('title', 'Gauge Chart')
            theme = config.get('theme', 'dark')
            
            if not value_col or value_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing value column'}
            
            if not pd.api.types.is_numeric_dtype(self.data[value_col]):
                return {'success': False, 'error': 'Value column must be numeric for gauge chart'}
            
            # Calculate gauge value (mean)
            gauge_value = self.data[value_col].mean()
            min_val = self.data[value_col].min()
            max_val = self.data[value_col].max()
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=gauge_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title},
                delta={'reference': (min_val + max_val) / 2},
                gauge={'axis': {'range': [min_val, max_val]},
                       'bar': {'color': self._get_color_palette(theme)[0]},
                       'steps': [{'range': [min_val, (min_val + max_val) / 2], 'color': "lightgray"},
                                {'range': [(min_val + max_val) / 2, max_val], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': gauge_value * 0.9}}))
            
            fig.update_layout(
                template=self._get_theme_template(theme),
                font={'color': "white" if theme == 'dark' else "black", 'family': "Arial"}
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'gauge',
                'description': f'Gauge chart showing average {value_col}: {gauge_value:.2f}'
            }
            
        except Exception as e:
            logger.error(f"Error creating gauge chart: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_surface_plot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create 3D surface plot"""
        try:
            x_col = config.get('x_col')
            y_col = config.get('y_col')
            z_col = config.get('z_col')
            title = config.get('title', '3D Surface Plot')
            theme = config.get('theme', 'dark')
            
            required_cols = [x_col, y_col, z_col]
            missing_cols = [col for col in required_cols if not col or col not in self.data.columns]
            
            if missing_cols:
                return {'success': False, 'error': f'Missing required columns: {missing_cols}'}
            
            # Check if all columns are numeric
            for col in [x_col, y_col, z_col]:
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    return {'success': False, 'error': f'Column {col} must be numeric for surface plot'}
            
            # Create grid for surface
            clean_data = self.data[[x_col, y_col, z_col]].dropna()
            
            if len(clean_data) < 9:  # Need at least 3x3 grid
                return {'success': False, 'error': 'Insufficient data points for surface plot'}
            
            # Create meshgrid
            x_unique = np.linspace(clean_data[x_col].min(), clean_data[x_col].max(), 20)
            y_unique = np.linspace(clean_data[y_col].min(), clean_data[y_col].max(), 20)
            
            # Interpolate Z values
            from scipy.interpolate import griddata
            
            points = clean_data[[x_col, y_col]].values
            values = clean_data[z_col].values
            
            X, Y = np.meshgrid(x_unique, y_unique)
            Z = griddata(points, values, (X, Y), method='cubic')
            
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
            
            fig.update_layout(
                title=title,
                template=self._get_theme_template(theme),
                scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col
                )
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'surface',
                'description': f'3D surface plot of {z_col} over {x_col} and {y_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating surface plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_contour_plot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create contour plot"""
        try:
            x_col = config.get('x_col')
            y_col = config.get('y_col')
            z_col = config.get('z_col')
            title = config.get('title', 'Contour Plot')
            theme = config.get('theme', 'dark')
            
            required_cols = [x_col, y_col, z_col]
            missing_cols = [col for col in required_cols if not col or col not in self.data.columns]
            
            if missing_cols:
                return {'success': False, 'error': f'Missing required columns: {missing_cols}'}
            
            # Check if all columns are numeric
            for col in [x_col, y_col, z_col]:
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    return {'success': False, 'error': f'Column {col} must be numeric for contour plot'}
            
            # Create grid for contour
            clean_data = self.data[[x_col, y_col, z_col]].dropna()
            
            if len(clean_data) < 9:
                return {'success': False, 'error': 'Insufficient data points for contour plot'}
            
            # Create meshgrid and interpolate
            from scipy.interpolate import griddata
            
            x_unique = np.linspace(clean_data[x_col].min(), clean_data[x_col].max(), 50)
            y_unique = np.linspace(clean_data[y_col].min(), clean_data[y_col].max(), 50)
            
            points = clean_data[[x_col, y_col]].values
            values = clean_data[z_col].values
            
            X, Y = np.meshgrid(x_unique, y_unique)
            Z = griddata(points, values, (X, Y), method='cubic')
            
            fig = go.Figure(data=go.Contour(x=x_unique, y=y_unique, z=Z, colorscale='Viridis'))
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=clean_data[x_col],
                y=clean_data[y_col],
                mode='markers',
                marker=dict(color='red', size=4),
                name='Data Points'
            ))
            
            fig.update_layout(
                title=title,
                template=self._get_theme_template(theme),
                xaxis_title=x_col,
                yaxis_title=y_col
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'contour',
                'description': f'Contour plot of {z_col} over {x_col} and {y_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating contour plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_candlestick_chart(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create candlestick chart (for financial data)"""
        try:
            date_col = config.get('x_col')
            open_col = config.get('open_col', 'Open')
            high_col = config.get('high_col', 'High')
            low_col = config.get('low_col', 'Low')
            close_col = config.get('close_col', 'Close')
            title = config.get('title', 'Candlestick Chart')
            theme = config.get('theme', 'dark')
            
            required_cols = [date_col, open_col, high_col, low_col, close_col]
            missing_cols = [col for col in required_cols if not col or col not in self.data.columns]
            
            if missing_cols:
                return {'success': False, 'error': f'Missing required columns: {missing_cols}'}
            
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
                try:
                    self.data[date_col] = pd.to_datetime(self.data[date_col])
                except:
                    return {'success': False, 'error': f'Cannot convert {date_col} to datetime'}
            
            fig = go.Figure(data=go.Candlestick(
                x=self.data[date_col],
                open=self.data[open_col],
                high=self.data[high_col],
                low=self.data[low_col],
                close=self.data[close_col]
            ))
            
            fig.update_layout(
                title=title,
                template=self._get_theme_template(theme),
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'candlestick',
                'description': f'Candlestick chart for financial data'
            }
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_ohlc_chart(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create OHLC (Open-High-Low-Close) chart"""
        try:
            date_col = config.get('x_col')
            open_col = config.get('open_col', 'Open')
            high_col = config.get('high_col', 'High')
            low_col = config.get('low_col', 'Low')
            close_col = config.get('close_col', 'Close')
            title = config.get('title', 'OHLC Chart')
            theme = config.get('theme', 'dark')
            
            required_cols = [date_col, open_col, high_col, low_col, close_col]
            missing_cols = [col for col in required_cols if not col or col not in self.data.columns]
            
            if missing_cols:
                return {'success': False, 'error': f'Missing required columns: {missing_cols}'}
            
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
                try:
                    self.data[date_col] = pd.to_datetime(self.data[date_col])
                except:
                    return {'success': False, 'error': f'Cannot convert {date_col} to datetime'}
            
            fig = go.Figure(data=go.Ohlc(
                x=self.data[date_col],
                open=self.data[open_col],
                high=self.data[high_col],
                low=self.data[low_col],
                close=self.data[close_col]
            ))
            
            fig.update_layout(
                title=title,
                template=self._get_theme_template(theme),
                xaxis_title='Date',
                yaxis_title='Price'
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'ohlc',
                'description': f'OHLC chart for financial data'
            }
            
        except Exception as e:
            logger.error(f"Error creating OHLC chart: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_network_plot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create network plot"""
        try:
            if not NETWORKX_AVAILABLE:
                return {'success': False, 'error': 'NetworkX library not available for network plots'}
            
            source_col = config.get('source_col') or config.get('x_col')
            target_col = config.get('target_col') or config.get('y_col')
            weight_col = config.get('weight_col')
            title = config.get('title', 'Network Plot')
            theme = config.get('theme', 'dark')
            
            if not source_col or not target_col or source_col not in self.data.columns or target_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing source/target columns'}
            
            # Create network graph
            G = nx.Graph()
            
            # Add edges
            for _, row in self.data.iterrows():
                weight = row[weight_col] if weight_col and weight_col in self.data.columns else 1
                G.add_edge(row[source_col], row[target_col], weight=weight)
            
            if len(G.nodes) == 0:
                return {'success': False, 'error': 'No network nodes found'}
            
            # Calculate layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Create edge traces
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                   line=dict(width=0.5, color='#888'),
                                   hoverinfo='none',
                                   mode='lines')
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node))
                node_size.append(G.degree[node] * 10 + 10)  # Size by degree
            
            node_trace = go.Scatter(x=node_x, y=node_y,
                                   mode='markers+text',
                                   text=node_text,
                                   textposition='middle center',
                                   hoverinfo='text',
                                   marker=dict(size=node_size,
                                             color=node_size,
                                             colorscale='Viridis',
                                             line=dict(width=2)))
            
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title=title,
                               template=self._get_theme_template(theme),
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               annotations=[ dict(
                                   text="Network plot showing relationships",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor='left', yanchor='bottom',
                                   font=dict(size=12))],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'network',
                'description': f'Network plot with {len(G.nodes)} nodes and {len(G.edges)} edges'
            }
            
        except Exception as e:
            logger.error(f"Error creating network plot: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_sankey_diagram(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Sankey diagram"""
        try:
            source_col = config.get('source_col') or config.get('x_col')
            target_col = config.get('target_col') or config.get('y_col')
            value_col = config.get('value_col')
            title = config.get('title', 'Sankey Diagram')
            theme = config.get('theme', 'dark')
            
            if not source_col or not target_col or source_col not in self.data.columns or target_col not in self.data.columns:
                return {'success': False, 'error': 'Invalid or missing source/target columns'}
            
            # Prepare data
            if value_col and value_col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[value_col]):
                sankey_data = self.data.groupby([source_col, target_col])[value_col].sum().reset_index()
            else:
                sankey_data = self.data.groupby([source_col, target_col]).size().reset_index(name='count')
                value_col = 'count'
            
            # Create node labels
            all_nodes = list(set(sankey_data[source_col].unique()) | set(sankey_data[target_col].unique()))
            node_dict = {node: idx for idx, node in enumerate(all_nodes)}
            
            # Create links
            source_indices = [node_dict[source] for source in sankey_data[source_col]]
            target_indices = [node_dict[target] for target in sankey_data[target_col]]
            values = sankey_data[value_col].tolist()
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=self._get_color_palette(theme)[:len(all_nodes)]
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values
                ))])
            
            fig.update_layout(
                title_text=title,
                template=self._get_theme_template(theme),
                font_size=10
            )
            
            return {
                'success': True,
                'figure': fig.to_dict(),
                'chart_type': 'sankey',
                'description': f'Sankey diagram showing flow from {source_col} to {target_col}'
            }
            
        except Exception as e:
            logger.error(f"Error creating Sankey diagram: {str(e)}")
            return {'success': False, 'error': str(e)}

# Flask Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        # Try to read the HTML file
        html_file_path = os.path.join(os.path.dirname(__file__), 'index.html')
        
        if os.path.exists(html_file_path):
            with open(html_file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            # Return a basic HTML page if file not found
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI-Powered EDA Platform</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
            <body>
                <h1>AI-Powered EDA Platform</h1>
                <p>Welcome to the AI-Powered Exploratory Data Analysis Platform!</p>
                <p>Please ensure 'index.html' is in the same directory as this script.</p>
                <p>Current working directory: """ + os.getcwd() + """</p>
                <p>Looking for file at: """ + html_file_path + """</p>
            </body>
            </html>
            """
    except Exception as e:
        return f"Error loading page: {str(e)}", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initial processing"""
    try:
        logger.info("File upload request received")
        
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        files = request.files.getlist('files')
        session_id = request.form.get('session_id')
        
        if not session_id or not SecurityManager.validate_session_id(session_id):
            return jsonify({'success': False, 'error': 'Invalid session ID'})
        
        if not files or files[0].filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        file = files[0]
        
        # Validate file type
        if not SecurityManager.validate_file_type(file.filename):
            return jsonify({'success': False, 'error': 'Unsupported file type'})
        
        # Get file extension
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        # Save file
        filename = f"{session_id}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        logger.info(f"File saved: {file_path}")
        
        # Initialize data analyzer
        analyzer = DataAnalyzer(session_id)
        
        # Load and analyze data
        result = analyzer.load_data(file_path, file_ext)
        
        if not result['success']:
            # Clean up file on failure
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify(result)
        
        # Store analyzer in session
        session_data[session_id] = {
            'analyzer': analyzer,
            'file_path': file_path,
            'filename': file.filename,
            'upload_time': datetime.now().isoformat(),
            'file_size': os.path.getsize(file_path)
        }
        
        # Generate preview
        preview_result = analyzer.get_data_preview('head', 10)
        
        # Calculate file size
        file_size = os.path.getsize(file_path)
        
        response_data = {
            'success': True,
            'data': result['data'][:100],  # Limit data sent to client
            'columns': result['columns'],
            'file_info': {
                'name': file.filename,
                'size': f"{file_size / (1024*1024):.2f} MB",
                'type': file_ext.upper(),
                'rows': result['shape'][0],
                'columns': result['shape'][1],
                'memory_usage': analyzer.metadata.get('memory_usage', 0)
            },
            'preview': {
                'data': preview_result.get('data', []),
                'columns': preview_result.get('columns', [])
            },
            'stats': {
                'rows': result['shape'][0],
                'columns': result['shape'][1],
                'missing_values': sum(analyzer.metadata.get('missing_values', {}).values()),
                'duplicates': analyzer.metadata.get('duplicated_rows', 0)
            },
            'data_quality': analyzer.metadata.get('data_quality_score', 0)
        }
        
        logger.info(f"Upload successful for session {session_id}")
        
        return jsonify(convert_np(response_data))
        
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/preview', methods=['POST'])
def get_preview():
    """Get data preview"""
    try:
        session_id = request.form.get('session_id')
        preview_type = request.form.get('type', 'head')
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        result = analyzer.get_data_preview(preview_type, 10)
        
        return jsonify(convert_np({
            'success': True,
            'preview': {
                'data': result.get('data', []),
                'columns': result.get('columns', [])
            }
        }))
        
    except Exception as e:
        logger.error(f"Error in preview: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/data', methods=['POST'])
def clean_data():
    """Apply data cleaning operations"""
    try:
        session_id = request.form.get('session_id')
        options = json.loads(request.form.get('options', '{}'))
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        result = analyzer.clean_data(options)
        
        if result['success']:
            # Update preview and stats
            preview_result = analyzer.get_data_preview('head', 10)
            
            return jsonify(convert_np({
                'success': True,
                'data': result['data'][:100],
                'columns': result['columns'],
                'preview': {
                    'data': preview_result.get('data', []),
                    'columns': preview_result.get('columns', [])
                },
                'stats': {
                    'rows': analyzer.data.shape[0],
                    'columns': analyzer.data.shape[1],
                    'missing_values': analyzer.data.isnull().sum().sum(),
                    'duplicates': analyzer.data.duplicated().sum()
                    

                }
            }))
        else:
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error in clean_data: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    """Start comprehensive EDA analysis"""
    try:
        session_id = request.form.get('session_id')
        options = json.loads(request.form.get('options', '{}'))
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        columns = options.get('columns', list(analyzer.data.columns))
        
        # Perform comprehensive EDA
        eda_result = analyzer.perform_comprehensive_eda(columns)
        
        if eda_result['success']:
            # Format overview data for frontend
            overview_data = eda_result['results']['basic_info']
            column_info = []
            
            for col in analyzer.data.columns:
                col_dtype = str(analyzer.data[col].dtype)
                missing_count = analyzer.data[col].isnull().sum()
                missing_pct = missing_count / len(analyzer.data) * 100
                
                # Determine missing level
                if missing_pct > 20:
                    missing_level = 'high'
                elif missing_pct > 5:
                    missing_level = 'medium'
                else:
                    missing_level = 'low'
                
                column_info.append({
                    'name': col,
                    'type': col_dtype,
                    'non_null_count': int(analyzer.data[col].count()),
                    'missing_percent': float(missing_pct),
                    'unique_values': int(analyzer.data[col].nunique()),
                    'memory_usage': int(analyzer.data[col].memory_usage(deep=True)),
                    'missing_level': missing_level
                })
            
            # Data types distribution
            dtypes_dist = analyzer.data.dtypes.value_counts().to_dict()
            dtypes_dist = {str(k): int(v) for k, v in dtypes_dist.items()}
            
            return jsonify(convert_np({
                'success': True,
                'overview': {
                    'basic_stats': overview_data,
                    'column_info': column_info,
                    'data_types': dtypes_dist
                },
                'insights': eda_result['results'].get('insights', [])
            }))
        else:
            return jsonify(eda_result)
            
    except Exception as e:
        logger.error(f"Error in start_analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/univariate_analysis', methods=['POST'])
def univariate_analysis():
    """Perform univariate analysis on selected column"""
    try:
        session_id = request.form.get('session_id')
        options = json.loads(request.form.get('options', '{}'))
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        column = options.get('column')
        
        if not column or column not in analyzer.data.columns:
            return jsonify({'success': False, 'error': 'Invalid column specified'})
        
        # Perform univariate analysis
        univariate_results = analyzer._perform_univariate_analysis([column])
        
        if column in univariate_results:
            col_result = univariate_results[column]
            
            # Generate visualizations based on data type
            visualizations = []
            
            if analyzer.data[column].dtype in ['int64', 'int32', 'float64', 'float32']:
                # Numerical column
                viz_engine = VisualizationEngine(analyzer.data)
                
                # Histogram
                hist_result = viz_engine._create_histogram({
                    'x_col': column,
                    'title': f'Distribution of {column}',
                    'theme': 'dark'
                })
                if hist_result['success']:
                    visualizations.append(hist_result)
                
                # Box plot
                box_result = viz_engine._create_boxplot({
                    'y_col': column,
                    'title': f'Box Plot of {column}',
                    'theme': 'dark'
                })
                if box_result['success']:
                    visualizations.append(box_result)
            
            return jsonify(convert_np({
                'success': True,
                'results': {
                    'statistics': col_result,
                    'plots': visualizations,
                    'insights': [
                        f"Column '{column}' analysis completed",
                        f"Data type: {col_result.get('dtype', 'unknown')}",
                        f"Missing values: {col_result.get('missing_count', 0)} ({col_result.get('missing_percentage', 0):.1f}%)",
                        f"Unique values: {col_result.get('unique_count', 0)}"
                    ]
                }
            }))
        else:
            return jsonify({'success': False, 'error': 'Analysis failed'})
            
    except Exception as e:
        logger.error(f"Error in univariate_analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/bivariate_analysis', methods=['POST'])
def bivariate_analysis():
    """Perform bivariate analysis on selected columns"""
    try:
        session_id = request.form.get('session_id')
        options = json.loads(request.form.get('options', '{}'))
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        x_column = options.get('x_column')
        y_column = options.get('y_column')
        
        if not x_column or not y_column or x_column not in analyzer.data.columns or y_column not in analyzer.data.columns:
            return jsonify({'success': False, 'error': 'Invalid columns specified'})
        
        # Perform bivariate analysis
        bivariate_results = analyzer._perform_bivariate_analysis([x_column, y_column])
        
        # Generate appropriate visualization
        viz_engine = VisualizationEngine(analyzer.data)
        
        # Choose visualization based on data types
        x_numeric = pd.api.types.is_numeric_dtype(analyzer.data[x_column])
        y_numeric = pd.api.types.is_numeric_dtype(analyzer.data[y_column])
        
        visualizations = []
        correlation_result = None
        
        if x_numeric and y_numeric:
            # Both numeric - scatter plot
            scatter_result = viz_engine._create_scatterplot({
                'x_col': x_column,
                'y_col': y_column,
                'title': f'{x_column} vs {y_column}',
                'theme': 'dark'
            })
            if scatter_result['success']:
                visualizations.append(scatter_result)
            
            # Calculate correlation
            correlation = analyzer.data[[x_column, y_column]].corr().iloc[0, 1]
            correlation_result = {
                'coefficient': float(correlation),
                'p_value': 0.05,  # Placeholder
                'interpretation': f"Correlation between {x_column} and {y_column}: {correlation:.3f}"
            }
        
        elif x_numeric and not y_numeric:
            # X numeric, Y categorical - box plot
            box_result = viz_engine._create_boxplot({
                'x_col': y_column,
                'y_col': x_column,
                'title': f'{x_column} by {y_column}',
                'theme': 'dark'
            })
            if box_result['success']:
                visualizations.append(box_result)
        
        elif not x_numeric and y_numeric:
            # X categorical, Y numeric - box plot
            box_result = viz_engine._create_boxplot({
                'x_col': x_column,
                'y_col': y_column,
                'title': f'{y_column} by {x_column}',
                'theme': 'dark'
            })
            if box_result['success']:
                visualizations.append(box_result)
        
        else:
            # Both categorical - stacked bar or heatmap
            bar_result = viz_engine._create_barplot({
                'x_col': x_column,
                'color_col': y_column,
                'title': f'{x_column} by {y_column}',
                'theme': 'dark'
            })
            if bar_result['success']:
                visualizations.append(bar_result)
        
        insights = [
            f"Bivariate analysis between {x_column} and {y_column}",
            f"X variable type: {'Numeric' if x_numeric else 'Categorical'}",
            f"Y variable type: {'Numeric' if y_numeric else 'Categorical'}"
        ]
        
        if correlation_result:
            insights.append(f"Correlation: {correlation_result['coefficient']:.3f}")
        
        return jsonify(convert_np({
            'success': True,
            'results': {
                'correlation': correlation_result,
                'plots': visualizations,
                'insights': insights,
                'analysis_details': bivariate_results
            }
        }))
        
    except Exception as e:
        logger.error(f"Error in bivariate_analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/multivariate_analysis', methods=['POST'])
def multivariate_analysis():
    """Perform multivariate analysis"""
    try:
        session_id = request.form.get('session_id')
        options = json.loads(request.form.get('options', '{}'))
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        data = analyzer.data
        
        results = {}
        insights = []
        
        # PCA Analysis
        if options.get('show_pca'):
            try:
                numerical_cols = data.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 1:
                    # Prepare data
                    pca_data = data[numerical_cols].dropna()
                    
                    if len(pca_data) > 0:
                        # Standardize data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(pca_data)
                        
                        # Apply PCA
                        n_components = min(5, len(numerical_cols), len(pca_data))
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(scaled_data)
                        
                        results['pca'] = {
                            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                            'components': pca.components_.tolist(),
                            'n_components': int(n_components),
                            'feature_names': list(numerical_cols)
                        }
                        
                        insights.append(f"PCA: First {n_components} components explain {pca.explained_variance_ratio_[:n_components].sum():.1%} of variance")
                        
            except Exception as e:
                results['pca'] = {'error': str(e)}
        
        # t-SNE Analysis
        if options.get('show_tsne'):
            try:
                numerical_cols = data.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 2:
                    tsne_data = data[numerical_cols].dropna()
                    
                    if len(tsne_data) > 10:
                        # Sample if too large
                        if len(tsne_data) > 1000:
                            tsne_data = tsne_data.sample(1000, random_state=42)
                        
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(tsne_data)
                        
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_data)-1))
                        tsne_result = tsne.fit_transform(scaled_data)
                        
                        results['tsne'] = {
                            'embedding': tsne_result.tolist(),
                            'perplexity': float(tsne.perplexity),
                            'n_samples': len(tsne_data)
                        }
                        
                        insights.append(f"t-SNE: 2D embedding created from {len(numerical_cols)} features")
                        
            except Exception as e:
                results['tsne'] = {'error': str(e)}
        
        # VIF Analysis
        if options.get('show_vif'):
            try:
                numerical_cols = data.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 2:
                    vif_data = data[numerical_cols].dropna()
                    
                    if len(vif_data) > 0:
                        vif_results = []
                        
                        for i, col in enumerate(numerical_cols):
                            try:
                                vif = variance_inflation_factor(vif_data.values, i)
                                vif_results.append({
                                    'feature': col,
                                    'vif': float(vif) if not np.isnan(vif) and not np.isinf(vif) else 0
                                })
                            except:
                                vif_results.append({'feature': col, 'vif': 0})
                        
                        results['vif'] = vif_results
                        
                        high_vif = [item for item in vif_results if item['vif'] > 10]
                        if high_vif:
                            insights.append(f"High multicollinearity detected in {len(high_vif)} features (VIF > 10)")
                        
            except Exception as e:
                results['vif'] = {'error': str(e)}
        
        # Clustering Analysis
        if options.get('show_cluster'):
            try:
                numerical_cols = data.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 1:
                    cluster_data = data[numerical_cols].dropna()
                    
                    if len(cluster_data) > 10:
                        # Scale data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(cluster_data)
                        
                        # K-means clustering
                        n_clusters = min(5, len(cluster_data) // 10, 8)
                        if n_clusters >= 2:
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            cluster_labels = kmeans.fit_predict(scaled_data)
                            
                            # Calculate silhouette score
                            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
                            
                            results['cluster'] = {
                                'labels': cluster_labels.tolist(),
                                'centers': kmeans.cluster_centers_.tolist(),
                                'n_clusters': int(n_clusters),
                                'silhouette_score': float(silhouette_avg),
                                'inertia': float(kmeans.inertia_)
                            }
                            
                            insights.append(f"K-means clustering: {n_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
                        
            except Exception as e:
                results['cluster'] = {'error': str(e)}
        
        # Correlation Heatmap Data
        if options.get('show_heatmap'):
            try:
                numerical_cols = data.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 1:
                    corr_matrix = data[numerical_cols].corr()
                    
                    results['heatmap'] = {
                        'correlation_matrix': corr_matrix.values.tolist(),
                        'columns': list(numerical_cols),
                        'method': 'pearson'
                    }
                    
                    # Find strong correlations
                    strong_corr_count = 0
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.7:
                                strong_corr_count += 1
                    
                    if strong_corr_count > 0:
                        insights.append(f"Found {strong_corr_count} strong correlations (|r| > 0.7)")
                        
            except Exception as e:
                results['heatmap'] = {'error': str(e)}
        
        return jsonify(convert_np({
            'success': True,
            'results': {
                **results,
                'insights': insights
            }
        }))
        
    except Exception as e:
        logger.error(f"Error in multivariate_analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/correlation_analysis', methods=['POST'])
def correlation_analysis():
    """Perform correlation analysis"""
    try:
        session_id = request.form.get('session_id')
        options = json.loads(request.form.get('options', '{}'))
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        method = options.get('method', 'pearson')
        threshold = options.get('threshold', 0.5)
        
        correlation_results = analyzer._perform_correlation_analysis()
        
        if correlation_results:
            # Filter high correlations based on threshold
            high_correlations = []
            if 'high_correlations' in correlation_results:
                high_correlations = [
                    corr for corr in correlation_results['high_correlations']
                    if abs(corr['correlation']) >= threshold
                ]
            
            # Generate correlation heatmap
            viz_engine = VisualizationEngine(analyzer.data)
            
            heatmap_result = viz_engine._create_heatmap({
                'correlation_method': method,
                'title': f'{method.title()} Correlation Matrix',
                'theme': 'dark'
            })
            
            plots = []
            if heatmap_result['success']:
                plots.append(heatmap_result)
            
            insights = [
                f"Correlation analysis using {method} method",
                f"Threshold for high correlations: {threshold}",
                f"Found {len(high_correlations)} correlations above threshold"
            ]
            
            if high_correlations:
                top_corr = max(high_correlations, key=lambda x: abs(x['correlation']))
                insights.append(f"Strongest correlation: {top_corr['variable1']} - {top_corr['variable2']} ({top_corr['correlation']:.3f})")
            
            return jsonify(convert_np({
                'success': True,
                'results': {
                    'high_correlations': high_correlations,
                    'correlation_matrix': correlation_results.get(method, {}),
                    'plots': plots,
                    'insights': insights,
                    'method': method,
                    'threshold': threshold
                }
            }))
        else:
            return jsonify({'success': False, 'error': 'Correlation analysis failed'})
            
    except Exception as e:
        logger.error(f"Error in correlation_analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/outlier_analysis', methods=['POST'])
def outlier_analysis():
    """Perform outlier analysis"""
    try:
        session_id = request.form.get('session_id')
        options = json.loads(request.form.get('options', '{}'))
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        columns = options.get('columns', list(analyzer.data.columns))
        
        outlier_results = analyzer._perform_outlier_analysis(columns)
        
        if outlier_results:
            # Generate summary
            summary = []
            total_outliers = 0
            
            for col, results in outlier_results.items():
                if col == 'summary':
                    continue
                
                if isinstance(results, dict) and 'iqr' in results:
                    iqr_count = results['iqr'].get('count', 0)
                    iqr_pct = results['iqr'].get('percentage', 0)
                    
                    summary.append({
                        'method': 'IQR',
                        'column': col,
                        'count': iqr_count,
                        'percentage': float(iqr_pct)
                    })
                    
                    total_outliers += iqr_count
            
            # Generate visualizations for top columns with outliers
            viz_engine = VisualizationEngine(analyzer.data)
            plots = []
            
            # Find columns with most outliers for visualization
            outlier_cols = []
            for col, results in outlier_results.items():
                if isinstance(results, dict) and 'iqr' in results:
                    if results['iqr'].get('count', 0) > 0:
                        outlier_cols.append((col, results['iqr']['count']))
            
            # Sort by outlier count and visualize top 3
            outlier_cols.sort(key=lambda x: x[1], reverse=True)
            
            for col, count in outlier_cols[:3]:
                if pd.api.types.is_numeric_dtype(analyzer.data[col]):
                    box_result = viz_engine._create_boxplot({
                        'y_col': col,
                        'title': f'Box Plot of {col} (Outliers Highlighted)',
                        'theme': 'dark'
                    })
                    if box_result['success']:
                        plots.append(box_result)
            
            insights = [
                f"Outlier analysis completed on {len(columns)} columns",
                f"Total outliers detected: {total_outliers}",
                f"Columns with outliers: {len([col for col, results in outlier_results.items() if isinstance(results, dict) and results.get('iqr', {}).get('count', 0) > 0])}"
            ]
            
            if outlier_cols:
                top_col = outlier_cols[0]
                insights.append(f"Column with most outliers: {top_col[0]} ({top_col[1]} outliers)")
            
            return jsonify(convert_np({
                'success': True,
                'results': {
                    'summary': summary,
                    'detailed_results': outlier_results,
                    'plots': plots,
                    'insights': insights
                }
            }))
        else:
            return jsonify({'success': False, 'error': 'Outlier analysis failed'})
            
    except Exception as e:
        logger.error(f"Error in outlier_analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/run_statistical_test', methods=['POST'])
def run_statistical_test():
    """Run statistical tests"""
    try:
        session_id = request.form.get('session_id')
        test_type = request.form.get('test_type')
        options = json.loads(request.form.get('options', '{}'))
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        tester = StatisticalTester(analyzer.data)
        
        # Route to appropriate test based on test type
        if test_type in ['shapiro_wilk', 'anderson_darling', 'kolmogorov_smirnov', 
                        'dagostino_pearson', 'jarque_bera', 'lilliefors']:
            # Normality tests
            variables = options.get('variables', [])
            alpha = options.get('alpha', 0.05)
            result = tester.run_normality_tests(variables, alpha)
            
        elif test_type in ['one_sample_t', 'two_sample_t', 'paired_t', 'welch_t', 'one_way_anova']:
            # Hypothesis tests
            var1 = options.get('var1')
            var2 = options.get('var2')
            alpha = options.get('alpha', 0.05)
            alternative = options.get('alternative', 'two-sided')
            result = tester.run_hypothesis_tests(test_type, var1, var2, alpha, alternative)
            
        elif test_type in ['pearson_correlation', 'spearman_correlation', 'kendall_tau']:
            # Correlation tests
            var1 = options.get('var1')
            var2 = options.get('var2')
            alpha = options.get('alpha', 0.05)
            result = tester.run_correlation_tests(var1, var2, alpha)
            
        elif test_type in ['chi_square', 'fisher_exact', 'g_test']:
            # Independence tests
            var1 = options.get('var1')
            var2 = options.get('var2')
            alpha = options.get('alpha', 0.05)
            result = tester.run_independence_tests(var1, var2, alpha)
            
        elif test_type in ['levene', 'bartlett', 'fligner', 'f_test']:
            # Variance tests
            variables = options.get('variables', [])
            alpha = options.get('alpha', 0.05)
            result = tester.run_variance_tests(variables, alpha)
            
        elif test_type in ['mann_whitney', 'wilcoxon_signed_rank', 'kruskal_wallis', 
                          'friedman', 'mood_median']:
            # Non-parametric tests
            variables = options.get('variables', [])
            alpha = options.get('alpha', 0.05)
            result = tester.run_nonparametric_tests(test_type, variables, alpha)
            
        else:
            return jsonify({'success': False, 'error': f'Unknown test type: {test_type}'})
        
        return jsonify(convert_np(result))
        
    except Exception as e:
        logger.error(f"Error in run_statistical_test: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_visualization', methods=['POST'])
def generate_visualization():
    """Generate visualizations"""
    try:
        session_id = request.form.get('session_id')
        options = json.loads(request.form.get('options', '{}'))
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        viz_engine = VisualizationEngine(analyzer.data)
        
        # Create configuration for visualization
        viz_config = {
            'chart_type': options.get('chart_type'),
            'x_col': options.get('x_axis'),
            'y_col': options.get('y_axis'),
            'color_col': options.get('color_by'),
            'size_col': options.get('size_by'),
            'facet_col': options.get('facet_by'),
            'theme': options.get('theme', 'dark'),
            'title': options.get('title')
        }
        
        result = viz_engine.create_comprehensive_visualization(viz_config)
        
        if result['success']:
            return jsonify(convert_np({
                'success': True,
                'plot': {
                    'type': 'plotly',
                    'data': result['figure']['data'],
                    'layout': result['figure']['layout'],
                    'config': {'responsive': True, 'displayModeBar': True},
                    'title': result['description']
                }
            }))
        else:
            return jsonify(result)
    except Exception as e:
        logger.error(f"Error in generate_visualization: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/feature_engineering', methods=['POST'])
def feature_engineering():
    """Perform feature engineering operations"""
    try:
        session_id = request.form.get('session_id')
        options = json.loads(request.form.get('options', '{}'))
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        engineer = FeatureEngineer(analyzer.data)
        
        # Apply feature engineering transformations
        result = engineer.apply_comprehensive_transformations(options)
        
        if result['success']:
            # Update the analyzer's data with engineered features
            analyzer.data = engineer.get_data()
            
            # Get feature history and transformer info
            history = engineer.get_feature_history()
            transformer_info = engineer.get_transformer_info()
            
            return jsonify(convert_np({
                'success': True,
                'results': result['results'],
                'feature_history': history,
                'transformer_info': transformer_info,
                'new_shape': analyzer.data.shape,
                'new_columns': list(analyzer.data.columns)
            }))
        else:
            return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in feature_engineering: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/export_results', methods=['POST'])
def export_results():
    """Export analysis results and visualizations"""
    try:
        session_id = request.form.get('session_id')
        export_format = request.form.get('format', 'json')
        
        if not session_id or session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        analyzer = session_data[session_id]['analyzer']
        
        # Prepare export data
        export_data = {
            'session_id': session_id,
            'export_timestamp': datetime.now().isoformat(),
            'data_info': {
                'shape': analyzer.data.shape,
                'columns': list(analyzer.data.columns),
                'dtypes': {col: str(dtype) for col, dtype in analyzer.data.dtypes.items()}
            },
            'metadata': analyzer.metadata,
            'insights': analyzer.insights,
            'transformation_history': analyzer.transformation_history
        }
        
        if export_format == 'json':
            # Create export file
            export_filename = f"eda_results_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            export_path = os.path.join(app.config['UPLOAD_FOLDER'], export_filename)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(convert_np(export_data), f, indent=2, ensure_ascii=False)
            
            return send_file(
                export_path,
                as_attachment=True,
                download_name=export_filename,
                mimetype='application/json'
            )
        
        elif export_format == 'csv':
            # Export processed data as CSV
            export_filename = f"processed_data_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            export_path = os.path.join(app.config['UPLOAD_FOLDER'], export_filename)
            
            analyzer.data.to_csv(export_path, index=False, encoding='utf-8')
            
            return send_file(
                export_path,
                as_attachment=True,
                download_name=export_filename,
                mimetype='text/csv'
            )
        
        elif export_format == 'excel':
            # Export to Excel with multiple sheets
            export_filename = f"eda_report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            export_path = os.path.join(app.config['UPLOAD_FOLDER'], export_filename)
            
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                # Data sheet
                analyzer.data.to_excel(writer, sheet_name='Data', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Rows', 'Columns', 'Missing Values', 'Duplicates', 'Data Quality Score'],
                    'Value': [
                        analyzer.data.shape[0],
                        analyzer.data.shape[1],
                        analyzer.data.isnull().sum().sum(),
                        analyzer.data.duplicated().sum(),
                        analyzer.metadata.get('data_quality_score', 0)
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Column info sheet
                col_info = []
                for col in analyzer.data.columns:
                    col_info.append({
                        'Column': col,
                        'Type': str(analyzer.data[col].dtype),
                        'Non-Null Count': analyzer.data[col].count(),
                        'Missing Count': analyzer.data[col].isnull().sum(),
                        'Unique Values': analyzer.data[col].nunique(),
                        'Memory Usage (bytes)': analyzer.data[col].memory_usage(deep=True)
                    })
                pd.DataFrame(col_info).to_excel(writer, sheet_name='Column Info', index=False)
            
            return send_file(
                export_path,
                as_attachment=True,
                download_name=export_filename,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        
        else:
            return jsonify({'success': False, 'error': f'Unsupported export format: {export_format}'})
            
    except Exception as e:
        logger.error(f"Error in export_results: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_session_info', methods=['GET'])
def get_session_info():
    """Get information about active sessions"""
    try:
        session_info = {}
        for sid, data in session_data.items():
            session_info[sid] = {
                'filename': data.get('filename', 'unknown'),
                'upload_time': data.get('upload_time', ''),
                'file_size': data.get('file_size', 0),
                'data_shape': data['analyzer'].data.shape if 'analyzer' in data else (0, 0)
            }
        
        return jsonify({
            'success': True,
            'sessions': session_info,
            'active_sessions': len(session_info)
        })
    
    except Exception as e:
        logger.error(f"Error getting session info: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/cleanup_session', methods=['POST'])
def cleanup_session():
    """Clean up a specific session"""
    try:
        session_id = request.form.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'No session ID provided'})
        
        if session_id in session_data:
            # Remove uploaded file if it exists
            if 'file_path' in session_data[session_id]:
                file_path = session_data[session_id]['file_path']
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Remove session data
            del session_data[session_id]
            
            return jsonify({
                'success': True,
                'message': f'Session {session_id} cleaned up successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Session {session_id} not found'
            })
    
    except Exception as e:
        logger.error(f"Error cleaning up session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'active_sessions': len(session_data)
    })

@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 100MB.'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error occurred'
    }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

def cleanup_old_files():
    """Clean up old uploaded files and sessions"""
    try:
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, data in session_data.items():
            upload_time_str = data.get('upload_time', '')
            if upload_time_str:
                try:
                    upload_time = datetime.fromisoformat(upload_time_str)
                    # Remove sessions older than 1 hour
                    if (current_time - upload_time).total_seconds() > app.config.get('SESSION_TIMEOUT', 3600):
                        expired_sessions.append(session_id)
                except:
                    expired_sessions.append(session_id)
        
        # Clean up expired sessions
        for session_id in expired_sessions:
            if 'file_path' in session_data[session_id]:
                file_path = session_data[session_id]['file_path']
                if os.path.exists(file_path):
                    os.remove(file_path)
            del session_data[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
        
        # Clean up orphaned files in upload directory
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if (current_time - file_time).total_seconds() > app.config.get('SESSION_TIMEOUT', 3600):
                        os.remove(file_path)
                        logger.info(f"Removed old file: {filename}")
                        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

# Background cleanup task (runs every 30 minutes)
import threading
import time

def periodic_cleanup():
    """Run periodic cleanup in background"""
    while True:
        try:
            time.sleep(1800)  # 30 minutes
            cleanup_old_files()
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {str(e)}")
            time.sleep(60)  # Wait 1 minute before retrying

# Start background cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    logger.info("Starting AI-Powered EDA Platform")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f} MB")
    
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Check available libraries
    logger.info(f"TextBlob available: {TEXTBLOB_AVAILABLE}")
    logger.info(f"NetworkX available: {NETWORKX_AVAILABLE}")
    logger.info(f"WordCloud available: {WORDCLOUD_AVAILABLE}")
    
    # Run the Flask application
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,  # Set to False in production
        threaded=True,
        use_reloader=False  # Disable reloader to prevent duplicate cleanup threads
    )


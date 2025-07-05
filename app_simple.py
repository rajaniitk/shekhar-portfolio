#!/usr/bin/env python3
"""
Simplified Flask Backend for Advanced EDA & ML Platform
Provides all API endpoints expected by the frontend with mock responses
Minimal dependencies - only uses Python standard library
"""

import json
import os
import uuid
import time
import random
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import urllib.parse

class EDARequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for EDA API endpoints"""
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_file('index.html', 'text/html')
        elif self.path == '/health':
            self.send_json_response({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            if content_length > 0:
                if 'multipart/form-data' in self.headers.get('Content-Type', ''):
                    # Handle file upload
                    request_data = self.parse_multipart(post_data)
                else:
                    # Handle JSON data
                    request_data = json.loads(post_data.decode('utf-8'))
            else:
                request_data = {}
        except:
            request_data = {}
        
        # Route API endpoints
        if path == '/api/upload':
            response = self.handle_upload(request_data)
        elif path == '/api/preprocess':
            response = self.handle_preprocess(request_data)
        elif path == '/api/eda/global':
            response = self.handle_global_eda(request_data)
        elif path == '/api/visualizations/generate':
            response = self.handle_visualization(request_data)
        elif path == '/api/statistics/test':
            response = self.handle_statistical_test(request_data)
        elif path == '/api/feature_engineering/transform':
            response = self.handle_feature_engineering(request_data)
        elif path == '/api/ml/train':
            response = self.handle_ml_training(request_data)
        elif path == '/api/reports/generate':
            response = self.handle_report_generation(request_data)
        elif path.startswith('/api/analysis/column/'):
            column_name = path.split('/')[-1]
            response = self.handle_column_analysis(column_name, request_data)
        elif path == '/api/quality/completeness':
            response = self.handle_completeness_assessment(request_data)
        elif path.startswith('/api/data_quality/'):
            analysis_type = path.split('/')[-1]
            response = self.handle_data_quality_analysis(analysis_type, request_data)
        elif path.startswith('/api/ai/'):
            feature_type = path.split('/')[-1]
            response = self.handle_ai_features(feature_type, request_data)
        elif path.startswith('/api/advanced/'):
            analysis_type = path.split('/')[-1]
            response = self.handle_advanced_analytics(analysis_type, request_data)
        elif path.startswith('/api/profiling/'):
            profiling_type = path.split('/')[-1]
            response = self.handle_data_profiling(profiling_type, request_data)
        else:
            response = {'error': 'Endpoint not found'}
            self.send_response(404)
        
        self.send_json_response(response)
    
    def parse_multipart(self, post_data):
        """Simple multipart form data parser"""
        return {'file': 'mock_file_data', 'session_id': str(uuid.uuid4())}
    
    def send_json_response(self, data):
        """Send JSON response with CORS headers"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response_json = json.dumps(data, default=str)
        self.wfile.write(response_json.encode('utf-8'))
    
    def serve_file(self, filename, content_type):
        """Serve static files"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
    
    def handle_upload(self, data):
        """Handle file upload"""
        return {
            'success': True,
            'message': 'File uploaded successfully. Shape: (1000, 15)',
            'data': {
                'head': [{'col1': i, 'col2': f'value_{i}', 'col3': random.random()} for i in range(10)],
                'tail': [{'col1': i+990, 'col2': f'value_{i+990}', 'col3': random.random()} for i in range(10)],
                'sample': [{'col1': random.randint(1, 1000), 'col2': f'sample_{i}', 'col3': random.random()} for i in range(10)]
            },
            'summary': {
                'shape': [1000, 15],
                'memory_usage': 120000,
                'dtypes': {'col1': 'int64', 'col2': 'object', 'col3': 'float64'},
                'null_counts': {'col1': 0, 'col2': 5, 'col3': 12},
                'null_percentages': {'col1': 0.0, 'col2': 0.5, 'col3': 1.2},
                'unique_counts': {'col1': 1000, 'col2': 995, 'col3': 988},
                'duplicate_rows': 3,
                'columns': ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15']
            }
        }
    
    def handle_preprocess(self, data):
        """Handle data preprocessing"""
        return {
            'success': True,
            'message': 'Preprocessing completed successfully',
            'summary': {
                'shape': [997, 15],
                'processed_operations': ['removed_duplicates', 'filled_missing'],
                'operations_applied': data.get('missing_strategy', 'unknown')
            }
        }
    
    def handle_global_eda(self, data):
        """Handle global EDA analysis"""
        return {
            'success': True,
            'results': {
                'overview': {
                    'total_rows': 1000,
                    'total_columns': 15,
                    'numerical_columns': 8,
                    'categorical_columns': 5,
                    'datetime_columns': 2,
                    'memory_usage': 120000,
                    'missing_cells': 17,
                    'duplicate_rows': 3
                },
                'missing_analysis': {
                    'total_missing': 17,
                    'missing_by_column': {'col2': 5, 'col3': 12},
                    'missing_percentages': {'col2': 0.5, 'col3': 1.2},
                    'columns_with_missing': ['col2', 'col3'],
                    'complete_rows': 983
                },
                'correlation_analysis': {
                    'high_correlations': [
                        {'var1': 'col1', 'var2': 'col4', 'correlation': 0.85},
                        {'var1': 'col3', 'var2': 'col7', 'correlation': -0.72}
                    ],
                    'max_correlation': 0.85
                },
                'outlier_analysis': {
                    'col1': {'outlier_count': 15, 'outlier_percentage': 1.5},
                    'col3': {'outlier_count': 8, 'outlier_percentage': 0.8}
                },
                'insights': [
                    'Dataset contains 1000 rows and 15 columns',
                    'Low missing data: 1.7% of values are missing',
                    'Found 3 duplicate rows (0.3%)',
                    'Data contains 8 numerical and 5 categorical columns',
                    'Strong correlation detected between col1 and col4'
                ]
            }
        }
    
    def handle_visualization(self, data):
        """Handle visualization generation"""
        viz_type = data.get('type', 'histogram')
        return {
            'success': True,
            'figure': {
                'data': [{'type': viz_type, 'name': f'Mock {viz_type}'}],
                'layout': {
                    'title': f'Mock {viz_type} Visualization',
                    'template': 'plotly_dark',
                    'xaxis': {'title': 'X Axis'},
                    'yaxis': {'title': 'Y Axis'}
                }
            }
        }
    
    def handle_statistical_test(self, data):
        """Handle statistical tests"""
        test_type = data.get('test_type', 'normality')
        return {
            'success': True,
            'results': {
                'test_type': test_type,
                'statistic': random.uniform(0.5, 2.0),
                'p_value': random.uniform(0.01, 0.1),
                'significant': random.choice([True, False]),
                'interpretation': f'{test_type} test completed successfully'
            }
        }
    
    def handle_feature_engineering(self, data):
        """Handle feature engineering"""
        transform_type = data.get('type', 'scaling')
        return {
            'success': True,
            'message': f'Applied {transform_type} transformation',
            'new_columns': [f'transformed_feature_{i}' for i in range(3)],
            'transformation_info': {
                'type': transform_type,
                'columns': data.get('columns', []),
                'parameters': data.get('parameters', {})
            }
        }
    
    def handle_ml_training(self, data):
        """Handle ML model training"""
        model_type = data.get('model_type', 'random_forest')
        return {
            'success': True,
            'model_id': str(uuid.uuid4()),
            'metrics': {
                'accuracy': round(random.uniform(0.75, 0.95), 3),
                'precision': round(random.uniform(0.70, 0.90), 3),
                'recall': round(random.uniform(0.75, 0.92), 3),
                'f1_score': round(random.uniform(0.72, 0.91), 3)
            },
            'feature_importance': {
                f'feature_{i}': round(random.random(), 3) for i in range(5)
            },
            'training_info': {
                'model_type': model_type,
                'training_samples': 800,
                'test_samples': 200,
                'feature_columns': data.get('feature_columns', [])
            }
        }
    
    def handle_report_generation(self, data):
        """Handle report generation"""
        return {
            'success': True,
            'report': {
                'dataset_info': {
                    'shape': [1000, 15],
                    'columns': [f'col_{i}' for i in range(1, 16)],
                    'memory_usage': 120000,
                    'generated_at': datetime.now().isoformat()
                },
                'data_quality': {
                    'missing_values': {'col2': 5, 'col3': 12},
                    'duplicate_rows': 3,
                    'completeness_score': 98.3
                },
                'summary_statistics': {
                    'numerical_columns': 8,
                    'categorical_columns': 5,
                    'datetime_columns': 2
                },
                'recommendations': [
                    'Consider handling missing values in col2 and col3',
                    'Remove duplicate rows if they are not meaningful',
                    'Apply feature scaling for machine learning algorithms',
                    'Strong correlations detected - consider dimensionality reduction'
                ]
            }
        }
    
    def handle_column_analysis(self, column_name, data):
        """Handle individual column analysis"""
        return {
            'success': True,
            'analysis': {
                'column_name': column_name,
                'data_type': 'float64',
                'non_null_count': 988,
                'null_count': 12,
                'unique_count': 976,
                'memory_usage': 8000,
                'mean': round(random.uniform(10, 100), 2),
                'std': round(random.uniform(5, 20), 2),
                'min': round(random.uniform(0, 10), 2),
                'max': round(random.uniform(90, 200), 2),
                'median': round(random.uniform(40, 60), 2),
                'skewness': round(random.uniform(-1, 1), 3),
                'kurtosis': round(random.uniform(-1, 3), 3)
            }
        }
    
    def handle_completeness_assessment(self, data):
        """Handle data completeness assessment"""
        return {
            'success': True,
            'completeness_score': 98.3,
            'missing_cells': 17,
            'total_cells': 15000,
            'column_completeness': {
                f'col_{i}': round(random.uniform(95, 100), 1) for i in range(1, 16)
            }
        }
    
    def handle_data_quality_analysis(self, analysis_type, data):
        """Handle data quality analysis"""
        responses = {
            'missing_patterns': {
                'total_missing': 150,
                'missing_percentage': 5.2,
                'columns_with_missing': 3,
                'complete_rows': 2850,
                'patterns': [
                    {'description': 'Random missing pattern', 'count': 100, 'percentage': 66.7},
                    {'description': 'Systematic missing pattern', 'count': 50, 'percentage': 33.3}
                ]
            },
            'consistency': {
                'overall_score': 85,
                'column_consistency': [
                    {
                        'column_name': 'email',
                        'score': 95,
                        'issues': ['Some invalid email formats'],
                        'recommendations': ['Apply email validation rules']
                    }
                ]
            },
            'validity': {
                'overall_validity': 92,
                'column_validity': [
                    {
                        'column_name': 'age',
                        'validity_score': 98,
                        'validation_rules': [
                            {'rule_name': 'Range check', 'passed': True, 'passed_count': 2950, 'total_count': 3000}
                        ],
                        'violations': []
                    }
                ]
            }
        }
        
        return {
            'success': True,
            'analysis': responses.get(analysis_type, {'placeholder': True, 'type': analysis_type})
        }
    
    def handle_ai_features(self, feature_type, data):
        """Handle AI features"""
        responses = {
            'generate_insights': {
                'key_findings': [
                    {'title': 'High Correlation Detected', 'description': 'Strong correlation between feature A and B', 'confidence': 95},
                    {'title': 'Data Quality Issue', 'description': 'Missing values in critical columns', 'confidence': 88}
                ],
                'recommendations': [
                    {'category': 'Data Quality', 'recommendation': 'Address missing values', 'priority': 'high'},
                    {'category': 'Feature Engineering', 'recommendation': 'Create interaction features', 'priority': 'medium'}
                ]
            },
            'detect_patterns': {
                'total_patterns': 5,
                'strong_patterns': 2,
                'confidence_avg': 82,
                'patterns': [
                    {'type': 'Seasonal', 'description': 'Monthly seasonality detected', 'confidence': 92, 'variables': ['sales', 'date']},
                    {'type': 'Trend', 'description': 'Upward trend identified', 'confidence': 87, 'variables': ['revenue']}
                ]
            }
        }
        
        return {
            'success': True,
            'insights' if feature_type == 'generate_insights' else feature_type: responses.get(feature_type, {'placeholder': True, 'type': feature_type})
        }
    
    def handle_advanced_analytics(self, analysis_type, data):
        """Handle advanced analytics"""
        responses = {
            'clustering': {
                'cluster_labels': [0, 1, 0, 2, 1] * 20,
                'silhouette_score': 0.75,
                'n_clusters': 3,
                'cluster_centers': [[1.2, 2.3], [4.5, 1.8], [2.1, 5.4]]
            },
            'dimensionality_reduction': {
                'reduced_data': [[i, i*2] for i in range(100)],
                'explained_variance_ratio': [0.6, 0.3],
                'method': 'PCA',
                'total_explained_variance': 0.9
            },
            'anomaly_detection': {
                'anomaly_labels': [1] * 95 + [-1] * 5,
                'anomaly_indices': [95, 96, 97, 98, 99],
                'n_anomalies': 5,
                'contamination': 0.05
            },
            'time_series': {
                'trend': 'increasing',
                'seasonality': 'monthly',
                'stationarity': 'non-stationary',
                'trend_strength': 0.75
            },
            'time_series_decomposition': {
                'trend_strength': 75,
                'seasonal_strength': 60,
                'noise_level': 15,
                'model_type': 'Additive'
            },
            'survival_analysis': {
                'total_subjects': 1000,
                'events_observed': 350,
                'median_survival': 24.5,
                'censoring_rate': 65
            }
        }
        
        return {
            'success': True,
            'results': responses.get(analysis_type, {'placeholder': True, 'type': analysis_type})
        }
    
    def handle_data_profiling(self, profiling_type, data):
        """Handle data profiling"""
        responses = {
            'schema': {
                'total_columns': 15,
                'numerical_columns': 8,
                'categorical_columns': 5,
                'datetime_columns': 2,
                'data_types': {
                    'int64': 3,
                    'float64': 5,
                    'object': 5,
                    'datetime64[ns]': 2
                }
            },
            'content': {
                'data_quality_score': 87,
                'completeness': 94,
                'consistency': 89,
                'validity': 92,
                'uniqueness': 86
            }
        }
        
        return {
            'success': True,
            'profile': responses.get(profiling_type, {'placeholder': True, 'type': profiling_type})
        }

def run_server(port=5000):
    """Run the HTTP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, EDARequestHandler)
    print(f"🚀 Advanced EDA & ML Platform started on http://localhost:{port}")
    print("📊 Ready to analyze data!")
    print("🔗 Open http://localhost:5000 in your browser")
    print("💡 All API endpoints are available and returning mock data")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()
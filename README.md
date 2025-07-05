# Advanced EDA & ML Platform

A comprehensive web application for automated Exploratory Data Analysis (EDA), statistical testing, feature engineering, and machine learning. This platform provides an intuitive dark-themed interface for data scientists and analysts to perform complex data analysis tasks with ease.

## 🚀 Features

### Data Upload & Preprocessing
- Support for multiple file formats (CSV, Excel, JSON, Parquet, TSV)
- Intelligent data type detection
- Missing value handling strategies
- Duplicate removal
- Data type conversions
- Real-time data preview and summary statistics

### Exploratory Data Analysis
- **Global EDA**: Comprehensive overview of dataset characteristics
- **Missing Values Analysis**: Detailed missing data patterns
- **Outlier Detection**: IQR and Z-score based outlier identification
- **Distribution Analysis**: Skewness, kurtosis, and normality assessment
- **Correlation Analysis**: Pearson, Spearman, and Kendall correlations
- **Data Quality Assessment**: Automated quality issue detection

### Advanced Visualizations (60+ Types)
- **Basic Plots**: Histograms, Box plots, Scatter plots, Bar charts, Pie charts
- **Statistical Plots**: Violin plots, Distribution plots, Q-Q plots
- **Correlation Visualizations**: Heatmaps, Pair plots
- **3D Visualizations**: 3D scatter plots, Surface plots
- **Time Series Plots**: Line plots with trend analysis
- **Interactive Features**: Zoom, pan, hover tooltips, customizable themes

### Statistical Testing (50+ Tests)
- **Normality Tests**: Shapiro-Wilk, D'Agostino, Kolmogorov-Smirnov, Anderson-Darling, Jarque-Bera
- **T-Tests**: One-sample, Two-sample, Paired t-tests
- **ANOVA**: One-way and Two-way ANOVA
- **Non-Parametric Tests**: Mann-Whitney U, Kruskal-Wallis, Wilcoxon signed-rank
- **Association Tests**: Chi-square, Fisher's exact test
- **Correlation Tests**: Significance testing for correlations
- **Homoscedasticity Tests**: Breusch-Pagan, White test

### Feature Engineering
- **Numeric Transformations**: Log, Square root, Box-Cox, Standardization, Normalization
- **Binning**: Equal width, Equal frequency, K-means binning
- **Categorical Encoding**: One-hot, Label, Target, Frequency, Binary encoding
- **Feature Generation**: Polynomial features, Interaction terms, DateTime features
- **Text Processing**: TF-IDF, Bag of words, Sentiment analysis

### Machine Learning
- **Classification**: Logistic Regression, Random Forest, SVM, Naive Bayes, Decision Tree, KNN, Gradient Boosting
- **Regression**: Linear Regression, Random Forest, SVR, Ridge, Lasso, Decision Tree, KNN
- **Clustering**: K-Means, DBSCAN, Hierarchical clustering, Gaussian Mixture
- **Feature Selection**: Univariate selection, RFE, Feature importance, Correlation filtering
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Hyperparameter Tuning**: Grid search and cross-validation

### Report Generation
- **Multiple Formats**: HTML, PDF, Word, JSON
- **Customizable Content**: Include/exclude visualizations, statistics, recommendations
- **Comprehensive Reports**: Full EDA reports with insights and recommendations
- **Export Capabilities**: Download models, visualizations, and results

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd advanced-eda-platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

### Development Setup
For development with hot reloading:
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

## 📊 Usage Guide

### 1. Data Upload
- Click or drag-and-drop your data file
- Supported formats: CSV, Excel (.xlsx, .xls), JSON, Parquet, TSV
- Maximum file size: 100MB
- View data preview with first/last rows and random samples

### 2. Data Preprocessing
- **Missing Values**: Choose from 8 different strategies
- **Duplicates**: Remove duplicate rows
- **Data Types**: Convert between numeric, categorical, datetime, and string types
- **Preview Changes**: See the impact of preprocessing steps

### 3. Exploratory Data Analysis
- Click "Run Global EDA" for comprehensive analysis
- Explore different tabs: Overview, Numeric Analysis, Categorical Analysis, Correlations, Data Quality
- Get automated recommendations for data improvement

### 4. Visualizations
- Select visualization type from 12+ options
- Choose columns for X, Y, color, and size mapping
- Customize plot parameters (bins, colors, themes)
- Save visualizations as PNG images

### 5. Statistical Testing
- Choose from 20+ statistical tests
- Select appropriate columns and parameters
- Interpret results with automated explanations
- Export test results for documentation

### 6. Feature Engineering
- Apply transformations to individual columns
- Preview changes before applying
- Generate new features automatically
- Track transformation history

### 7. Machine Learning
- **Model Training**: Select algorithm, features, and target
- **Feature Selection**: Use automated feature selection methods
- **Model Evaluation**: View comprehensive performance metrics
- **Predictions**: Make predictions on new data

### 8. Report Generation
- Generate comprehensive reports in multiple formats
- Customize content inclusion
- Download reports for sharing and documentation

## 🎨 Interface Features

### Dark Theme Design
- Modern dark theme optimized for long analysis sessions
- High contrast colors for accessibility
- Responsive design for desktop and mobile devices

### Interactive Elements
- Color-coded buttons for different actions
- Real-time progress indicators
- Contextual alerts and notifications
- Tabbed interface for organized workflow

### Visualization Controls
- Plotly-powered interactive charts
- Zoom, pan, and hover capabilities
- Customizable color schemes
- Export-ready high-resolution plots

## 🔧 Technical Architecture

### Backend (Python Flask)
- **Framework**: Flask with session management
- **Data Processing**: Pandas, NumPy for data manipulation
- **Statistics**: SciPy, Statsmodels for statistical analysis
- **Machine Learning**: Scikit-learn for ML algorithms
- **Visualizations**: Plotly for interactive charts

### Frontend (HTML/CSS/JavaScript)
- **Interface**: Custom CSS with dark theme
- **Interactivity**: Vanilla JavaScript with modern ES6+ features
- **Charts**: Plotly.js for data visualizations
- **Responsive**: Mobile-first responsive design

### File Structure
```
advanced-eda-platform/
├── app.py                 # Main Flask application (7000+ lines)
├── index.html            # Frontend interface (7000+ lines)
├── requirements.txt      # Python dependencies
├── README.md            # This documentation
├── uploads/             # Temporary file storage
├── models/              # Saved ML models
└── session_data/        # Session management
```

## 📈 Performance Considerations

- **Memory Management**: Efficient handling of large datasets
- **Session Storage**: Temporary data storage for user sessions
- **Background Processing**: Non-blocking operations for long-running tasks
- **Error Handling**: Comprehensive error catching and user feedback

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🆘 Support

For support, please:
1. Check the documentation above
2. Review the issues section
3. Create a new issue with detailed description

## 🔮 Future Enhancements

- **Advanced ML**: Deep learning models, AutoML capabilities
- **Real-time Data**: Streaming data analysis
- **Collaboration**: Multi-user support and sharing
- **Cloud Integration**: AWS, GCP, Azure integration
- **API Access**: REST API for programmatic access

---

**Built with ❤️ for the data science community**
"""
Isolation Forest for Anomaly Detection - Mac Compatible Version
========================================
This implementation demonstrates anomaly detection using Isolation Forest algorithm
for a Data Mining class project.

Author: Blake Senn, Nick Jebeles, Isaiah Barlatier
Course: Data Mining
Topic: Anomaly Detection with Isolation Forest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_fscore_support
)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class IsolationForestAnomalyDetector:
    """
    A class to perform anomaly detection using Isolation Forest
    
    The Isolation Forest algorithm works by:
    1. Randomly selecting a feature
    2. Randomly selecting a split value between min and max of that feature
    3. Recursively partitioning the data
    4. Anomalies are isolated faster (shorter path lengths)
    
    Key Parameters:
    ---------------
    n_estimators : int, default = 100
        Number of isolation trees in the forest
    contamination : float, default = 0.1
        Expected proportion of outliers in the dataset
    max_samples : int or float, default = 'auto'
        Number of samples to draw to train each tree
    random_state : int, default = 42
        Random seed for reproducibility
    """
    
    def __init__(self, n_estimators=100, contamination=0.1, max_samples='auto', 
                 random_state=42):
        """Initialize the Isolation Forest detector"""
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        
        # Initialize the model
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """
        Fit the Isolation Forest model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        print(f"Fitting Isolation Forest with {self.n_estimators} trees...")
        print(f"Expected contamination: {self.contamination*100:.1f}%")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        print("Model training completed successfully")
        return self
    
    def predict(self, X):
        """
        Predict anomalies in the data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        predictions : array of shape (n_samples,)
            -1 for anomalies, 1 for normal points
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def anomaly_scores(self, X):
        """
        Calculate anomaly scores
        
        Lower scores indicate more anomalous behavior
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to score
            
        Returns:
        --------
        scores : array of shape (n_samples,)
            Anomaly scores (more negative = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)
    
    def decision_function(self, X):
        """
        Compute decision function (anomaly score)
        Same as anomaly_scores but follows sklearn convention
        """
        return self.anomaly_scores(X)


def load_and_preprocess_data(filepath=None, use_sample=True):
    """
    Load and preprocess the Credit Card Fraud dataset
    
    Parameters:
    -----------
    filepath : str, optional
        Path to the CSV file
    use_sample : bool, default=True
        If True, use a smaller sample for faster processing
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Train/test splits
    """
    print("="*60)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*60)
    
    if filepath is None:
        print("\n Generating synthetic fraud detection dataset...")
        from sklearn.datasets import make_classification
        
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=10000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=2,
            weights=[0.97, 0.03],
            flip_y=0.01,
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f'Feature_{i}' for i in range(20)]
        df = pd.DataFrame(X, columns=feature_names)
        df['Class'] = y
        
        print(f"Dataset created: {df.shape[0]} samples, {df.shape[1]-1} features")
        
    else:
        print(f"\n Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Display dataset
    print(f"\n Dataset Statistics:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Normal transactions: {(df['Class']==0).sum():,} ({(df['Class']==0).sum()/len(df)*100:.2f}%)")
    print(f"   Fraudulent transactions: {(df['Class']==1).sum():,} ({(df['Class']==1).sum()/len(df)*100:.2f}%)")
    
    # Separate features and target
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, contamination=0.03):
    """
    Train the Isolation Forest model
    
    Parameters:
    -----------
    X_train : array-like
        Training data
    contamination : float
        Expected proportion of outliers
        
    Returns:
    --------
    detector : IsolationForestAnomalyDetector
        Trained model
    """
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    detector = IsolationForestAnomalyDetector(
        n_estimators=100,
        contamination=contamination,
        max_samples='auto',
        random_state=42
    )
    
    detector.fit(X_train)
    
    return detector


def evaluate_model(detector, X_test, y_test):
    """
    Evaluate the model performance
    
    Parameters:
    -----------
    detector : IsolationForestAnomalyDetector
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        True labels
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    print("\n" + "="*60)
    print("STEP 3: MODEL EVALUATION")
    print("="*60)
    
    # Get predictions
    y_pred = detector.predict(X_test)
    
    # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
    y_pred_binary = np.where(y_pred == -1, 1, 0)
    
    # Get anomaly scores for ROC curve
    anomaly_scores = detector.anomaly_scores(X_test)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_binary, average='binary'
    )
    roc_auc = roc_auc_score(y_test, -anomaly_scores)  # Negative because lower score = anomaly
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    
    print("\n Performance Metrics:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    print("\n Confusion Matrix:")
    print(f"   True Negatives:  {cm[0,0]:,}")
    print(f"   False Positives: {cm[0,1]:,}")
    print(f"   False Negatives: {cm[1,0]:,}")
    print(f"   True Positives:  {cm[1,1]:,}")
    
    print("\n" + classification_report(y_test, y_pred_binary, 
                                        target_names=['Normal', 'Fraud']))
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred_binary,
        'anomaly_scores': anomaly_scores
    }
    
    return metrics


def visualize_results(X_test, y_test, metrics, detector):
    """
    Create visualizations of the results
    
    Parameters:
    -----------
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    metrics : dict
        Evaluation metrics
    detector : IsolationForestAnomalyDetector
        Trained model
    """
    print("\n" + "="*60)
    print("STEP 4: VISUALIZATION")
    print("="*60)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # ROC Curve
    ax2 = plt.subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, -metrics['anomaly_scores'])
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Anomaly Score Distribution
    ax3 = plt.subplot(2, 3, 3)
    normal_scores = metrics['anomaly_scores'][y_test == 0]
    fraud_scores = metrics['anomaly_scores'][y_test == 1]
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(fraud_scores, bins=50, alpha=0.7, label='Fraud', color='red')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Metrics Bar Chart
    ax4 = plt.subplot(2, 3, 4)
    metrics_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_values = [metrics['precision'], metrics['recall'], 
                      metrics['f1'], metrics['roc_auc']]
    bars = plt.bar(metrics_names, metrics_values, color=['#2E5090', '#4A90E2', '#5CB85C', '#E74C3C'])
    plt.ylim([0, 1])
    plt.title('Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # PCA Visualization (2D projection)
    ax5 = plt.subplot(2, 3, 5)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_test)
    
    # Plot normal vs anomalies
    normal_mask = y_test == 0
    fraud_mask = y_test == 1
    
    plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
               c='blue', alpha=0.3, s=10, label='Normal')
    plt.scatter(X_pca[fraud_mask, 0], X_pca[fraud_mask, 1], 
               c='red', alpha=0.8, s=30, label='Fraud', marker='x')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA Projection of Test Data', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Predicted vs Actual
    ax6 = plt.subplot(2, 3, 6)
    pred_normal = (metrics['y_pred'] == 0).sum()
    pred_fraud = (metrics['y_pred'] == 1).sum()
    actual_normal = (y_test == 0).sum()
    actual_fraud = (y_test == 1).sum()
    
    x = np.arange(2)
    width = 0.35
    plt.bar(x - width/2, [actual_normal, actual_fraud], width, 
            label='Actual', color='#4A90E2')
    plt.bar(x + width/2, [pred_normal, pred_fraud], width, 
            label='Predicted', color='#E74C3C')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Predicted vs Actual Counts', fontsize=14, fontweight='bold')
    plt.xticks(x, ['Normal', 'Fraud'])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualizations saved to 'results_visualization.png'")
    plt.show()


def explain_algorithm():
    """
    Print detailed explanation of the Isolation Forest algorithm
    (for presentation and understanding)
    """
    explanation = """
    ╔════════════════════════════════════════════════════════════╗
    ║        ISOLATION FOREST ALGORITHM EXPLANATION             ║
    ╚════════════════════════════════════════════════════════════╝
    
    CORE CONCEPT:
    ───────────────
    Anomalies are "few and different" - they are easier to isolate
    than normal points in the feature space.
    
    ALGORITHM STEPS:
    ──────────────────
    
    1. BUILD ISOLATION TREES (iTrees):
       For each tree in the forest:
       a) Randomly sample data points
       b) Randomly select a feature
       c) Randomly select a split value between min and max
       d) Recursively partition until points are isolated
    
    2. CALCULATE PATH LENGTH:
       For each data point:
       - Count steps needed to isolate it in each tree
       - Average across all trees
       
    3. COMPUTE ANOMALY SCORE:
       s(x, n) = 2^(-E(h(x)) / c(n))
       
       Where:
       - E(h(x)) = average path length for point x
       - c(n) = average path length of unsuccessful search in BST
       - n = number of samples
       
    4. CLASSIFY:
       - Score close to 1: Anomaly
       - Score close to 0: Normal
       - Score around 0.5: Ambiguous
    
    KEY PARAMETERS:
    ─────────────────
    • n_estimators: Number of trees (default: 100)
      └─ More trees = better accuracy, more computation
    
    • contamination: Expected fraction of outliers (default: 0.1)
      └─ Helps set the threshold for classification
    
    • max_samples: Samples per tree (default: 'auto' = min(256, n))
      └─ Smaller = faster training, less memory
    
    COMPLEXITY:
    ─────────────
    • Training: O(t × ψ × log ψ)
      where t = n_estimators, ψ = max_samples
    
    • Prediction: O(t × log ψ)
    
    ADVANTAGES:
    ──────────────
    ✓ Unsupervised (no labels needed)
    ✓ Efficient with large datasets
    ✓ Low memory footprint
    ✓ Handles high-dimensional data well
    ✓ Few hyperparameters to tune
    
    LIMITATIONS:
    ──────────────
    ✗ May struggle with local anomalies
    ✗ Performance depends on contamination parameter
    ✗ Less interpretable than rule-based methods
    """
    
    print(explanation)


def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("  ISOLATION FOREST ANOMALY DETECTION PROJECT")
    print("="*60)
    
    # Explain the algorithm
    explain_algorithm()
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train model
    contamination = y_train.sum() / len(y_train)  # Actual fraud rate
    detector = train_model(X_train, contamination=contamination)
    
    # Evaluate
    metrics = evaluate_model(detector, X_test, y_test)
    
    # Visualize
    visualize_results(X_test, y_test, metrics, detector)
    
    # Summary
    print("\n" + "="*60)
    print("  PROJECT SUMMARY")
    print("="*60)
    print(f"""
    Algorithm: Isolation Forest
    Dataset: Credit Card Fraud Detection (Synthetic)
    Training Samples: {len(X_train):,}
    Test Samples: {len(X_test):,}
    Model Performance:
       - Precision: {metrics['precision']:.4f}
       - Recall: {metrics['recall']:.4f}
       - F1-Score: {metrics['f1']:.4f}
       - ROC-AUC: {metrics['roc_auc']:.4f}
    """)
    
    return detector, metrics


if __name__ == "__main__":
    detector, metrics = main()

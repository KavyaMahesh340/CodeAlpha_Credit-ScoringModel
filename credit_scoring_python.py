

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class CreditScoringModel:
    """Credit scoring model using multiple ML algorithms"""
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """
        Generate synthetic credit data for demonstration
        In production, replace this with real dataset loading
        """
        print("Generating synthetic dataset...")
        
        # Generate features
        income = np.random.normal(50000, 20000, n_samples).clip(20000, 200000)
        age = np.random.normal(40, 12, n_samples).clip(18, 80)
        employment_years = np.random.normal(8, 5, n_samples).clip(0, 40)
        debt = np.random.normal(15000, 10000, n_samples).clip(0, 100000)
        credit_utilization = np.random.normal(30, 20, n_samples).clip(0, 100)
        payment_history = np.random.normal(75, 15, n_samples).clip(0, 100)
        num_credit_accounts = np.random.poisson(5, n_samples).clip(0, 20)
        credit_age_months = np.random.normal(60, 36, n_samples).clip(6, 360)
        
        # Create derived features
        debt_to_income = (debt / income * 100).clip(0, 200)
        
        # Create DataFrame
        df = pd.DataFrame({
            'income': income,
            'age': age,
            'employment_years': employment_years,
            'debt': debt,
            'credit_utilization': credit_utilization,
            'payment_history': payment_history,
            'num_credit_accounts': num_credit_accounts,
            'credit_age_months': credit_age_months,
            'debt_to_income': debt_to_income
        })
        
        # Generate target variable (creditworthy: 1, not creditworthy: 0)
        # Based on weighted combination of features
        credit_score = (
            (df['payment_history'] * 0.35) +
            ((100 - df['credit_utilization']) * 0.30) +
            (df['income'] / 2000 * 0.20) +
            ((100 - df['debt_to_income']) * 0.10) +
            (df['credit_age_months'] / 12 * 0.05)
        )
        
        # Add some randomness
        credit_score += np.random.normal(0, 10, n_samples)
        
        # Convert to binary classification (threshold at median)
        df['creditworthy'] = (credit_score > credit_score.median()).astype(int)
        
        print(f"Dataset created: {n_samples} samples, {len(df.columns)-1} features")
        print(f"Class distribution: {df['creditworthy'].value_counts().to_dict()}")
        
        return df
    
    def load_data(self, filepath=None):
        """
        Load credit data from CSV file
        If no file provided, generate synthetic data
        """
        if filepath:
            print(f"Loading data from {filepath}...")
            df = pd.read_csv(filepath)
        else:
            df = self.generate_synthetic_data()
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess and prepare data for modeling"""
        print("\nPreprocessing data...")
        
        # Separate features and target
        X = df.drop('creditworthy', axis=1)
        y = df['creditworthy']
        
        self.feature_names = X.columns.tolist()
        
        # Handle missing values (if any)
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return trained_models
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models and compare performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        results = {}
        best_score = 0
        
        for name, model in models.items():
            print(f"\n{name}")
            print("-" * 40)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Print metrics
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                                       target_names=['Not Creditworthy', 'Creditworthy']))
            
            # Track best model
            if f1 > best_score:
                best_score = f1
                self.best_model = name
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {self.best_model} (F1-Score: {best_score:.4f})")
        print(f"{'='*50}")
        
        return results
    
    def plot_results(self, results, y_test):
        """Visualize model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison - Bar Chart
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(results.keys())
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model_name in enumerate(model_names):
            values = [results[model_name][metric] for metric in metrics]
            axes[0, 0].bar(x + i*width, values, width, label=model_name)
        
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(metrics, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. ROC Curves
        for model_name, data in results.items():
            fpr, tpr, _ = roc_curve(y_test, data['y_pred_proba'])
            axes[0, 1].plot(fpr, tpr, label=f"{model_name} (AUC={data['roc_auc']:.3f})")
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Confusion Matrix for best model
        best_model_data = results[self.best_model]
        cm = confusion_matrix(y_test, best_model_data['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title(f'Confusion Matrix - {self.best_model}')
        
        # 4. Feature Importance (for Random Forest)
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            axes[1, 1].bar(range(len(importances)), importances[indices])
            axes[1, 1].set_xlabel('Features')
            axes[1, 1].set_ylabel('Importance')
            axes[1, 1].set_title('Feature Importance - Random Forest')
            axes[1, 1].set_xticks(range(len(importances)))
            axes[1, 1].set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('credit_scoring_results.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'credit_scoring_results.png'")
        plt.show()
    
    def predict_creditworthiness(self, customer_data):
        """
        Predict creditworthiness for new customer
        customer_data: dict with feature values
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Get best model
        model = self.models[self.best_model]
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0, 1]
        
        result = {
            'creditworthy': bool(prediction),
            'probability': probability,
            'credit_score': int(probability * 100),
            'recommendation': 'APPROVED' if prediction else 'REVIEW NEEDED'
        }
        
        return result


def main():
    """Main execution function"""
    print("="*60)
    print("CREDIT SCORING MODEL - CODEALPHA ML INTERNSHIP")
    print("="*60)
    
    # Initialize model
    model = CreditScoringModel()
    
    # Load data (or generate synthetic)
    # To use real data: df = model.load_data('your_dataset.csv')
    df = model.load_data()
    
    # Preprocess
    X_train, X_test, y_train, y_test = model.preprocess_data(df)
    
    # Train models
    trained_models = model.train_models(X_train, y_train)
    
    # Evaluate models
    results = model.evaluate_models(trained_models, X_test, y_test)
    
    # Visualize results
    model.plot_results(results, y_test)
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    
    sample_customer = {
        'income': 60000,
        'age': 35,
        'employment_years': 5,
        'debt': 12000,
        'credit_utilization': 25,
        'payment_history': 85,
        'num_credit_accounts': 4,
        'credit_age_months': 48,
        'debt_to_income': 20
    }
    
    print("\nCustomer Profile:")
    for key, value in sample_customer.items():
        print(f"  {key}: {value}")
    
    prediction = model.predict_creditworthiness(sample_customer)
    
    print("\nPrediction Results:")
    print(f"  Creditworthy: {prediction['creditworthy']}")
    print(f"  Probability: {prediction['probability']:.2%}")
    print(f"  Credit Score: {prediction['credit_score']}/100")
    print(f"  Recommendation: {prediction['recommendation']}")
    
    print("\n" + "="*60)
    print("TASK COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()

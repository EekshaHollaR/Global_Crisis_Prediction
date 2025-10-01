
"""
COMPREHENSIVE CRISIS PREDICTION MODEL
=====================================

This script implements a complete machine learning pipeline for predicting 
economic and food crises using historical country-level data.

Author: Crisis Prediction System
Date: 2024
Version: 1.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class CrisisPredictionModel:
    """
    Comprehensive Crisis Prediction Model using Machine Learning
    """

    def __init__(self):
        """Initialize the crisis prediction model"""
        self.models = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None

    def load_and_merge_data(self, food_file, economic_file):
        """
        Load and merge food and economic crisis datasets

        Parameters:
        food_file (str): Path to food crisis data Excel file
        economic_file (str): Path to economic crisis data Excel file

        Returns:
        pd.DataFrame: Merged dataset
        """
        # Load datasets
        food_data = pd.read_excel(food_file)
        economic_data = pd.read_excel(economic_file)

        # Merge datasets
        merged_data = pd.merge(
            food_data, 
            economic_data, 
            on=['Country Name', 'Country Code', 'Year'], 
            suffixes=('_food', '_economic')
        )

        # Clean duplicate columns
        merged_data = merged_data.drop(columns=[
            'GDP (current US$)',
            'GDP growth (annual %)_food',
            'GDP per capita (current US$)_food',
            'Inflation, consumer prices (annual %)_food'
        ])

        # Rename columns
        merged_data = merged_data.rename(columns={
            'GDP growth (annual %)_economic': 'GDP growth (annual %)',
            'GDP per capita (current US$)_economic': 'GDP per capita (current US$)',
            'Inflation, consumer prices (annual %)_economic': 'Inflation, consumer prices (annual %)'
        })

        return merged_data

    def create_crisis_indicators(self, df):
        """
        Create crisis indicators based on research thresholds

        Parameters:
        df (pd.DataFrame): Input dataset

        Returns:
        pd.DataFrame: Dataset with crisis indicators
        """
        df_crisis = df.copy()

        # Economic crisis indicators
        df_crisis['gdp_growth_crisis'] = (df_crisis['GDP growth (annual %)'] < -2.0).astype(int)
        df_crisis['high_inflation_crisis'] = (df_crisis['Inflation, consumer prices (annual %)'] > 10.0).astype(int)
        df_crisis['unemployment_crisis'] = (df_crisis['Unemployment, total (% of total labor force) (modeled ILO estimate)'] > 15.0).astype(int)
        df_crisis['credit_crisis'] = (df_crisis['Domestic credit to private sector (% of GDP)'] < 10.0).astype(int)
        df_crisis['investment_crisis'] = (df_crisis['Gross fixed capital formation (% of GDP)'] < 15.0).astype(int)
        df_crisis['trade_imbalance_crisis'] = (
            (df_crisis['Imports of goods and services (% of GDP)'] - 
             df_crisis['Exports of goods and services (% of GDP)']) > 10.0
        ).astype(int)

        # Food crisis indicators
        cereal_yield_threshold = df_crisis['Cereal yield (kg per hectare)'].quantile(0.25)
        df_crisis['low_cereal_yield_crisis'] = (df_crisis['Cereal yield (kg per hectare)'] < cereal_yield_threshold).astype(int)
        df_crisis['high_food_import_crisis'] = (df_crisis['Food imports (% of merchandise imports)'] > 15.0).astype(int)
        df_crisis['low_food_production_crisis'] = (df_crisis['Food production index (2014-2016 = 100)'] < 85.0).astype(int)
        df_crisis['population_pressure_crisis'] = (
            (df_crisis['Population growth (annual %)'] > 2.5) & 
            (df_crisis['GDP growth (annual %)'] < 2.0)
        ).astype(int)

        # Composite crisis scores
        df_crisis['economic_crisis_score'] = (
            df_crisis['gdp_growth_crisis'] +
            df_crisis['high_inflation_crisis'] +
            df_crisis['unemployment_crisis'] +
            df_crisis['credit_crisis'] +
            df_crisis['investment_crisis'] +
            df_crisis['trade_imbalance_crisis']
        )

        df_crisis['food_crisis_score'] = (
            df_crisis['low_cereal_yield_crisis'] +
            df_crisis['high_food_import_crisis'] +
            df_crisis['low_food_production_crisis'] +
            df_crisis['population_pressure_crisis']
        )

        # Overall crisis indicator
        df_crisis['overall_crisis'] = (
            (df_crisis['economic_crisis_score'] >= 3) | 
            (df_crisis['food_crisis_score'] >= 2)
        ).astype(int)

        # Crisis type classification
        def classify_crisis_type(row):
            if row['economic_crisis_score'] >= 3 and row['food_crisis_score'] >= 2:
                return 'Combined Crisis'
            elif row['economic_crisis_score'] >= 3:
                return 'Economic Crisis'
            elif row['food_crisis_score'] >= 2:
                return 'Food Crisis'
            else:
                return 'No Crisis'

        df_crisis['crisis_type'] = df_crisis.apply(classify_crisis_type, axis=1)

        return df_crisis

    def create_advanced_features(self, df):
        """
        Create advanced features for machine learning

        Parameters:
        df (pd.DataFrame): Input dataset

        Returns:
        pd.DataFrame: Dataset with advanced features
        """
        df_features = df.copy()
        df_features = df_features.sort_values(['Country Name', 'Year'])

        # Lag features
        for col in ['GDP growth (annual %)', 'Inflation, consumer prices (annual %)', 
                    'Unemployment, total (% of total labor force) (modeled ILO estimate)',
                    'Cereal yield (kg per hectare)', 'Food production index (2014-2016 = 100)']:
            df_features[f'{col}_lag1'] = df_features.groupby('Country Name')[col].shift(1)
            df_features[f'{col}_change'] = df_features[col] - df_features[f'{col}_lag1']

        # Volatility indicators
        for col in ['GDP growth (annual %)', 'Inflation, consumer prices (annual %)']:
            df_features[f'{col}_volatility'] = df_features.groupby('Country Name')[col].rolling(window=3).std().reset_index(0, drop=True)

        # Additional economic indicators
        df_features['credit_gdp_lag1'] = df_features.groupby('Country Name')['Domestic credit to private sector (% of GDP)'].shift(1)
        df_features['credit_growth'] = df_features['Domestic credit to private sector (% of GDP)'] - df_features['credit_gdp_lag1']
        df_features['trade_balance'] = df_features['Exports of goods and services (% of GDP)'] - df_features['Imports of goods and services (% of GDP)']
        df_features['export_import_ratio'] = df_features['Exports of goods and services (% of GDP)'] / df_features['Imports of goods and services (% of GDP)']
        df_features['food_import_dependency'] = df_features['Food imports (% of merchandise imports)'] / 100
        df_features['food_production_per_capita'] = df_features['Food production index (2014-2016 = 100)'] / df_features['Population growth (annual %)']
        df_features['gdp_per_capita_lag1'] = df_features.groupby('Country Name')['GDP per capita (current US$)'].shift(1)
        df_features['gdp_per_capita_growth'] = ((df_features['GDP per capita (current US$)'] - df_features['gdp_per_capita_lag1']) / df_features['gdp_per_capita_lag1']) * 100
        df_features['investment_efficiency'] = df_features['GDP growth (annual %)'] / df_features['Gross fixed capital formation (% of GDP)']

        return df_features

    def prepare_features(self, df):
        """
        Prepare features for machine learning

        Parameters:
        df (pd.DataFrame): Input dataset

        Returns:
        tuple: X (features), y (target), cleaned dataframe
        """
        # Define feature columns
        self.feature_columns = [
            'GDP growth (annual %)', 'Inflation, consumer prices (annual %)',
            'Unemployment, total (% of total labor force) (modeled ILO estimate)',
            'Domestic credit to private sector (% of GDP)',
            'Exports of goods and services (% of GDP)',
            'Imports of goods and services (% of GDP)',
            'Gross fixed capital formation (% of GDP)',
            'GDP per capita (current US$)',
            'Cereal yield (kg per hectare)',
            'Food imports (% of merchandise imports)',
            'Food production index (2014-2016 = 100)',
            'Population growth (annual %)',
            'GDP growth (annual %)_change',
            'Inflation, consumer prices (annual %)_change',
            'Unemployment, total (% of total labor force) (modeled ILO estimate)_change',
            'GDP growth (annual %)_volatility',
            'Inflation, consumer prices (annual %)_volatility',
            'credit_growth',
            'trade_balance',
            'export_import_ratio',
            'food_import_dependency',
            'food_production_per_capita',
            'gdp_per_capita_growth',
            'investment_efficiency'
        ]

        # Clean data
        df_clean = df.dropna(subset=['GDP growth (annual %)_volatility']).copy()

        # Separate features and target
        X = df_clean[self.feature_columns].copy()
        y = df_clean['overall_crisis'].copy()

        # Handle missing values
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns, index=X.index)

        return X_imputed, y, df_clean

    def train_models(self, X, y):
        """
        Train multiple machine learning models

        Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable

        Returns:
        dict: Model results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True)
        }

        model_results = {}

        for name, model in models.items():
            print(f"Training {name}...")

            # Train model
            if name == 'SVM':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            model_results[name] = metrics
            self.models[name] = model

        # Create ensemble model
        ensemble_models = [
            ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
            ('gb', GradientBoostingClassifier(random_state=42, n_estimators=100)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ]

        ensemble_model = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble_model.fit(X_train, y_train)
        ensemble_pred = ensemble_model.predict(X_test)
        ensemble_proba = ensemble_model.predict_proba(X_test)[:, 1]

        model_results['Ensemble'] = {
            'model': ensemble_model,
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred, zero_division=0),
            'recall': recall_score(y_test, ensemble_pred),
            'f1_score': f1_score(y_test, ensemble_pred),
            'auc_roc': roc_auc_score(y_test, ensemble_proba),
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        }

        self.models['Ensemble'] = ensemble_model

        # Select best model
        performance_comparison = pd.DataFrame(model_results).T[['f1_score']]
        self.best_model_name = performance_comparison['f1_score'].idxmax()
        self.best_model = model_results[self.best_model_name]['model']

        print(f"\nBest performing model: {self.best_model_name}")

        return model_results, X_test, y_test

    def make_future_predictions(self, data, years_ahead=5):
        """
        Make future crisis predictions

        Parameters:
        data (pd.DataFrame): Historical data
        years_ahead (int): Number of years to predict ahead

        Returns:
        pd.DataFrame: Future predictions
        """
        latest_data = data.groupby('Country Name').tail(1).copy()
        predictions_list = []

        for year_ahead in range(1, years_ahead + 1):
            future_year = latest_data['Year'].max() + year_ahead

            for _, row in latest_data.iterrows():
                country = row['Country Name']
                features = row[self.feature_columns].values.reshape(1, -1)

                # Make prediction
                if self.best_model_name == 'SVM':
                    features_scaled = self.scaler.transform(features)
                    crisis_prob = self.best_model.predict_proba(features_scaled)[0, 1]
                    crisis_pred = self.best_model.predict(features_scaled)[0]
                else:
                    crisis_prob = self.best_model.predict_proba(features)[0, 1]
                    crisis_pred = self.best_model.predict(features)[0]

                predictions_list.append({
                    'Country': country,
                    'Prediction_Year': future_year,
                    'Years_Ahead': year_ahead,
                    'Crisis_Probability': crisis_prob,
                    'Crisis_Prediction': 'Crisis' if crisis_pred else 'No Crisis',
                })

        return pd.DataFrame(predictions_list)

    def save_model(self, filename):
        """
        Save the trained model and preprocessors

        Parameters:
        filename (str): Output filename
        """
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'models': self.models
        }

        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """
        Load a previously trained model

        Parameters:
        filename (str): Model filename
        """
        model_data = joblib.load(filename)

        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.feature_columns = model_data['feature_columns']
        self.models = model_data['models']

        print(f"Model loaded from {filename}")

def main():
    """
    Main execution function
    """
    print("Crisis Prediction Model - Starting Analysis...")
    print("=" * 60)

    # Initialize model
    crisis_model = CrisisPredictionModel()

    # Load and prepare data
    print("Loading and merging datasets...")
    merged_data = crisis_model.load_and_merge_data(r'Prediction_Model_(Gradient_Boosting)\food-crisis-data.xlsx', r'Prediction_Model_(Gradient_Boosting)\economic-crisis-data.xlsx')

    print("Creating crisis indicators...")
    crisis_data = crisis_model.create_crisis_indicators(merged_data)

    print("Creating advanced features...")
    enhanced_data = crisis_model.create_advanced_features(crisis_data)

    print("Preparing features for machine learning...")
    X, y, clean_data = crisis_model.prepare_features(enhanced_data)

    print(f"Dataset shape: {X.shape}")

    print(f"Crisis rate: {y.mean():.2%}")

    # Train models
    print("\nTraining machine learning models...")
    model_results, X_test, y_test = crisis_model.train_models(X, y)

    # Display results
    print("\nModel Performance Comparison:")
    performance_df = pd.DataFrame(model_results).T[['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']]
    print(performance_df.round(3))

    # Make future predictions
    print("\nMaking future predictions...")
    future_predictions = crisis_model.make_future_predictions(clean_data)

    # Save results
    future_predictions.to_csv('crisis_predictions.csv', index=False)
    crisis_model.save_model('crisis_prediction_model.pkl')

    print("\nAnalysis complete!")
    print("Files created:")
    print("- crisis_predictions.csv: Future crisis predictions")
    print("- crisis_prediction_model.pkl: Trained model")

    # High-risk countries summary
    high_risk = future_predictions.groupby('Country')['Crisis_Probability'].mean().sort_values(ascending=False)
    print("\nTop 10 High-Risk Countries:")
    for i, (country, prob) in enumerate(high_risk.head(10).items(), 1):
        print(f"{i:2d}. {country:<20} {prob:.3f}")

if __name__ == "__main__":
    main()

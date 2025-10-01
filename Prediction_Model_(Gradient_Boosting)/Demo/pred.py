"""
Advanced, Research-Driven Crisis Prediction Model
-------------------------------------------------
- Reads food and economic datasets (Excel) with given schema
- Builds research-based crisis indicators and time-series features
- Trains multiple ML models, selects the best (by F1 then AUC)
- Generates predictions for 1–5 years ahead per country
- Attributes causes (parameters) for predicted crises
- Outputs CSV and Excel files, plus dynamic Markdown report
- Generalizes to any new dataset with the same headers
"""

import pandas as pd
import numpy as np
import datetime
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')


class AdvancedCrisisPredictionModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        self.imputer = None
        self.scaler = None
        self.feature_importance = {}
        self.model_performance = {}
        self.crisis_thresholds = {}

    # 1) Data loading and merging
    def load_and_merge_data(self, food_file, economic_file):
        food_data = pd.read_excel(food_file)
        economic_data = pd.read_excel(economic_file)

        merged_data = pd.merge(
            food_data,
            economic_data,
            on=['Country Name', 'Country Code', 'Year'],
            suffixes=('_food', '_economic')
        )

        # Remove duplicate economic columns from food file
        merged_data = merged_data.drop(columns=[
            'GDP (current US$)',
            'GDP growth (annual %)_food',
            'GDP per capita (current US$)_food',
            'Inflation, consumer prices (annual %)_food'
        ])
        # Rename to unified names
        merged_data = merged_data.rename(columns={
            'GDP growth (annual %)_economic': 'GDP growth (annual %)',
            'GDP per capita (current US$)_economic': 'GDP per capita (current US$)',
            'Inflation, consumer prices (annual %)_economic': 'Inflation, consumer prices (annual %)'
        })
        return merged_data

    # 2) Research-based crisis indicators and composite scoring
    def create_research_based_crisis_indicators(self, df):
        df = df.copy()

        # Economic crisis indicators (research-aligned thresholds)
        df['severe_recession'] = (df['GDP growth (annual %)'] < -3.0).astype(int)
        df['hyperinflation'] = (df['Inflation, consumer prices (annual %)'] > 20.0).astype(int)
        df['high_unemployment'] = (df['Unemployment, total (% of total labor force) (modeled ILO estimate)'] > 10.0).astype(int)
        df['credit_crunch'] = (df['Domestic credit to private sector (% of GDP)'] < 15.0).astype(int)
        df['investment_collapse'] = (df['Gross fixed capital formation (% of GDP)'] < 10.0).astype(int)
        df['trade_deficit_crisis'] = ((df['Imports of goods and services (% of GDP)'] - df['Exports of goods and services (% of GDP)']) > 15.0).astype(int)

        # Food crisis indicators
        cereal_p25 = df['Cereal yield (kg per hectare)'].quantile(0.25)
        df['low_cereal_yield'] = (df['Cereal yield (kg per hectare)'] < cereal_p25).astype(int)
        df['high_food_dependency'] = (df['Food imports (% of merchandise imports)'] > 15.0).astype(int)
        df['low_food_production'] = (df['Food production index (2014-2016 = 100)'] < 90.0).astype(int)
        df['demographic_stress'] = ((df['Population growth (annual %)'] > 3.0) &
                                    (df['GDP growth (annual %)'] < 1.0)).astype(int)

        # Composite scores
        df['economic_crisis_score'] = df[[
            'severe_recession', 'hyperinflation', 'high_unemployment',
            'credit_crunch', 'investment_collapse', 'trade_deficit_crisis'
        ]].sum(axis=1)

        df['food_crisis_score'] = df[[
            'low_cereal_yield', 'high_food_dependency',
            'low_food_production', 'demographic_stress'
        ]].sum(axis=1)

        # Overall crisis definition
        df['overall_crisis'] = ((df['economic_crisis_score'] >= 2) |
                                (df['food_crisis_score'] >= 2)).astype(int)

        # Type classification
        def classify(row):
            if row['economic_crisis_score'] >= 2 and row['food_crisis_score'] >= 2:
                return 'Combined Crisis'
            elif row['economic_crisis_score'] >= 2:
                return 'Economic Crisis'
            elif row['food_crisis_score'] >= 2:
                return 'Food Crisis'
            return 'No Crisis'
        df['crisis_type'] = df.apply(classify, axis=1)

        # Save thresholds used to allow generalization
        self.crisis_thresholds = {
            'cereal_p25': float(cereal_p25)
        }

        return df

    # 3) Time series features (lags, momentum, rolling stats, volatility)
    def create_time_series_features(self, df):
        df = df.sort_values(['Country Name', 'Year']).copy()
        base_vars = [
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
            'Population growth (annual %)'
        ]
        for v in base_vars:
            df[f'{v}_lag1'] = df.groupby('Country Name')[v].shift(1)
            df[f'{v}_change'] = df[v] - df[f'{v}_lag1']
            df[f'{v}_roll_std3'] = df.groupby('Country Name')[v].rolling(3, min_periods=2).std().reset_index(0, drop=True)
            df[f'{v}_roll_mean3'] = df.groupby('Country Name')[v].rolling(3, min_periods=2).mean().reset_index(0, drop=True)

        # External balance and efficiency
        df['trade_balance'] = df['Exports of goods and services (% of GDP)'] - df['Imports of goods and services (% of GDP)']
        df['investment_efficiency'] = np.where(
            df['Gross fixed capital formation (% of GDP)'] > 0,
            df['GDP growth (annual %)'] / df['Gross fixed capital formation (% of GDP)'],
            0
        )
        return df

    # 4) Training data preparation
    def prepare_training_data(self, df):
        features = [
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
            'Domestic credit to private sector (% of GDP)_change',
            'Exports of goods and services (% of GDP)_change',
            'Imports of goods and services (% of GDP)_change',
            'Gross fixed capital formation (% of GDP)_change',
            'Cereal yield (kg per hectare)_change',
            'Food imports (% of merchandise imports)_change',
            'Food production index (2014-2016 = 100)_change',

            'trade_balance',
            'investment_efficiency'
        ]

        # Drop early-year rows where rolling features are insufficient
        df_clean = df.dropna(subset=['GDP growth (annual %)_roll_std3']).copy()

        X = df_clean[features]
        y = df_clean['overall_crisis']

        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns, index=X.index)

        self.feature_columns = features
        self.imputer = imputer
        self.scaler = scaler
        return X_scaled, y, df_clean

    # 5) Model training and selection
    def train_and_select_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        candidates = {
            'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_split=5, random_state=42,
                class_weight='balanced_subsample'
            ),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }

        performance = {}
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            performance[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
                'model': model
            }

        best = max(performance.items(), key=lambda kv: (kv[1]['f1'], kv[1]['auc']))
        self.best_model_name = best[0]
        self.best_model = best[1]['model']
        self.model_performance = performance

        # Feature importance for attribution
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.best_model.feature_importances_))
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importance = dict(zip(X.columns, np.abs(self.best_model.coef_[0])))
        else:
            self.feature_importance = {f: 0 for f in X.columns}

        return performance

    # 6) Future predictions with cause attribution
    def predict_future(self, df_clean, years_ahead=5):
        latest = df_clean.groupby('Country Name').tail(1).copy()
        results = []

        for _, row in latest.iterrows():
            X_row = pd.DataFrame([row[self.feature_columns]])
            X_row = pd.DataFrame(self.imputer.transform(X_row), columns=self.feature_columns)
            X_row = pd.DataFrame(self.scaler.transform(X_row), columns=self.feature_columns)

            if hasattr(self.best_model, 'predict_proba'):
                prob = float(self.best_model.predict_proba(X_row)[0, 1])
            else:
                prob = float(self.best_model.decision_function(X_row)[0])

            pred = (prob >= 0.5)

            # Causes: top important features
            top_causes = sorted(self.feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:8]
            cause_params = [c[0] for c in top_causes]

            # Crisis type attribution based on indicators
            econ_score = int(row['severe_recession']) + int(row['hyperinflation']) + int(row['high_unemployment']) + \
                         int(row['credit_crunch']) + int(row['investment_collapse']) + int(row['trade_deficit_crisis'])
            food_score = int(row['low_cereal_yield']) + int(row['high_food_dependency']) + \
                         int(row['low_food_production']) + int(row['demographic_stress'])

            if econ_score >= 2 and food_score >= 2:
                ctype = 'Combined Crisis'
            elif econ_score >= 2:
                ctype = 'Economic Crisis'
            elif food_score >= 2:
                ctype = 'Food Crisis'
            else:
                ctype = 'No Crisis'

            for a in range(1, years_ahead + 1):
                results.append({
                    'Country': row['Country Name'],
                    'Prediction_Year': int(row['Year'] + a),
                    'Crisis_Probability': prob,
                    'Crisis_Prediction': 'Crisis' if pred else 'No Crisis',
                    'Predicted_Crisis_Type': ctype,
                    'Top_Causes': ', '.join(cause_params)
                })

        return pd.DataFrame(results)

    # 7) Export results (CSV, Excel, dynamic Markdown report)
    def export_outputs(self, future_predictions, df_clean, output_prefix='Advanced_Crisis'):
        country_summary = future_predictions.groupby('Country').agg(
            Avg_Crisis_Prob=('Crisis_Probability', 'mean'),
            Max_Crisis_Prob=('Crisis_Probability', 'max'),
            Crisis_Years=('Crisis_Prediction', lambda x: (x == 'Crisis').sum()),
            Most_Recent_Type=('Predicted_Crisis_Type', 'last'),
            Top_Causes=('Top_Causes', lambda s: s.mode().iloc[0] if not s.mode().empty else '')
        ).reset_index().sort_values('Avg_Crisis_Prob', ascending=False)

        latest_indicators = df_clean.groupby('Country Name').tail(1)[[
            'Country Name', 'Year',
            'GDP growth (annual %)', 'Inflation, consumer prices (annual %)',
            'Unemployment, total (% of total labor force) (modeled ILO estimate)',
            'Cereal yield (kg per hectare)',
            'Food production index (2014-2016 = 100)',
            'economic_crisis_score', 'food_crisis_score', 'overall_crisis', 'crisis_type'
        ]]

        # Excel workbook
        with pd.ExcelWriter(f'{output_prefix}_Results.xlsx', engine='openpyxl') as writer:
            future_predictions.to_excel(writer, sheet_name='Future Predictions', index=False)
            country_summary.to_excel(writer, sheet_name='Country Summary', index=False)
            latest_indicators.to_excel(writer, sheet_name='Latest Indicators', index=False)
            pd.DataFrame(self.model_performance).T.to_excel(writer, sheet_name='Model Performance')
            fi = pd.DataFrame(
                {'feature': list(self.feature_importance.keys()), 'importance': list(self.feature_importance.values())}
            ).sort_values('importance', ascending=False)
            fi.to_excel(writer, sheet_name='Feature Importance', index=False)

        # CSV
        future_predictions.to_csv(f'{output_prefix}_Predictions.csv', index=False)

        # Dynamic Markdown report
        doc = []
        doc.append('# Crisis Prediction Results - Dynamic Report')
        doc.append(f'Generated: {datetime.datetime.utcnow().isoformat()}Z')

        doc.append('\n## Executive Summary')
        doc.append('This report presents time series–based crisis predictions using research-driven indicators and machine learning. '
                   'The model identifies crisis risk, likely crisis type, and the top contributing parameters for each country. '
                   'It generalizes to any dataset with the same column headers.')

        doc.append('\n## Top High-Risk Countries')
        doc.append(country_summary.head(10).to_markdown(index=False))

        doc.append('\n## Model Selection and Performance')
        doc.append(f'Best Model: {self.best_model_name}')
        perf = pd.DataFrame(self.model_performance).T[['accuracy', 'precision', 'recall', 'f1', 'auc']].round(3)
        doc.append(perf.to_markdown())

        doc.append('\n## Feature Importance (Top 20)')
        doc.append(pd.DataFrame({'feature': list(self.feature_importance.keys()),
                                 'importance': list(self.feature_importance.values())}
                                ).sort_values('importance', ascending=False).head(20).to_markdown(index=False))

        doc.append('\n## Methodology Overview')
        doc.append('- Research-based thresholds to construct economic and food crisis indicators')
        doc.append('- Time series lags, changes, rolling stats to capture momentum, cycles, and volatility')
        doc.append('- Model selection across Logistic Regression, Random Forest, and Gradient Boosting; best by F1 and AUC')
        doc.append('- Predictions generalize to any dataset with the same schema (column headers)')

        doc_text = '\n\n'.join(doc)
        with open(f'{output_prefix}_Report.md', 'w') as f:
            f.write(doc_text)

        return {
            'excel': f'{output_prefix}_Results.xlsx',
            'csv': f'{output_prefix}_Predictions.csv',
            'report': f'{output_prefix}_Report.md'
        }


def main(food_file=r'Prediction_Model_(Gradient_Boosting)\food-crisis-data.xlsx', economic_file=r'Prediction_Model_(Gradient_Boosting)\economic-crisis-data.xlsx',
         output_prefix='Advanced_Crisis'):
    model = AdvancedCrisisPredictionModel()
    merged = model.load_and_merge_data(food_file, economic_file)
    with_ind = model.create_research_based_crisis_indicators(merged)
    with_ts = model.create_time_series_features(with_ind)
    X, y, clean = model.prepare_training_data(with_ts)
    model.train_and_select_model(X, y)
    future_preds = model.predict_future(clean, years_ahead=5)
    outputs = model.export_outputs(future_preds, clean, output_prefix=output_prefix)
    print('Best model selected:', model.best_model_name)
    print('Outputs created:', outputs)


if __name__ == '__main__':
    main()

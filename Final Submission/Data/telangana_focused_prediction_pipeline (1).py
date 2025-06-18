# Telangana Maternal Health - Focused Prediction Pipeline
# Predicts only the 6 specified outcomes using existing data from 4 Excel files

import pandas as pd
import numpy as np
import gc
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class TelanganaMaternalHealthPipeline:
    """
    Comprehensive pipeline for 6 key maternal health predictions
    Uses only data available in the 4 Excel files
    """
    
    def __init__(self, sample_size=500000):
        self.sample_size = sample_size
        self.models = {}
        self.feature_importance = {}
        self.critical_case_stats = {}
        
        # Define the 6 prediction targets
        self.prediction_targets = {
            'high_risk_pregnancy': {
                'name': 'High-Risk Pregnancy Classification',
                'type': 'composite',
                'threshold': 0.4,
                'scale_pos_weight': 3
            },
            'stillbirth_risk': {
                'name': 'Stillbirth Risk',
                'type': 'outcome',
                'threshold': 0.15,
                'scale_pos_weight': 30
            },
            'premature_birth_risk': {
                'name': 'Premature Birth Risk',
                'type': 'outcome',
                'threshold': 0.35,
                'scale_pos_weight': 5
            },
            'maternal_mortality_risk': {
                'name': 'Maternal Mortality Risk',
                'type': 'outcome',
                'threshold': 0.1,
                'scale_pos_weight': 50
            },
            'birth_defect_risk': {
                'name': 'Birth Defect Risk',
                'type': 'outcome',
                'threshold': 0.2,
                'scale_pos_weight': 20
            },
            'anc_dropout': {
                'name': 'ANC Dropout Prediction',
                'type': 'behavioral',
                'threshold': 0.3,
                'scale_pos_weight': 3
            }
        }
    
    def merge_data_files(self, pregnancy_file, anc_file, delivery_file, child_file, output_file):
        """
        Merge the 4 Excel/Parquet files into one comprehensive dataset
        """
        print("Merging Telangana maternal health data files...")
        
        # Load each file
        print("Loading pregnancy data...")
        if pregnancy_file.endswith('.xlsx'):
            pregnancy_df = pd.read_excel(pregnancy_file)
        else:
            pregnancy_df = pd.read_parquet(pregnancy_file)
        
        print("Loading ANC data...")
        if anc_file.endswith('.xlsx'):
            anc_df = pd.read_excel(anc_file)
        else:
            anc_df = pd.read_parquet(anc_file)
        
        print("Loading delivery data...")
        if delivery_file.endswith('.xlsx'):
            delivery_df = pd.read_excel(delivery_file)
        else:
            delivery_df = pd.read_parquet(delivery_file)
        
        print("Loading child data...")
        if child_file.endswith('.xlsx'):
            child_df = pd.read_excel(child_file)
        else:
            child_df = pd.read_parquet(child_file)
        
        # Aggregate ANC visits per mother
        anc_agg = anc_df.groupby(['MOTHER_ID', 'GRAVIDA']).agg({
            'ANC_ID': 'count',
            'HEMOGLOBIN': ['mean', 'min', 'max'],
            'BP': 'last',
            'WEIGHT': ['first', 'last', 'max'],
            'NO_OF_WEEKS': 'max',
            'TWIN_PREGNANCY': 'max',
            'PHQ_SCORE': 'max',
            'GAD_SCORE': 'max',
            'HIV': 'max',
            'THYROID': 'max',
            'SYPHYLIS': 'max'
        }).reset_index()
        
        # Flatten column names
        anc_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                          for col in anc_agg.columns.values]
        anc_agg.rename(columns={'ANC_ID_count': 'TOTAL_ANC_VISITS'}, inplace=True)
        
        # Merge all datasets
        merged_df = pregnancy_df.merge(anc_agg, on=['MOTHER_ID', 'GRAVIDA'], how='left')
        merged_df = merged_df.merge(delivery_df, on=['MOTHER_ID', 'GRAVIDA'], how='left')
        
        # Aggregate child data for twins
        child_agg = child_df.groupby(['MOTHER_ID', 'GRAVIDA']).agg({
            'IS_CHILD_DEATH': 'max',
            'IS_DEFECTIVE_BIRTH': 'max',
            'WEIGHT': ['mean', 'min'],
            'IS_BF_IN_HOUR': 'min'
        }).reset_index()
        
        child_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in child_agg.columns.values]
        
        merged_df = merged_df.merge(child_agg, on=['MOTHER_ID', 'GRAVIDA'], how='left')
        
        print(f"Merged dataset: {len(merged_df)} records, {len(merged_df.columns)} columns")
        
        # Save merged data
        merged_df.to_parquet(output_file, compression='snappy', index=False)
        print(f"Saved merged data to {output_file}")
        
        return merged_df
    
    def create_target_variables(self, df):
        """
        Create the 6 specific target variables from existing data
        """
        print("\nCreating target variables...")
        
        # 1. Stillbirth risk (from delivery data)
        if 'DELIVERY_OUTCOME' in df.columns:
            df['stillbirth_risk'] = df['DELIVERY_OUTCOME'].str.lower().isin(['stillbirth', 'still birth']).astype(int)
        else:
            df['stillbirth_risk'] = 0  # Default if not available
        
        # 2. Premature birth risk (from ANC data)
        if 'NO_OF_WEEKS_max' in df.columns:
            df['premature_birth_risk'] = (df['NO_OF_WEEKS_max'] < 37).astype(int)
        else:
            df['premature_birth_risk'] = 0
        
        # 3. Maternal mortality risk (from delivery data)
        if 'IS_MOTHER_ALIVE' in df.columns:
            df['maternal_mortality_risk'] = (df['IS_MOTHER_ALIVE'] == 0).astype(int)
        else:
            df['maternal_mortality_risk'] = 0
        
        # 4. Birth defect risk (from child data)
        if 'IS_DEFECTIVE_BIRTH_max' in df.columns:
            df['birth_defect_risk'] = (df['IS_DEFECTIVE_BIRTH_max'] == 1).astype(int)
        else:
            df['birth_defect_risk'] = 0
        
        # 5. ANC dropout (from pregnancy data flags)
        if all(col in df.columns for col in ['MISSANC1FLG', 'MISSANC2FLG', 'MISSANC3FLG', 'MISSANC4FLG']):
            df['anc_dropout'] = ((df['MISSANC1FLG'] + df['MISSANC2FLG'] + 
                                 df['MISSANC3FLG'] + df['MISSANC4FLG']) >= 2).astype(int)
        else:
            df['anc_dropout'] = 0
        
        # 6. High-risk pregnancy (composite of multiple factors)
        risk_factors = []
        
        # Age risk
        if 'AGE' in df.columns:
            risk_factors.append((df['AGE'] < 18) | (df['AGE'] > 35))
        
        # Obstetric history risk
        if 'ABORTIONS' in df.columns:
            risk_factors.append(df['ABORTIONS'] >= 2)
        if 'DEATH' in df.columns:
            risk_factors.append(df['DEATH'] > 0)
        
        # Clinical risk
        if 'HEMOGLOBIN_min' in df.columns:
            risk_factors.append(df['HEMOGLOBIN_min'] < 9)
        if 'BP_last' in df.columns:
            bp_systolic = df['BP_last'].astype(str).str.extract(r'(\d+)')[0].astype(float)
            risk_factors.append(bp_systolic >= 140)
        if 'TWIN_PREGNANCY_max' in df.columns:
            risk_factors.append(df['TWIN_PREGNANCY_max'] == 1)
        
        # High-risk if 2 or more risk factors
        if risk_factors:
            risk_count = sum(risk_factors)
            df['high_risk_pregnancy'] = (risk_count >= 2).astype(int)
        else:
            df['high_risk_pregnancy'] = 0
        
        # Print distribution
        for target in self.prediction_targets.keys():
            if target in df.columns:
                rate = df[target].mean()
                count = df[target].sum()
                print(f"{self.prediction_targets[target]['name']}: {count:,} cases ({rate:.2%})")
        
        return df
    
    def engineer_features(self, df):
        """
        Feature engineering using only available columns
        """
        print("\nEngineering features...")
        df = df.copy()
        
        # Age features
        if 'AGE' in df.columns:
            df['age_adolescent'] = (df['AGE'] < 18).astype(int)
            df['age_elderly'] = (df['AGE'] > 35).astype(int)
            df['age_very_young'] = (df['AGE'] < 16).astype(int)
            df['age_risk_score'] = df['age_adolescent'] + df['age_elderly'] * 2 + df['age_very_young'] * 3
        
        # Obstetric history
        if all(col in df.columns for col in ['GRAVIDA', 'PARITY', 'ABORTIONS']):
            df['multigravida'] = (df['GRAVIDA'] > 1).astype(int)
            df['grand_multipara'] = (df['PARITY'] > 5).astype(int)
            df['previous_loss'] = (df['ABORTIONS'] > 0).astype(int)
            df['recurrent_loss'] = (df['ABORTIONS'] >= 2).astype(int)
        
        # ANC visit patterns
        if 'TOTAL_ANC_VISITS' in df.columns:
            df['inadequate_anc'] = (df['TOTAL_ANC_VISITS'] < 4).astype(int)
            df['no_anc'] = (df['TOTAL_ANC_VISITS'] == 0).astype(int)
        
        # Clinical measurements
        if 'HEMOGLOBIN_mean' in df.columns:
            df['anemia_mild'] = ((df['HEMOGLOBIN_mean'] >= 10) & (df['HEMOGLOBIN_mean'] < 11)).astype(int)
            df['anemia_moderate'] = ((df['HEMOGLOBIN_mean'] >= 7) & (df['HEMOGLOBIN_mean'] < 10)).astype(int)
            df['anemia_severe'] = (df['HEMOGLOBIN_mean'] < 7).astype(int)
        
        # BMI calculation
        if all(col in df.columns for col in ['WEIGHT_max', 'HEIGHT']):
            df['HEIGHT'] = df['HEIGHT'].replace(0, np.nan)
            df['BMI'] = df['WEIGHT_max'] / ((df['HEIGHT'] / 100) ** 2)
            df['BMI'] = df['BMI'].replace([np.inf, -np.inf], np.nan)
            df['underweight'] = (df['BMI'] < 18.5).astype(int)
            df['obese'] = (df['BMI'] > 30).astype(int)
        
        # Blood pressure
        if 'BP_last' in df.columns:
            bp_split = df['BP_last'].astype(str).str.extract(r'(\d+)[/\s]+(\d+)')
            df['systolic_bp'] = pd.to_numeric(bp_split[0], errors='coerce')
            df['diastolic_bp'] = pd.to_numeric(bp_split[1], errors='coerce')
            df['hypertension'] = ((df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90)).astype(int)
            df['severe_htn'] = ((df['systolic_bp'] >= 160) | (df['diastolic_bp'] >= 110)).astype(int)
        
        # Mental health
        if 'PHQ_SCORE_max' in df.columns:
            df['depression'] = (df['PHQ_SCORE_max'] >= 10).astype(int)
        if 'GAD_SCORE_max' in df.columns:
            df['anxiety'] = (df['GAD_SCORE_max'] >= 10).astype(int)
        
        # Birth weight features
        if 'WEIGHT_min' in df.columns:
            df['low_birth_weight'] = (df['WEIGHT_min'] < 2500).astype(int)
            df['very_low_birth_weight'] = (df['WEIGHT_min'] < 1500).astype(int)
        
        print(f"Features engineered. Total features: {len(df.columns)}")
        return df
    
    def create_stratified_sample(self, df, target_column):
        """
        Create stratified sample ensuring all critical cases are included
        """
        print(f"\nCreating stratified sample for {target_column}...")
        
        # Identify critical cases that must be included
        critical_indices = set()
        
        # Always include maternal deaths
        if 'maternal_mortality_risk' in df.columns:
            maternal_deaths = df[df['maternal_mortality_risk'] == 1].index
            critical_indices.update(maternal_deaths)
            print(f"Including {len(maternal_deaths)} maternal deaths")
        
        # Always include stillbirths
        if 'stillbirth_risk' in df.columns:
            stillbirths = df[df['stillbirth_risk'] == 1].index
            critical_indices.update(stillbirths)
            print(f"Including {len(stillbirths)} stillbirths")
        
        # Always include child deaths
        if 'IS_CHILD_DEATH_max' in df.columns:
            child_deaths = df[df['IS_CHILD_DEATH_max'] == 1].index
            critical_indices.update(child_deaths)
            print(f"Including {len(child_deaths)} child deaths")
        
        # Calculate remaining sample size
        remaining_size = self.sample_size - len(critical_indices)
        
        if remaining_size <= 0:
            return df.loc[list(critical_indices)]
        
        # Get non-critical indices
        non_critical_indices = set(df.index) - critical_indices
        non_critical_df = df.loc[list(non_critical_indices)]
        
        # Stratified sampling of remaining cases
        if target_column in non_critical_df.columns and non_critical_df[target_column].nunique() > 1:
            # Stratify by target variable
            sample_frac = min(1.0, remaining_size / len(non_critical_df))
            additional_sample = non_critical_df.groupby(target_column).apply(
                lambda x: x.sample(frac=sample_frac, random_state=42)
            ).reset_index(drop=True)
        else:
            # Random sample if stratification not possible
            additional_sample = non_critical_df.sample(
                n=min(remaining_size, len(non_critical_df)), 
                random_state=42
            )
        
        # Combine critical cases with additional sample
        final_indices = list(critical_indices) + list(additional_sample.index)
        final_sample = df.loc[final_indices]
        
        print(f"Final sample size: {len(final_sample)}")
        print(f"Target distribution: {final_sample[target_column].value_counts()}")
        
        return final_sample
    
    def train_model(self, df, target_column, config):
        """
        Train a single LightGBM model for one target
        """
        print(f"\nTraining model for {config['name']}...")
        
        # Create sample
        df_sample = self.create_stratified_sample(df, target_column)
        
        # Prepare features
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['MOTHER_ID', 'CHILD_ID'] + list(self.prediction_targets.keys())
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = df_sample[feature_cols].fillna(0)
        y = df_sample[target_column].astype(int)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        print(f"Positive cases in training: {y_train.sum()}")
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': -1,
            'max_depth': 8,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'early_stopping_rounds': 50,
            'num_boost_round': 1000,
            'scale_pos_weight': config['scale_pos_weight']
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        start_time = datetime.now()
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            callbacks=[lgb.log_evaluation(100)]
        )
        training_time = (datetime.now() - start_time).total_seconds() / 60
        
        # Evaluate
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba >= config['threshold']).astype(int)
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Recall: {report['1']['recall']:.3f}")
        print(f"Precision: {report['1']['precision']:.3f}")
        print(f"Training time: {training_time:.2f} minutes")
        
        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'metrics': {
                'auc': auc,
                'recall': report['1']['recall'],
                'precision': report['1']['precision'],
                'confusion_matrix': cm
            },
            'feature_importance': feature_imp_df,
            'training_time': training_time
        }
    
    def train_all_models(self, data_path):
        """
        Train models for all 6 prediction targets
        """
        print("="*60)
        print("TRAINING ALL MATERNAL HEALTH PREDICTION MODELS")
        print("="*60)
        
        # Load data
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df):,} records")
        
        # Create targets
        df = self.create_target_variables(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Train each model
        for target, config in self.prediction_targets.items():
            if target in df.columns and df[target].sum() > 0:
                result = self.train_model(df, target, config)
                self.models[target] = result['model']
                self.feature_importance[target] = result['feature_importance']
                
                print(f"\nCompleted: {config['name']}")
                print(f"Top 5 features:")
                print(result['feature_importance'].head())
            else:
                print(f"\nSkipped {config['name']}: No positive cases found")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
    
    def create_early_warning_score(self, patient_data):
        """
        Early Warning System: Calculate composite risk score
        """
        warnings = []
        risk_score = 0
        
        # Check each model's prediction
        predictions = self.predict_all_risks(patient_data)
        
        for target, pred in predictions.items():
            if pred['risk_level'] == 'CRITICAL':
                risk_score += 10
                warnings.append(f"CRITICAL: {pred['name']} ({pred['probability']:.1%})")
            elif pred['risk_level'] == 'HIGH':
                risk_score += 5
                warnings.append(f"HIGH: {pred['name']} ({pred['probability']:.1%})")
            elif pred['risk_level'] == 'MODERATE':
                risk_score += 2
        
        # Overall early warning level
        if risk_score >= 20:
            warning_level = "CRITICAL - IMMEDIATE ACTION"
        elif risk_score >= 10:
            warning_level = "HIGH - URGENT REVIEW"
        elif risk_score >= 5:
            warning_level = "MODERATE - CLOSE MONITORING"
        else:
            warning_level = "LOW - ROUTINE CARE"
        
        return {
            'warning_level': warning_level,
            'risk_score': risk_score,
            'warnings': warnings,
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_all_risks(self, patient_data):
        """
        Make predictions for all 6 outcomes
        """
        # Convert to DataFrame
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Engineer features
        patient_df = self.engineer_features(patient_df)
        
        predictions = {}
        
        for target, config in self.prediction_targets.items():
            if target in self.models:
                model = self.models[target]
                
                # Get features
                feature_cols = model.feature_name()
                for col in feature_cols:
                    if col not in patient_df.columns:
                        patient_df[col] = 0
                
                X = patient_df[feature_cols].fillna(0)
                prob = model.predict(X, num_iteration=model.best_iteration)[0]
                
                # Risk level
                threshold = config['threshold']
                if prob >= threshold * 2:
                    risk_level = 'CRITICAL'
                elif prob >= threshold:
                    risk_level = 'HIGH'
                elif prob >= threshold * 0.5:
                    risk_level = 'MODERATE'
                else:
                    risk_level = 'LOW'
                
                predictions[target] = {
                    'name': config['name'],
                    'probability': float(prob),
                    'risk_level': risk_level,
                    'threshold': threshold
                }
        
        return predictions


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = TelanganaMaternalHealthPipeline(sample_size=500000)
    
    # Option 1: Merge your Excel files first
    """
    merged_df = pipeline.merge_data_files(
        pregnancy_file='pregnancy_sample.xlsx',
        anc_file='anc_sample.xlsx',
        delivery_file='delivery_sample.xlsx',
        child_file='child_sample.xlsx',
        output_file='telangana_merged.parquet'
    )
    """
    
    # Option 2: Use pre-merged parquet file
    # Assume you have already merged your data
    data_path = 'telangana_merged.parquet'
    
    # Train all models
    pipeline.train_all_models(data_path)
    
    # Example: Predict for a new patient
    patient = {
        'AGE': 17,
        'GRAVIDA': 2,
        'PARITY': 1,
        'ABORTIONS': 1,
        'HEIGHT': 150,
        'HEMOGLOBIN_mean': 8.5,
        'BP_last': '145/95',
        'WEIGHT_max': 48,
        'NO_OF_WEEKS_max': 32,
        'TOTAL_ANC_VISITS': 2,
        'MISSANC1FLG': 1,
        'PHQ_SCORE_max': 12
    }
    
    # Get all predictions
    predictions = pipeline.predict_all_risks(patient)
    
    print("\n" + "="*60)
    print("PATIENT RISK ASSESSMENT")
    print("="*60)
    
    for target, pred in predictions.items():
        emoji = "🔴" if pred['risk_level'] == 'CRITICAL' else "🟡" if pred['risk_level'] == 'HIGH' else "🟢"
        print(f"{emoji} {pred['name']:<35} {pred['risk_level']:<10} {pred['probability']:.1%}")
    
    # Get early warning score
    early_warning = pipeline.create_early_warning_score(patient)
    
    print(f"\n⚠️  EARLY WARNING SYSTEM")
    print(f"Status: {early_warning['warning_level']}")
    print(f"Risk Score: {early_warning['risk_score']}")
    if early_warning['warnings']:
        print("Alerts:")
        for warning in early_warning['warnings']:
            print(f"  - {warning}")

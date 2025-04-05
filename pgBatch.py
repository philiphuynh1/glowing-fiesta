import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import scipy.stats as stats

st.set_page_config(page_title="Batch Job", layout="centered")

@st.cache_data
def load_grouped_data(file_path='Grouped_Data.csv'):
    return pd.read_csv(file_path)

def preprocess_data(df):
    target_col = 'Waste %'
    y = df[target_col].astype(np.float32)
    selected_features = [
        'Flute Code Grouped', 'Qty Bucket', 'Component Code Grouped',
        'Machine Group 1', 'Last Operation', 'qty_ordered',
        'number_up_entry_grouped', 'OFFSET?', 'Operation', 'Test Code'
    ]
    X = df[selected_features]
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded, y

@st.cache_resource
def train_models(X_encoded, y, n_iterations=10):  # Reduced iterations here
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42
    )
    
    best_params = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1  # Use all available cores
    }
    
    trained_models = []
    new_train_nrmse_list, new_test_nrmse_list, original_test_nrmse_list = [], [], []
    new_train_mse_list, new_test_mse_list, original_test_mse_list = [], [], []
    
    def compute_nrmse(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        norm_factor = y_true.max() - y_true.min()
        return rmse / norm_factor if norm_factor != 0 else rmse
    
    with st.spinner(f"Training models ({n_iterations} iterations)..."):
        for i in range(n_iterations):
            X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(
                X_train_full, y_train_full, test_size=0.20, random_state=i
            )
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_new_train, y_new_train)
            trained_models.append(model)
    
            pred_new_train = model.predict(X_new_train)
            pred_new_test = model.predict(X_new_test)
            new_train_mse_list.append(mean_squared_error(y_new_train, pred_new_train))
            new_test_mse_list.append(mean_squared_error(y_new_test, pred_new_test))
            new_train_nrmse_list.append(compute_nrmse(y_new_train, pred_new_train))
            new_test_nrmse_list.append(compute_nrmse(y_new_test, pred_new_test))
    
            pred_original_test = model.predict(X_test)
            original_test_mse_list.append(mean_squared_error(y_test, pred_original_test))
            original_test_nrmse_list.append(compute_nrmse(y_test, pred_original_test))
    
    metrics = {
        'avg_new_train_nrmse': np.mean(new_train_nrmse_list),
        'avg_new_test_nrmse': np.mean(new_test_nrmse_list),
        'avg_original_test_nrmse': np.mean(original_test_nrmse_list),
        'avg_new_train_mse': np.mean(new_train_mse_list),
        'avg_new_test_mse': np.mean(new_test_mse_list),
        'avg_original_test_mse': np.mean(original_test_mse_list)
    }
    return trained_models, metrics, X_test, y_test

def predict_jobs(trained_models, X_encoded, uploaded_file):
    df_jobs = pd.read_excel(uploaded_file)
    df_jobs = df_jobs[~df_jobs['Machine Group 1'].str.strip().eq('PURCHASED BOARD/OFFSET')]
    feature_cols = [
        'Flute Code', 'Qty Bucket', 'Component Code', 'Machine Group 1',
        'Last Operation', 'qty_ordered', 'number_up_entry_1', 'OFFSET?',
        'Operation', 'Test Code'
    ]
    X_jobs = df_jobs[feature_cols].copy()
    X_jobs_encoded = pd.get_dummies(X_jobs, drop_first=True)
    X_jobs_encoded = X_jobs_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    
    all_preds = np.array([model.predict(X_jobs_encoded) for model in trained_models])
    all_preds = all_preds.T  # shape: (num_jobs, n_iterations)
    df_jobs['pred_mean'] = all_preds.mean(axis=1)
    df_jobs['pred_std'] = all_preds.std(axis=1)
    
    grouped = df_jobs.groupby(['job_number', 'Machine Group 1']).agg({
        'pred_mean': 'mean',
        'pred_std': 'mean',
        'qty_ordered': 'first'
    }).reset_index()
    
    grouped.to_excel('Predicted_Jobs_Grouped.xlsx', index=False)
    return grouped

def compute_optimal_Q1(final_demand, machine_sequence, Cu, Co, n_simulations=1000, random_seed=42):
    """
    Reduced simulation count to 1000 for faster computation.
    """
    np.random.seed(random_seed)
    multipliers = np.ones(n_simulations)
    for machine_name, waste_mean, waste_std in machine_sequence:
        w_mean = waste_mean / 100.0
        w_std  = waste_std / 100.0
        waste_samples = np.random.normal(loc=w_mean, scale=w_std, size=n_simulations)
        multipliers *= (1 - waste_samples)
    
    Q1_samples = final_demand / multipliers
    Q1_mean = np.mean(Q1_samples)
    Q1_std  = np.std(Q1_samples)
    
    critical_ratio = Cu / (Cu + Co)
    Z_value = stats.norm.ppf(critical_ratio)
    safety_stock = Z_value * Q1_std
    Q1_optimal = Q1_mean + safety_stock
    
    return Q1_optimal, Q1_mean, Q1_std

# --- Streamlit App Flow ---
if st.button("Return to Home Page"):
    st.switch_page("pgTitle.py")

st.markdown("### CODING IS A BITCH:")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    st.success("Upload complete!")
    
    # Load and preprocess training data
    df_grouped = load_grouped_data()
    X_encoded, y = preprocess_data(df_grouped)
    
    # Train models with fewer iterations
    trained_models, metrics, X_test, y_test = train_models(X_encoded, y, n_iterations=10)
    st.write("#### Training Metrics")
    st.write(metrics)
    
    # Predict on new jobs data and save grouped predictions
    grouped = predict_jobs(trained_models, X_encoded, uploaded_file)
    st.write("Grouped predictions saved to Predicted_Jobs_Grouped.xlsx")
    
    # PHASE 2: Compute optimal Q1 values for each job using reduced simulation count
    Cu = 3.41  # Underage cost
    Co = 0.71  # Overage cost
    jobs_df = pd.read_excel('Predicted_Jobs_Grouped.xlsx')
    
    results = []
    for job_number, group in jobs_df.groupby('job_number', sort=False):
        group = group.sort_index()
        final_demand = group.iloc[0]['qty_ordered']
        machine_sequence = []
        for idx, row in group.iterrows():
            machine_name = row['Machine Group 1']
            waste_mean = row['pred_mean']
            waste_std = row['pred_std']
            machine_sequence.append((machine_name, waste_mean, waste_std))
        Q1_optimal, Q1_mean, Q1_std = compute_optimal_Q1(final_demand, machine_sequence, Cu, Co, n_simulations=1000)
        feed = Q1_optimal
        for order, (machine_name, waste_mean, waste_std) in enumerate(machine_sequence, start=1):
            results.append({
                'job_number': job_number,
                'final_demand': final_demand,
                'Q1_optimal': Q1_optimal,
                'Q1_mean': Q1_mean,
                'Q1_std': Q1_std,
                'machine_order': order,
                'machine_name': machine_name,
                'machine_input': feed
            })
            feed = feed * (1 - waste_mean / 100.0)
    
    results_df = pd.DataFrame(results)
    results_df.to_excel('Job_Machine_Quantities.xlsx', index=False)
    st.write("Results saved to Job_Machine_Quantities.xlsx")
    
    # File download section
    download_path = "Job_Machine_Quantities.xlsx"
    if os.path.exists(download_path):
        with open(download_path, "rb") as f:
            st.download_button(label="DOWNLOAD", data=f, file_name="Job_Machine_Quantities.xlsx")
    else:
        st.write("Job_Machine_Quantities.xlsx not found!")
else:
    st.write("To view, import a file first.")

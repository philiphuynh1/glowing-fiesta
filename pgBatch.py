import streamlit as st
import math
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import scipy.stats as stats




# FIRST SECTION

# Set page title
st.set_page_config(page_title="Batch Job", layout="centered")

if st.button(label="Return to Home Page", key=None, help=None, type="secondary", icon=None,
            disabled=False, use_container_width=False):
    st.switch_page("pgTitle.py")

# Title
st.markdown("### CODING IS A BITCH:")

# File Uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    st.success("Upload complete!")

# SECOND SECTION
st.markdown("---")  # Horizontal line

#  File Download
st.markdown("# OPTIMAL STARTING QUANTITIES")

if uploaded_file:
    
    # PHASE 1
    
    # -------------------------------
    # Helper Function: NRMSE
    # -------------------------------
    def compute_nrmse(y_true, y_pred):
        """Compute Normalized RMSE = RMSE / (max - min)"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        norm_factor = y_true.max() - y_true.min()
        return rmse / norm_factor if norm_factor != 0 else rmse

    # -------------------------------
    # 1. Data Loading & Preprocessing (Main Training Data)
    # -------------------------------
    data_file = 'Grouped_Data.xlsx'
    df = pd.read_excel(data_file)

    target_col = 'Waste %'
    y = df[target_col].astype(np.float32)

    selected_features = [
        'Flute Code Grouped',
        'Qty Bucket',
        'Component Code Grouped',
        'Machine Group 1',
        'Last Operation',
        'qty_ordered',
        'number_up_entry_grouped',
        'OFFSET?',
        'Operation',
        'Test Code'
    ]
    X = df[selected_features]

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    # -------------------------------
    # 2. Split data: 80% training, 20% test
    # -------------------------------
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42
    )

    # -------------------------------
    # 3. XGBoost Hyperparameters
    # -------------------------------
    best_params = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'reg:squarederror',
        'random_state': 42
    }

    # -------------------------------
    # 4. Prepare for Iterations
    # -------------------------------
    n_iterations = 100

    # Containers for NRMSE
    new_train_nrmse_list = []
    new_test_nrmse_list = []
    original_test_nrmse_list = []

    # Containers for MSE
    new_train_mse_list = []
    new_test_mse_list = []
    original_test_mse_list = []

    # List to store each trained model
    trained_models = []

    # -------------------------------
    # 5. Main Loop (100 iterations)
    # -------------------------------
    for i in range(n_iterations):
        # Split the original training data (80%) into new training (80%) and new testing (20%)
        X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(
            X_train_full, y_train_full, test_size=0.20, random_state=i
        )

        # Train an XGBoost model on the new training split
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_new_train, y_new_train)

        # Store this trained model
        trained_models.append(model)

        # Predictions on new training and new testing splits
        pred_new_train = model.predict(X_new_train)
        pred_new_test = model.predict(X_new_test)

        # Compute and store MSE for new training and new testing
        mse_new_train = mean_squared_error(y_new_train, pred_new_train)
        mse_new_test = mean_squared_error(y_new_test, pred_new_test)
        new_train_mse_list.append(mse_new_train)
        new_test_mse_list.append(mse_new_test)

        # Compute and store NRMSE for new training and new testing
        nrmse_new_train = compute_nrmse(y_new_train, pred_new_train)
        nrmse_new_test = compute_nrmse(y_new_test, pred_new_test)
        new_train_nrmse_list.append(nrmse_new_train)
        new_test_nrmse_list.append(nrmse_new_test)

        # Predict on the original 20% test set
        pred_original_test = model.predict(X_test)
        mse_original_test = mean_squared_error(y_test, pred_original_test)
        original_test_mse_list.append(mse_original_test)

        nrmse_original_test = compute_nrmse(y_test, pred_original_test)
        original_test_nrmse_list.append(nrmse_original_test)

    # -------------------------------
    # 6. Summaries & Outputs (Training Performance)
    # -------------------------------
    # A) NRMSE Averages
    avg_new_train_nrmse = np.mean(new_train_nrmse_list)
    avg_new_test_nrmse = np.mean(new_test_nrmse_list)
    avg_original_test_nrmse = np.mean(original_test_nrmse_list)

    print("Average New Training NRMSE (100 iterations):", avg_new_train_nrmse)
    print("Average New Testing NRMSE (100 iterations):", avg_new_test_nrmse)
    print("Average Original Testing NRMSE (100 iterations):", avg_original_test_nrmse)

    # B) MSE Averages
    avg_new_train_mse = np.mean(new_train_mse_list)
    avg_new_test_mse = np.mean(new_test_mse_list)
    avg_original_test_mse = np.mean(original_test_mse_list)

    print("\nAverage New Training MSE (100 iterations):", avg_new_train_mse)
    print("Average New Testing MSE (100 iterations):", avg_new_test_mse)
    print("Average Original Testing MSE (100 iterations):", avg_original_test_mse)

    # ------------------------------------------------------------------
    # 7. Predicting on a New Jobs File (Ignoring PURCHASED BOARD/OFFSET)
    # ------------------------------------------------------------------

    new_jobs_file = uploaded_file
    df_jobs = pd.read_excel(new_jobs_file)

    # (A) Filter out the rows where 'Machine Group 1' == 'PURCHASED BOARD/OFFSET'
    df_jobs = df_jobs[~df_jobs['Machine Group 1'].str.strip().eq('PURCHASED BOARD/OFFSET')]

    # (B) Keep columns needed for prediction (matching your training features)
    feature_cols = [
        'Flute Code',  # must match exactly what you had in training
        'Qty Bucket',
        'Component Code',
        'Machine Group 1',
        'Last Operation',
        'qty_ordered',
        'number_up_entry_1',
        'OFFSET?',
        'Operation',
        'Test Code'
    ]

    X_jobs = df_jobs[feature_cols].copy()

    # (C) One-hot encode new jobs data & align columns
    X_jobs_encoded = pd.get_dummies(X_jobs, drop_first=True)
    X_jobs_encoded = X_jobs_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # (D) Predict using all 100 trained models
    all_preds = []
    for model in trained_models:
        preds = model.predict(X_jobs_encoded)
        all_preds.append(preds)

    # Convert list to NumPy array, shape: (100, num_rows_in_new_jobs)
    all_preds = np.array(all_preds)
    # Transpose to shape: (num_rows_in_new_jobs, 100)
    all_preds = all_preds.T

    # Compute mean & std across the 100 predictions for each row
    df_jobs['pred_mean'] = all_preds.mean(axis=1)
    df_jobs['pred_std'] = all_preds.std(axis=1)

    # (E) Group by job_number and Machine Group 1 and also carry over qty_ordered for each job.
    #     Here we use the first qty_ordered encountered per group.
    group_cols = ['job_number', 'Machine Group 1']
    grouped = df_jobs.groupby(group_cols).agg({
        'pred_mean': 'mean',
        'pred_std': 'mean',
        'qty_ordered': 'first'
    }).reset_index()

    # Display grouped results
    print("\nPredictions for each job-machine combination (mean & std over 100 models) with qty_ordered:")
    print(grouped)

    # -------------------------------
    # 8. Exporting Predictions to Excel
    # -------------------------------
    output_file_grouped = 'Predicted_Jobs_Grouped.xlsx'
    grouped.to_excel(output_file_grouped, index=False)
    print("Grouped predictions (with qty_ordered) saved to", output_file_grouped)


    # PHASE 2

    def compute_optimal_Q1(final_demand, machine_sequence, Cu, Co, n_simulations=10000, random_seed=42):
        """
        Computes the optimal Q1 (initial feed quantity) using a simulation-based approach.

        Parameters:
            final_demand (float): The final required demand (D) for the job.
            machine_sequence (list of tuples): Each tuple is (machine_name, waste_mean, waste_std),
                                            with waste values in percentages.
            Cu (float): Underage cost (cost per unit short).
            Co (float): Overage cost (cost per unit leftover).
            n_simulations (int): Number of simulation iterations.
            random_seed (int): Seed for reproducibility.

        Returns:
            Q1_optimal (float): The total quantity to be fed into Machine 1 (includes safety stock).
            Q1_mean (float): Mean of the simulated Q1 values.
            Q1_std (float): Standard deviation of the simulated Q1 values.
        """
        # Set the random seed for reproducibility
        np.random.seed(random_seed)

        # Start with a multiplier of 1 for each simulation run
        multiplier = np.ones(n_simulations)

        # Multiply by (1 - waste) for each machine in the sequence
        for machine_name, waste_mean, waste_std in machine_sequence:
            w_mean = waste_mean / 100.0  # convert percentage to decimal
            w_std  = waste_std / 100.0
            waste_samples = np.random.normal(loc=w_mean, scale=w_std, size=n_simulations)
            multiplier *= (1 - waste_samples)

        # Calculate Q1 for each simulation run so that the final output equals final_demand
        Q1_samples = final_demand / multiplier

        # Compute mean and standard deviation of Q1 samples
        Q1_mean = np.mean(Q1_samples)
        Q1_std  = np.std(Q1_samples)

        # Compute safety stock using the Newsvendor model
        critical_ratio = Cu / (Cu + Co)
        Z_value = stats.norm.ppf(critical_ratio)
        safety_stock = Z_value * Q1_std

        # Total order quantity (optimal feed into Machine 1)
        Q1_optimal = Q1_mean + safety_stock

        return Q1_optimal, Q1_mean, Q1_std

    # =============================================
    # PARAMETERS (these are the same for all jobs)
    # =============================================
    Cu = 3.41       # Underage cost
    Co = 0.71       # Overage cost
    n_simulations = 10000

    # =============================================
    # READ THE INPUT FILE
    # =============================================
    # The Excel file is assumed to have the following columns:
    # 'job_number', 'qty_ordered', 'machine_name', 'waste_mean', 'waste_std'
    input_file = 'Predicted_Jobs_Grouped.xlsx'
    jobs_df = pd.read_excel(input_file)

    # (Optional) Strip extra spaces from column names
    jobs_df.columns = [col.strip() for col in jobs_df.columns]

    # =============================================
    # PROCESS EACH JOB (group by job_number)
    # =============================================
    # We'll produce one row per machine for each job, including:
    # - job_number, final_demand, Q1_optimal, Q1_mean, Q1_std (all repeated for the job)
    # - machine_order (the sequence number)
    # - machine_name, and the input feed into that machine.
    results = []

    # Group by job_number while preserving the file order
    for job_number, group in jobs_df.groupby('job_number', sort=False):
        # Sort the group by the original row order (if necessary)
        group = group.sort_index()
        print(group.columns)

        # Use the first row's qty_ordered as the final demand for the job
        final_demand = group.iloc[0]['qty_ordered']

        # Build the machine sequence for this job from the rows (in order)
        machine_sequence = []
        for idx, row in group.iterrows():
            # Adjust these column names if needed based on your file
            machine_name = row['Machine Group 1']
            waste_mean = row['pred_mean']
            waste_std = row['pred_std']
            machine_sequence.append((machine_name, waste_mean, waste_std))

        # Compute Q1_optimal using the simulation for this job's final demand and machine sequence
        Q1_optimal, Q1_mean, Q1_std = compute_optimal_Q1(final_demand, machine_sequence, Cu, Co, n_simulations)

        # Now, compute the feed (input) into each machine along the sequence.
        # The input to Machine 1 is Q1_optimal.
        # The input to each subsequent machine is the output from the previous machine,
        # calculated as the previous feed multiplied by (1 - waste_mean/100).
        feed = Q1_optimal
        for order, (machine_name, waste_mean, waste_std) in enumerate(machine_sequence, start=1):
            machine_input = feed
            results.append({
                'job_number': job_number,
                'final_demand': final_demand,
                'Q1_optimal': Q1_optimal,
                'Q1_mean': Q1_mean,
                'Q1_std': Q1_std,
                'machine_order': order,
                'machine_name': machine_name,
                'machine_input': machine_input
            })
            # Update the feed for the next machine
            feed = machine_input * (1 - waste_mean / 100.0)

    # =============================================
    # CREATE AND EXPORT THE RESULTS
    # =============================================
    results_df = pd.DataFrame(results)
    print(results_df)

    output_file = 'Job_Machine_Quantities.xlsx'
    results_df.to_excel(output_file, index=False)
    print("Results saved to", output_file)


# OUTPUT

    st.write("##### Important features")
    st.write('Flute Code Grouped, Qty Bucket, Component Code Grouped, Machine Group 1, Last Operation, qty_ordered, number_up_entry_grouped, OFFSET?, Operation, Test Code')


if uploaded_file:
    download_path = "Job_Machine_Quantities.xlsx"
    if os.path.exists(download_path):
        with open(download_path, "rb") as f:
            st.download_button(label="DOWNLOAD", data=f, file_name="Job_Machine_Quantities.xlsx")
    else:
        st.write("Job_Machine_Quantities.xlsx not found!")


    # Dropdown Selection for Quick View
    st.markdown("## QUICK VIEW")

    if os.path.exists(download_path):
        df = pd.read_excel(download_path)
    else:
        st.error("Job_Machine_Quantities.xlsx not found!")

    # **Cache the file loading function to speed up performance**
    @st.cache_data
    def load_excel(file_path):
        return pd.read_excel(file_path)

    # Load Excel File (Cached for speed)
    file_path = "Job_Machine_Quantities.xlsx"
    df = load_excel(file_path)


    if df.shape[1] >= 8:
        # Extract Job Numbers from Column 1
        STjob_numbers = df.iloc[:, 0].dropna().unique().tolist()  

        # Create a selectbox for job numbers
        STjob_selection = st.selectbox("Select a Job Number", STjob_numbers, index=0)

        # Filter the dataframe based on selected job number
        filtered_df = df[df.iloc[:, 0] == STjob_selection]

        if not filtered_df.empty:
        # Read the Machine Input (Column 8) and Final Demand (Column 2)
            STmachine_input = filtered_df.iloc[:, 7].values  # Column index 8 (0-based index → 7)
            STfinal_demand = filtered_df.iloc[:, 1].unique()  # Column index 2 (0-based index → 1)
            STmachine_name = filtered_df.iloc[:, 6].values  # Column index 7 (0-based index → 6)

            final_demand_val = float(STfinal_demand[0]) 
            
            initial_input = float(STmachine_input[0])
            initial_input = round(initial_input, 3)

            # Display Results
            st.write(f"### Selected Job: {STjob_selection}")
            st.write(f"#### Starting Input: {initial_input}")

            col1,col2 = st.columns(2)
            with col1:
                st.write("**Machine Name:**")
                st.write(STmachine_name)
            with col2:
                st.write("**Machine Input:**")
                st.write(STmachine_input)

            st.write(f"#### Final Output: {final_demand_val}")
        else:
            st.error("No data found for the selected job number.")
    else:
        st.error("The uploaded file does not contain any columns.")

else:
    st.write("To view, import a file first.")
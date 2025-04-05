import streamlit as st
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from pgBatch import simulate as simulate_batch
from pgSingle import simulate as simulate_single
from pgTitle import display_title

st.set_page_config(layout="wide")
display_title("swLogo.png")

# Caching Excel loading
@st.cache_data
def load_excel(path):
    return pd.read_excel(path)

# Load data
jobs_to_predict = load_excel("JobsToPredict.xlsx")
grouped_data = load_excel("Grouped_Data.xlsx")
job_machine_quantities = load_excel("Job_Machine_Quantities.xlsx")

st.sidebar.title("Controls")
run_prediction = st.sidebar.button("Run Prediction")
num_iterations = st.sidebar.slider("Simulation Iterations", 100, 5000, 1000)

if run_prediction:
    with st.spinner("Training models and running predictions..."):
        results = []

        for job_id, group in grouped_data.groupby("Job ID"):
            X = group.drop(columns=["Waste"])
            y = group["Waste"]

            @st.cache_resource
            def train_model(X, y):
                model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
                model.fit(X, y)
                return model

            model = train_model(X, y)

            job_data = jobs_to_predict[jobs_to_predict["Job ID"] == job_id]
            if not job_data.empty:
                pred = model.predict(job_data.drop(columns=["Job ID"]))
                job_data = job_data.copy()
                job_data["Predicted Waste"] = pred
                results.append(job_data)

        if results:
            prediction_df = pd.concat(results, ignore_index=True)
            st.success("Prediction complete.")

            with st.expander("View Prediction Results"):
                st.dataframe(prediction_df)

            with st.spinner("Running simulation for optimal starting quantities..."):
                final_results = simulate_batch(prediction_df, job_machine_quantities, iterations=num_iterations)
                with st.expander("Simulation Output"):
                    st.dataframe(final_results)
        else:
            st.warning("No matching jobs found for prediction.")
else:
    st.info("Click the 'Run Prediction' button in the sidebar to begin.")

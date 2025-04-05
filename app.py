import streamlit as st
import pandas as pd
import os

pg = st.navigation([
    st.Page("pgTitle.py", title="Title", icon=":material/favorite:"),
    st.Page("pgBatch.py", title="Batch", icon="🔥"),
    st.Page("pgSingle.py", title="Single", icon="🔥")
], position='hidden')
pg.run()

@st.cache_data
def load_grouped_data():
    csv_file = "Grouped_Data.csv"
    # Check if the CSV file exists; if not, convert the Excel file.
    if not os.path.exists(csv_file):
        df = pd.read_excel("Grouped_Data.xlsx")
        df.to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)
    return df

# Load data (this is now cached and will only re-run if the file changes)
data = load_grouped_data()

st.title("My Streamlit App")
st.write("Data loaded from Grouped_Data.csv:")
st.dataframe(data)

import streamlit as st
import math

# Set page title
st.set_page_config(page_title="Single Job", layout="centered")

if st.button(label="Return to Home Page", key=None, help=None, type="secondary", icon=None,
            disabled=False, use_container_width=False):
    st.switch_page("pgTitle.py")

# Title
st.markdown("## Enter Job Information:")
    
left, middle, right = st.columns([3, 5, 8], vertical_alignment="top")

with left:
    orderQty = st.number_input("Order Qty", min_value=0)
with middle:
    job1 = st.selectbox("Select Machine 1", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job2 = st.selectbox("Select Machine 2", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job3 = st.selectbox("Select Machine 3", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job4 = st.selectbox("Select Machine 4", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job5 = st.selectbox("Select Machine 5", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
with right:
    col1, col2 = st.columns(2, vertical_alignment="top")
    with col1:
        ftOFFSET = st.selectbox("OFFSET", ["YES", "NO"],
                        index=None,
                        placeholder="Select")
        ftFLUTECODE = st.selectbox("FLUTE CODE",
                        ["0","B","BC","C","E","EB","EC","F","SBS","STRN","X"],
                        index=None,
                        placeholder="Select")
        ftCLOSURE = st.selectbox("CLOSURE TYPE", ["0"],
                        index=None,
                        placeholder="Select")
        ftCOMPONENT = st.selectbox("COMPONENT CODE",
                        ["0", "10PT","12PT","16PT","18PT","20PT","22PT","24PT","28PT","BB","BK","BM","IB","II"
                         "IK","IM","K","KI","KK","KM","M","MK","MM","PK","TB","TI","TK","TM","TSPL"],
                        index=None,
                        placeholder="Select")
        ftROTARY = st.selectbox("ROTARY DC", ["YES", "NO"],
                        index=None,
                        placeholder="Select")
        
    with col2:
        ftTESTCODE = st.number_input("TEST CODE", min_value=0, max_value=999,
                        value=None, placeholder="Enter Value")
        ftNUMBERUP = st.number_input("NUMBER UP ENTRY", min_value=0, max_value=100,
                        value=None, placeholder="Enter Value")
        ftBLANKWIDTH = st.number_input("BLANK WIDTH", min_value=0, value=None,
                        placeholder="Enter Value")
        ftBLANKLENGTH = st.number_input("BLANK LENGTH", min_value=0, value=None,
                        placeholder="Enter Value")
        ftITEMWIDTH = st.number_input("ITEM WIDTH", min_value=0, value=None,
                        placeholder="Enter Value")
        ftITEMLENGTH = st.number_input("ITEM LENGTH", min_value=0, value=None,
                        placeholder="Enter Value")


st.markdown("---")  # Horizontal line

wasteQty = round(0.05 * orderQty)
optQty = round(orderQty - wasteQty)
finalQty = round(orderQty + (0.001 * orderQty))

#  Information Return
("# OPTIMAL STARTING QUANTITIES")

# Starting Quantity
st.write("### Start with: ", optQty)

# Estimated amount of waste
st.write("Estimated Waste: ", wasteQty)

# Finished Quantity
st.write("Final Output Quantity: ", finalQty)
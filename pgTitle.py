import streamlit as st
import math

#Startup Page
st.image("swLogo.png", caption=None, width=200)
st.title("QUANTITY HELLO")

if st.button(label="Single", key=None, help=None, type="primary", icon=None,
            disabled=False, use_container_width=True):
    st.switch_page("pgSingle.py")
if st.button(label="Batch", key=None, help=None, type="primary", icon=None,
            disabled=False, use_container_width=True):
        st.switch_page("pgBatch.py")
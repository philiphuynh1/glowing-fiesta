import streamlit as st
import math



pg = st.navigation([
    st.Page("pgTitle.py", title="Title", icon=":material/favorite:"),
    st.Page("pgBatch.py", title="Batch", icon="🔥"),
    st.Page("pgSingle.py", title="Single", icon="🔥")
], position='hidden')
pg.run()





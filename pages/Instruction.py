import streamlit as st
import pandas as pd
import textwrap


def app():   

    st.write('`Instruction page`')

    st.write("""

    <h2 style="color:Black"><b>Instruction:</b></h3>

    - The tool accepts only csv files. 
    - Step 1. Upload Dataset.

    - Step 2. follow EDA steps to see what each step produces.

    - Step 3. Perform EDA & Anomaly Detection & Treatment (Missing Value & Outlier)

    """ , unsafe_allow_html=True)
    st.write('---')
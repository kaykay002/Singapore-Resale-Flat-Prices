# Packages
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import streamlit as st
import json
from streamlit_option_menu import option_menu
import pickle
import time
#----------------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.header("SINGAPORE RESALE FLAT PRICES PREDICTION")
#_____________________________________________________________________________________________________________
mappings_df = pd.read_csv('mappings.csv')
data = pd.read_csv('rfp.csv')

def town_mapping(town):
    filtered_rows = mappings_df[mappings_df['Original_Town'] == town]
    if not filtered_rows.empty:
        return filtered_rows['Encoded_Town'].iloc[0]
    else:
        return None  

def flat_type_mapping(flat_type):
    filtered_rows = mappings_df[mappings_df['Original_Flat_Type'] == flat_type]
    if not filtered_rows.empty:
        return filtered_rows['Encoded_Flat_Type'].iloc[0]
    else:
        return None 

def flat_model_mapping(flat_model):
    filtered_rows = mappings_df[mappings_df['Original_Flat_Model'] == flat_model]
    if not filtered_rows.empty:
        return filtered_rows['Encoded_Flat_Model'].iloc[0]
    else:
        return None  


def predict_price(month, town1, flat_type1, floor_area_sqm, flat_model1,
                   lease_commence_date, year, storey_start,
                   storey_end, address):
    with open(r"Resale_Flat_Prices_Model_1.pkl", "rb") as f:
        regg_model = pickle.load(f)

    user_data = np.array([[month, town1, flat_type1, floor_area_sqm, flat_model1,
                           lease_commence_date, year, storey_start,
                           storey_end, address]])
    y_pred_1 = regg_model.predict(user_data)
    price = np.exp(y_pred_1[0])

    return round(price)
#----------------------------------------------------------------------------------------------------
#______________________________________________________________________________________________________
with st.sidebar:
    select = option_menu(menu_title="Explore",menu_icon="building",options=["About", "Flat Price Prediction"])

if select=="About":
    st.image("sealion.jpg")
    with open("about.txt", "r") as file:
        file_contents = file.read()
    file_contents

elif select == "Flat Price Prediction":
    st.image("flats.jpg")
    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox("Select the Month", data['month'].unique())
        year = st.selectbox("Select the Year", data['year'].unique())
        town_name = st.selectbox("Select the Town", mappings_df['Original_Town'].unique())
        flat_type_name = st.selectbox("Select the Flat Type", mappings_df['Original_Flat_Type'].unique())
        floor_area_sqm_min = data['floor_area_sqm'].min()
        floor_area_sqm_max = data['floor_area_sqm'].max()
        floor_area_sqm = st.number_input("Enter the Value of Floor Area sqm", min_value=floor_area_sqm_min, max_value=floor_area_sqm_max)

    with col2:
        flat_model_name = st.selectbox("Select the Model of The Flat", mappings_df['Original_Flat_Model'].unique())
        storey_start = st.number_input("Enter the Value of Storey Start")
        storey_end = st.number_input("Enter the Value of Storey End")
        lease_commence_date = st.selectbox("Select the Lease Commence Year", data['lease_commence_date'].unique())
        address = st.selectbox("Select the Address", data['address'].unique())

    month = int(month)
    town = town_mapping(town_name)
    flat_type = flat_type_mapping(flat_type_name)
    floor_area_sqm = int(floor_area_sqm)
    flat_model = flat_model_mapping(flat_model_name)
    lease_commence_date = int(lease_commence_date)
    year = int(year)
    storey_start = int(storey_start)
    storey_end = int(storey_end)
    address = int(address)

    button = st.button("Predict the Price", use_container_width=True)

    if button:
        with st.spinner("🎰 Model at Work......"):
            time.sleep(2)  # Add a short delay for spinner visibility
            pre_price = predict_price(month, town, flat_type, floor_area_sqm, flat_model,
                                    lease_commence_date, year, storey_start,
                                    storey_end, address)
        st.write("### :green[**The Predicted Flat Resale Price is : S$**]", pre_price)
        st.balloons()

st.markdown("---")
st.markdown("Created by: Kamayani 👩‍💻")





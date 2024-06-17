#import libraries
import numpy as np
import streamlit as st
from datetime import date
import pickle
import pandas as pd
import xgboost as xgb

#Load the model
model = xgb.XGBRegressor()
model.load_model('final_model.model')

st.write("""
# Predicting Used Car Prices
This app predicts the ** recommended car listing price ** using features input via the **side panel** 
""")
# Load the dataframe skeleton for prediction
df_skeleton = pd.read_csv('df_skeleton.csv', index_col = 0)
# Load the brand_list
brand_list = pickle.load(open('brand_list.pkl', 'rb'))

def addYears(d, years):
    try:
    # Return same day of the current year
        return d.replace(year=d.year + years)
    except ValueError:
    # If not same day, it will return other, i.e.  February 29 to March 1 etc.
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox
    return type : pandas dataframe

    """
    make = st.sidebar.selectbox("Select Make", options = brand_list)
    transmission_type = st.sidebar.selectbox("Transmission Type", options = ['Auto','Manual'])
    vehical_type = st.sidebar.selectbox("Type of Vehicle", options = ['Hatchback', 'Sports Car', 'Mid-Sized Sedan', 'SUV', 'Luxury Sedan', 'MPV', 'Stationwagon'])
    no_of_owners = st.sidebar.selectbox('Number of Owners', options = ['1', '2', '3', '4', '5', '6', 'More than 6'])
    mileage = st.sidebar.number_input('Mileage(km)', min_value= 10)
    reg_date = st.sidebar.date_input('Car Registration Date', max_value= date.today())
    coe_qp = st.sidebar.number_input('COE QP ($)', min_value= 10000)
    arf = st.sidebar.number_input('ARF ($)', min_value = 100)
    depreciation = st.sidebar.number_input('Depreciation ($)', min_value = 100)
    road_tax = st.sidebar.number_input('Road Tax ($ per annum)', min_value = 100)
    dereg_value = st.sidebar.number_input('Deregistration Value ($)', min_value = 100)
    power = st.sidebar.number_input('Power (Kw)', min_value = 10)
    
    coe_days_left = float((addYears(reg_date, 10) - date.today()).days -1)
    
    df_skeleton.loc[0, 'MILEAGE'] = mileage
    df_skeleton.loc[0, 'COE'] = coe_qp
    df_skeleton.loc[0, 'CURB_WEIGHT'] = arf
    df_skeleton.loc[0, 'COE_NUMBER_OF_DAYS_LEFT'] = coe_days_left
    df_skeleton.loc[0, 'AGE_OF_COE'] = float((date.today() - reg_date).days -1)
    df_skeleton.loc[0, 'log_DEPRECIATION'] = np.log1p(depreciation)
    df_skeleton.loc[0, 'log_ROAD_TAX'] = np.log1p(road_tax)
    df_skeleton.loc[0, 'log_DEREG_VALUE'] = np.log1p(dereg_value)
    df_skeleton.loc[0, 'log_ARF'] = np.log1p(arf)
    df_skeleton.loc[0, 'log_POWER'] = np.log1p(power)
    if transmission_type == 'Auto':
        df_skeleton.loc[0, 'TRANSMISSION_Auto'] = 1
        df_skeleton.loc[0, 'TRANSMISSION_Manual'] = 0
    else:
        df_skeleton.loc[0, 'TRANSMISSION_Auto'] = 0
        df_skeleton.loc[0, 'TRANSMISSION_Manual'] = 1
    if no_of_owners == 1:
        df_skeleton.loc[0, 'NO_OF_OWNERS_1'] = 1
        df_skeleton.loc[0, 'NO_OF_OWNERS_2'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_3'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_4'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_5'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_6'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_More than 6'] = 0
    elif no_of_owners == 2:
        df_skeleton.loc[0, 'NO_OF_OWNERS_1'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_2'] = 1
        df_skeleton.loc[0, 'NO_OF_OWNERS_3'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_4'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_5'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_6'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_More than 6'] = 0
    elif no_of_owners == 3:
        df_skeleton.loc[0, 'NO_OF_OWNERS_1'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_2'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_3'] = 1
        df_skeleton.loc[0, 'NO_OF_OWNERS_4'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_5'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_6'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_More than 6'] = 0
    elif no_of_owners == 4:
        df_skeleton.loc[0, 'NO_OF_OWNERS_1'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_2'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_3'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_4'] = 1
        df_skeleton.loc[0, 'NO_OF_OWNERS_5'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_6'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_More than 6'] = 0
    elif no_of_owners == 5:
        df_skeleton.loc[0, 'NO_OF_OWNERS_1'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_2'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_3'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_4'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_5'] = 1
        df_skeleton.loc[0, 'NO_OF_OWNERS_6'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_More than 6'] = 0
    elif no_of_owners == 6:
        df_skeleton.loc[0, 'NO_OF_OWNERS_1'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_2'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_3'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_4'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_5'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_6'] = 1
        df_skeleton.loc[0, 'NO_OF_OWNERS_More than 6'] = 0
    else:
        df_skeleton.loc[0, 'NO_OF_OWNERS_1'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_2'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_3'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_4'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_5'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_6'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_More than 6'] = 1

    if vehical_type == 'Hatchback':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Hatchback'] = 1
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
    elif vehical_type == 'Luxury Sedan':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Hatchback'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 1
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
        df_skeleton.loc[0, 'NO_OF_OWNERS_More than 6'] = 0
    elif vehical_type == 'MPV':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Hatchback'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 1
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
    elif vehical_type == 'Mid-Sized Sedan':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Hatchback'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 1
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
    elif vehical_type == 'SUV':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Hatchback'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 1
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
    elif vehical_type == 'Sports Car':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Hatchback'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 1
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
    else:
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Hatchback'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 1
        
    brand_col = {}
    for brand in brand_list:
        brand_col[brand] = 'BRAND_'+brand
    
    # Set indicator variables based on 'brand' column
    for brand in brand_list:
        if make == brand:
            temp = brand_col[brand]
            df_skeleton.loc[0, temp] = 1
        else:
            temp = brand_col[brand]
            df_skeleton.loc[0, temp] = 0

    return df_skeleton

df_skeleton = get_user_input()

# when 'Predict' is clicked, make the prediction and store it
if st.sidebar.button("Predict"):
    # Predict the result
    result = model.predict(df_skeleton.values)[0]
    
    # Format result to 2 decimal places
    formatted_result = '{:.2f}'.format(result)
    
    # Display success message with formatted result
    st.success(f'Recommended pricing of vehicle is: ${formatted_result}')




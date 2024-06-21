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
This app predicts the **recommended car listing price** and its **yearly depreciation** using features input via the **side panel** 
""")
# Load the dataframe skeleton for prediction
df_skeleton = pd.read_csv('df_skeleton.csv', index_col = 0)
# Load the brand_list
brand_list = pickle.load(open('brand_list.pkl', 'rb'))
# Load the columns to scale
columns_to_scale = pickle.load(open('columns_to_scale.pkl', 'rb'))
# load scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))
# load brand mean price dictionary
brand_mean_price_dict = pickle.load(open('brand_mean_price_dict.pkl', 'rb'))

def reduceYears(d, years):
    try:
    # Return same day of the current year
        return d.replace(year=d.year - years)
    except ValueError:
    # If not same day, it will return other, i.e.  February 29 to March 1 etc.
        return d + (date(d.year - years, 1, 1) - date(d.year, 1, 1))


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox
    return type : pandas dataframe

    """
    make = st.sidebar.selectbox("Car Brand", options = brand_list)
    vehical_type = st.sidebar.selectbox("Type of Vehicle", options = ['Hatchback', 'Sports Car', 'Mid-Sized Sedan', 'SUV', 'Luxury Sedan', 'MPV', 'Stationwagon'])
    no_of_owners = st.sidebar.selectbox('Number of Owners', options = ['1', '2', '3', '4', '5', '6', 'More than 6'])
    mileage = st.sidebar.number_input('Mileage (km)', min_value= 10)
    coe_type = st.sidebar.selectbox('COE Type', options = ['5 years', '10 years'])
    reg_date = st.sidebar.date_input('COE Expiry Date', min_value= date.today())
    coe_qp = st.sidebar.number_input('COE ($)', min_value= 10000)
    arf = st.sidebar.number_input('ARF ($)', min_value = 100)
    road_tax = st.sidebar.number_input('Road Tax ($ per annum)', min_value = 100)
    power = st.sidebar.number_input('Power (Kw)', min_value = 10)
    curb_weight = st.sidebar.number_input('Curb Weight (KG)', min_value = 100)
    
    df_skeleton.loc[0, 'MILEAGE'] = mileage
    df_skeleton.loc[0, 'COE'] = coe_qp
    df_skeleton.loc[0, 'CURB_WEIGHT'] = curb_weight
    df_skeleton.loc[0, 'COE_NUMBER_OF_DAYS_LEFT'] = int((reg_date - date.today()).days)
    if coe_type == '5 years':
        age_of_coe = float((date.today() - reduceYears(reg_date, 5)).days)
        df_skeleton.loc[0, 'AGE_OF_COE'] = age_of_coe
    else:
        age_of_coe = float((date.today() - reduceYears(reg_date, 10)).days)
        df_skeleton.loc[0, 'AGE_OF_COE'] = age_of_coe
    df_skeleton.loc[0, 'log_ROAD_TAX'] = np.log1p(road_tax)
    df_skeleton.loc[0, 'log_ARF'] = np.log1p(arf)
    df_skeleton.loc[0, 'log_POWER'] = np.log1p(power)
    
    if no_of_owners == 'More than 6':
        df_skeleton.loc[0, 'NO_OF_OWNERS'] = 7
    else:
        df_skeleton.loc[0, 'NO_OF_OWNERS'] = int(no_of_owners)
        
    if vehical_type == 'Hatchback':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
    elif vehical_type == 'Luxury Sedan':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 1
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
    elif vehical_type == 'MPV':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 1
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
    elif vehical_type == 'Mid-Sized Sedan':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 1
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
    elif vehical_type == 'SUV':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 1
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
    elif vehical_type == 'Sports Car':
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 1
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 0
    else:
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Luxury Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_MPV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Mid-Sized Sedan'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_SUV'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Sports Car'] = 0
        df_skeleton.loc[0, 'TYPE_OF_VEHICLE_Stationwagon'] = 1

    df_skeleton.loc[0, 'log_BRAND_MEAN_PRICE'] = np.log1p(brand_mean_price_dict[make])

    return df_skeleton, arf, age_of_coe

df_skeleton, arf, coe_days_left = get_user_input()

# Scale the Data
df_skeleton[columns_to_scale] = scaler.transform(df_skeleton[columns_to_scale])

# when 'Predict' is clicked, make the prediction and store it
if st.sidebar.button("Predict"):
    # Predict the result
    result = model.predict(df_skeleton.values)[0]
    
    # Format result to 2 decimal places
    formatted_result = '${:,.2f}'.format(result)
    
    # Display success message with formatted result
    st.success(f'Recommended pricing of vehicle is: ${formatted_result}')

    parf = 0.5 * arf
    depreciation = int((result - parf) / (age_of_coe / 365))
    st.write('Estimated depreciation is : ${:,.2f} /year'.format(depreciation))




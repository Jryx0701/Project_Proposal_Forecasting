import streamlit as st
import plotly.express as px
import pickle
import numpy as np
import pandas as pd
import time
from darts import TimeSeries
from darts.models import Theta, ExponentialSmoothing
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt
from store_level_ai_integration import train_and_forecast_product_store_level,get_top_products_store
from global_real_time import train_and_forecast_product,get_top_products_store_global

drive_csv = "https://drive.google.com/uc?id=1Dxgx5fWOG05OP4vtODifo3sd-p6M4dEK"
df_train = pd.read_csv(drive_csv,encoding = "latin1")
df_train['TrnDate'] = pd.to_datetime(df_train['TrnDate'], format='%d/%m/%Y %H:%M', errors='coerce')
toTrain = df_train[['TrnDate', 'PName','IssueQuantity', 'RetailPrice', 'PurchasePrice', 'OnHandQuantity']].copy()
#--------------------------------------------------------------------------------------
#For Recommendation System for onhandquantities and forecasting demands
def recommendation_system(forecasting_values,OnHandQuantity):
    Forecasted_IssueQuantity = sum(forecasting_values.values) #30 days forecasting
    coverage_ratio = OnHandQuantity/Forecasted_IssueQuantity
    recommendation = ""
    
    if Forecasted_IssueQuantity == 0 and OnHandQuantity > 0:
        recommendation = "❌ Dead Stock – Consider Liquidating"
    elif OnHandQuantity == 0 and Forecasted_IssueQuantity > 0:
        recommendation = "🔺 Stockout Risk – Urgent Reorder Needed"
    elif coverage_ratio < 0.5:
        recommendation = "⚠️ Understock – Reorder Soon" 
    elif coverage_ratio > 1.5:
        recommendation = "📦 Overstock – Reduce Purchase"
    elif coverage_ratio >= 0.8 and coverage_ratio <= 1.2:
        recommendation = "✅ Balanced – Maintain Level"
    else:
        recommendation = "🔁 Monitor – Demand Fluctuating"

    return recommendation

#--------------------------------------------------------------------------------------
#Cleaning the data
toTrain.isna().sum()
toTrain.replace('', np.nan, inplace=False).isna().sum()
#drop rows with missing dates
toTrain = toTrain.dropna(subset=['TrnDate'])
toTrain = toTrain.dropna(subset=['IssueQuantity'])
toTrain = toTrain.dropna(subset=['PName'])
toTrain = toTrain.dropna(subset=['RetailPrice'])
toTrain = toTrain.dropna(subset=['PurchasePrice'])
toTrain = toTrain.dropna(subset=['OnHandQuantity'])
product_sales = toTrain.groupby('PName')['IssueQuantity'].sum().reset_index()
#--------------------------------------------------------------------------------------
Market_list = list(df_train['Warehouse'].unique())
Product_list = list(df_train['PName'].unique())
Market_list.insert(0,"Global Products")
#--------------------------------------------------------------------------------------
st.set_page_config(layout="wide")
with st.sidebar:
    market_list = st.selectbox(
        "Market List",
        Market_list,
    )
#--------------------------------------------------------------------------------------
if market_list == "Global Products":
    #get_top 10 products
    global_top_products = get_top_products_store_global(df_train)

    with st.sidebar:
        product_list = st.selectbox(
        "Product List",
        Product_list
    )
        st.subheader(f"Top 10 {market_list}")
        st.write(global_top_products)
    #Prediction per product
    result_per_product = train_and_forecast_product(df_train,product_list)
    #Plotting of the Forecasting of the products
    #Need changes in handling Errors
    if "error" in result_per_product:
        st.warning(result_per_product["error"],icon="⚠️",width = "stretch")
    else:
        current_stock = df_train[df_train['PName'] == product_list]['OnHandQuantity'].iloc[0]
        st.subheader(f"Forecast for {result_per_product['product']}")
        st.line_chart(result_per_product['forecast_df'].set_index("Date"))
        st.divider()


        #Bar chart comparison of IssueDemand Forecast and CurrentStocks of the Items
        total_issue_demand = result_per_product['forecast_df']['Forecasted_IssueQuantity'].sum()
        st.subheader('Forecasted IssueDemand vs. OnHandQuantity')
        # Create a DataFrame
        df = pd.DataFrame({
            'Quantity': [total_issue_demand, current_stock]
        }, index=['IssueDemand', 'OnHandQuantity'])

        # Streamlit built-in bar chart
        st.bar_chart(df)
        st.divider()
        #Plotting the predicted table of the specific product (Table)
        st.subheader(f"Predicted Table in {product_list}")
        st.dataframe(result_per_product['forecast_df'],use_container_width = True)

        st.divider()
        global_products_total =  result_per_product['forecast_df']['Forecasted_IssueQuantity']
        total = sum(global_products_total)
        mean = np.mean(global_products_total)
        std_dev = np.std(global_products_total)
        trend = global_products_total.iloc[-1] - global_products_total.iloc[0]
        recommendation = recommendation_system(global_products_total,current_stock)
        #Recommendation System for forecasting in Global Product
        st.subheader(f"🧾 Product Summary:")
        st.write(f"- Total Forecasting Demand: {total:.2f}\n"
                    f"- Average Demand: {mean:.2f}\n"
                    f"- Trend: {trend:.2f} units/days\n"
                    f"- Std. Deviation: {std_dev:.2f}\n"
                    f"- Current Stock: {current_stock}\n"
                    f"- Reliability: {100 - result_per_product['rmse_percent']:.2f}%\n"
                    f"- 📢 Recommendation: {recommendation}")

elif market_list != "Global Product":
    #Filter out depending on the market list that was chosen
    chosen_store = df_train[df_train['Warehouse'] == market_list].copy()
    product_list_store = chosen_store['PName'].unique().copy()
    top_products = get_top_products_store(df_train,market_list)
    with st.sidebar:
        list_product_store = st.selectbox(
            "Product List",
            product_list_store
        )
        st.subheader(f"Top 10 products in {market_list}")
        st.write(top_products)
    product_prediction = train_and_forecast_product_store_level(df_train,market_list,list_product_store)
    

    if "error" in product_prediction:
        st.warning(product_prediction["error"],icon="⚠️",width = "stretch")
    else:
        current_stock_store = df_train[df_train['PName'] == list_product_store]['OnHandQuantity'].iloc[0]
        st.subheader(f"Forecast for {list_product_store} in {market_list}")
        st.line_chart(product_prediction['forecast_df'].set_index("Date"))
        st.divider()

        #Bar chart comparison of IssueDemand Forecast and CurrentStocks of the Items
        total_issue_demand = product_prediction['forecast_df']['Forecasted_IssueQuantity'].sum()
        st.subheader('Forecasted IssueDemand vs. OnHandQuantity')
        # Create a DataFrame
        df = pd.DataFrame({
            'Quantity': [total_issue_demand, current_stock_store]
        }, index=['IssueDemand', 'OnHandQuantity'])

        # Streamlit built-in bar chart
        st.bar_chart(df)
        st.divider()

        st.subheader(f"Predicted Table for {list_product_store} in {market_list}")
        st.dataframe(product_prediction['forecast_df'], use_container_width=True)

        #Calculating the values for Recommendation System
        forecasted_values = product_prediction['forecast_df']['Forecasted_IssueQuantity']
        total = sum(forecasted_values)
        mean = np.mean(forecasted_values)
        std_dev = np.std(forecasted_values)
        trend = forecasted_values.iloc[-1] - forecasted_values.iloc[0]
        Recommended = recommendation_system(forecasted_values,current_stock_store)
        st.divider()
        st.subheader(f"🧾 Product Summary:")
        st.write(f"- Total Forecasting Demand: {total:.2f}\n"
                    f"- Average Demand: {mean:.2f}\n"
                    f"- Trend: {trend:.2f} units/days\n"
                    f"- Std. Deviation: {std_dev:.2f}\n"
                    f"- Current Stock: {current_stock_store}\n"
                    f"- Reliability: {100 - product_prediction['rmse_percent']:.2f}%\n"
                    f"- 📢 Recommendation: {Recommended}")
#--------------------------------------------------------------------------------------
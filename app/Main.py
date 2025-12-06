import streamlit as st
import pandas as pd 
import plotly.express as px
import time
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import joblib
from scipy.stats import gaussian_kde
import sys

import joblib
import pandas as pd
import numpy as np

import sys




st.set_page_config(page_title="Descriptive Analytics ", page_icon="🌎", layout="wide")  

# Custom heading with beige background and dark text
def heading():
    st.markdown("""  
        <style>
        .custom-heading {
            background-color: #F5F5DC;  /* اللون البيج */
            color: #333333;             /* خط غامق عشان يبان */
            padding: 20px;
            border-radius: 12px;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            margin-bottom: 25px;
        }
        </style>

        <div class="custom-heading">
            📈 Descriptive Analytics 📊
        </div>
    """, unsafe_allow_html=True)





#remove default theme
theme_plotly = None # None or streamlit

 
st.markdown("""
    <style>
    /* Change st.info background to beige */
    div.stAlert {
        background-color: #F5DEB3 !important;   /* Beige */
        border-left: 6px solid #C5A880 !important; /* Dark beige border */
    }.plot-container > div {
    box-shadow: 0 0 4px #cccccc;
    padding: 10px;
    }

    /* Change text color inside st.info */
    div.stAlert p {
        color: #4a3f35 !important;  /* Dark brown text */
        font-weight: bold;
    }
    
    </style>
""", unsafe_allow_html=True)

# Reading parts
# Reading parts
@st.cache_data
def load_data():
    df = pd.read_parquet("https://raw.githubusercontent.com/MohamedHeshamrg/Zomato/main/data/preprocessed/data.parquet")
    return df
df = load_data()
df['success_cat'] = df["success_score"].map({1: "Success", 0: "Fail"})








def HomePage():
 heading()
  #1. print dataframe
 with st.expander("🧭 My database"):
  #st.dataframe(df_selection,use_container_width=True)
  st.dataframe(df,use_container_width=True)

 #2. compute top Analytics
 
 Total_restorant = 7148
 Most_have_branches = 89
 The_city_withthemostrestaurants = 2799
 The_most_type= 22021	 

 #3. columns
 total1,total2,total3,total4 = st.columns(4,gap='large')
 with total1:

    st.info('Total Restorant', icon="🔎")
    st.metric(label = 'Count', value= f"{Total_restorant}")
    
 with total2:
    st.info('Cafe Coffee Day Have', icon="💵")
    st.metric(label='Sum', value=f"{Most_have_branches} Branch")

 with total3:
    st.info('The city with the most restaurants', icon="🍔")
    st.metric(label= 'BTM',value=f"{The_city_withthemostrestaurants}")

 with total4:
    st.info('Most Type of Restorant', icon="📦")
    st.metric(label='Delivery',value=f"{The_most_type}")


    
 st.markdown("""---""")

 #graphs
 
def Graphs():
 

   with st.container():
      col1, col2 = st.columns([4,3])
      fig = px.histogram(
         df,
         x="approx_cost(for two people)",
         nbins=30,
         histnorm='density',
         template="plotly_white",
      )

      fig.update_layout(
         title=f'Distribution Approx_cost(for two people)"',
         xaxis_title="Approx Cost",
         yaxis_title='Density'
      )
      fig.update_xaxes(tickangle=45)
      fig.update_traces(marker=dict(color=px.colors.sequential.speed[0]))

      col1.plotly_chart(fig, use_container_width=True)

      counts = df["success_cat"].value_counts()

      fig = px.pie(
      names=counts.index,
      values=counts.values,
      title=f"Percentage of Success ",hole=.2
      )
      fig.update_traces(
      marker=dict(colors=px.colors.sequential.speed)
      )
      fig.update_traces(textposition='inside', textinfo='percent+label')
            
      

      col2.plotly_chart(fig, use_container_width=True)



   with st.container():
      rest_columns = df.filter(regex="^Rest_").columns

      long_data = []

      for col in rest_columns:
         avg_rating = (df[col] * df['rate']).sum() / df[col].sum() if df[col].sum() != 0 else 0
         long_data.append({'Restaurant Type': col, 'Average_Rating': avg_rating})

      rest_df_long = pd.DataFrame(long_data)

      rest_df_long = rest_df_long.sort_values("Average_Rating", ascending=False)

      fig = px.bar(
         rest_df_long,
         x='Restaurant Type',
         y='Average_Rating',
         title="Restaurant Types Rating Distribution",
         labels={'Restaurant Type': 'Restaurant Type', 'Average Rating':'Average Rating'},
         color='Average_Rating',
         color_continuous_scale=px.colors.sequential.speed_r)


      fig.update_layout(xaxis_tickangle=-45)


      col1.plotly_chart(fig, use_container_width=True)

      fig = px.violin(
         df,
         x="votes",  
         template="plotly_white",
         title=f' Violin Plot of Votes'
         )

      fig.update_layout(
         xaxis_title=col,
         )
      fig.update_traces(marker=dict(color=px.colors.sequential.speed[0]))

      fig.update_xaxes(tickangle=45)  

      col2.plotly_chart(fig, use_container_width=True)




HomePage()
Graphs()
     
  

footer = """
<style>
a {
    color: #333333;
    text-decoration: none;
}
a:hover {
    color: red;
    text-decoration: underline;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #F5DEB3;  /* Beige */
    color: #333333;             /* Dark text */
    text-align: center;
    padding: 8px 0;
    font-weight: bold;
    font-size: 16px;
}
</style>

<div class="footer">
    Developed by Mohamed Hesham Ragab
</div>
"""

st.markdown(footer, unsafe_allow_html=True)














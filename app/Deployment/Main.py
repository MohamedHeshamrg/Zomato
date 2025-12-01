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

# Custom heading with dark blue background and white text
def heading():
    st.markdown("""  
        <style>
        .custom-heading {
            background-color: #5ab8db;  /* لبني غامق */
            color: white;              /* خط أبيض */
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

 
# تطبيق التنسيق من CSS خارجي
with open(r'H:\ML Intern\zomato\app\Deployment\Style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("##")
# Reading parts
# Reading parts
@st.cache_data
def load_data():
    df = pd.read_parquet(r"H:\ML Intern\zomato\data\preprocessed\data.parquet")
    return df
df = load_data()
df['success_cat'] = df["success"].map({1: "Success", 0: "Fail"})
df['menu_item'] = df["menu_item"].map({1: "Have menu", 0: "Doesn't have menu"})
df['phone'] =  df['phone'].map({1: "Have one phone", 2: "Have two phone"})
df['online_order'] = df['online_order'].map({1: "Yes", 0: "No"})
df['book_table'] = df['book_table'].map({1: "Yes", 0: "No"})







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
     
  

footer="""<style>
 

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
height:5%;
bottom: 0;
width: 100%;
background-color: #243946;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by Eng. Mohamed Hesham Ragab<a style='display: block; text-align: center;target="_blank"></a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

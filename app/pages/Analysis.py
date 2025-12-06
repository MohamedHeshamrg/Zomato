
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import gaussian_kde
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

import sys




st.set_page_config(page_title="Explatory Data Analysis", page_icon="📈", layout="wide")

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

 
st.markdown("""
    <style>
    [data-testid=metric-container] {
        box-shadow: 0 0 4px #cccccc;
        padding: 10px;
    }

    .plot-container > div {
        box-shadow: 0 0 4px #cccccc;
        padding: 10px;
    }

    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.3rem;
        color: rgb(71, 146, 161);
    }
    </style>
""", unsafe_allow_html=True)

tab1, tab2  = st.tabs(['📊🟦 Categorical Analysis','📈🟧 Numerical Analysi'])



# Reading parts
@st.cache_data
def load_data():
    df = pd.read_parquet(r"https://raw.githubusercontent.com/MohamedHeshamrg/Zomato/main/data/preprocessed/data.parquet")
    return df
df = load_data()
df['success_cat'] = df["success_score"].map({1: "Success", 0: "Fail"})



# ==============================
# 📊 Univariate Analysis
# ==============================
st.sidebar.header("📊 Analysis")



# ------------------------------
# 🟦 Categorical
# ------------------------------
with tab1:
    st.markdown('<h3 style="text-align: center; color : #F5DEB3;">Charts of Categorical features</h3>', unsafe_allow_html=True)
    sts = st.selectbox('select How featureImpact on :',
                       ['Rate', 'Approxx cost',"count"], key=21)
    if sts == 'count':
        with st.container():
            st.markdown('<h3 style="text-align: center; color : #5ab8db;">Charts of restorant identity features</h3>', unsafe_allow_html=True)

            col= st.selectbox('select restorant identity feature to see its distribution : ',
                      ['online_order','book_table','phone','menu_item',"success_cat","cost_category"], key=20)
            col1, col2 = st.columns([4,3])
    
            

            counts = df[col].value_counts()
            
            # Create bar plot
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                title=f"Distribution of {col}",
                text=counts.values,
                color=counts.index,
                
            )

            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Count",
                showlegend=False
            )
            fig.update_traces(marker=dict(color=px.colors.sequential.speed))
            fig.update_traces(
                textposition="outside"
            )

                

        
            col1.plotly_chart(fig, use_container_width= True)

        
        
        
               
            counts = df[col].value_counts()

            fig = px.pie(
            names=counts.index,
            values=counts.values,
            title=f"Percentage of {col}",hole=.2
            )
            fig.update_traces(
            marker=dict(colors=px.colors.sequential.speed)
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
                
            col2.plotly_chart(fig, use_container_width=True)
            
            
            
        with st.container():
            st.markdown('<h3 style="text-align: center; color : #F5DEB3;">Charts of restorant Attributes features</h3>', unsafe_allow_html=True)
            
            col= st.selectbox('select restorant Attributes feature 👀 to see its distribution : ',
                      ['Restaurant Type','Cuisine Type', 'listed_in(type)'], key=23)
            col3,col4= st.columns([4,1])
    
            if col == 'Restaurant Type':
                rest_df = df.filter(regex="^Rest_").sum().reset_index()
                rest_df.columns = ["Restaurant Type", "Count"]
                rest_df = rest_df.sort_values("Count", ascending=False)
                fig = px.bar(
                    rest_df,
                    x="Restaurant Type",
                    y="Count",
                    title="Distribution of Restaurant Types",
                )

                fig.update_traces(marker=dict(color=px.colors.sequential.speed))
            elif col == 'Cuisine Type':
                rest_df = df.filter(regex="^Total_").sum().reset_index()
                rest_df.columns = ["cuisine Type", "Count"]
                rest_df = rest_df.sort_values("Count", ascending=False)
                fig = px.bar(
                    rest_df,
                    x="cuisine Type",
                    y="Count",
                    title="Distribution of Cuisine Types",
                )
                fig.update_traces(marker=dict(color=px.colors.sequential.speed))
            else :
                counts = df['listed_in(type)'].value_counts()
                fig = px.bar(
                    x = counts.index,
                    y = counts.values,
                    labels={"x": "Restaurant Service Type", "y": "Count"},
                    title="Distribution of Restaurant Service Types"
                )
                fig.update_traces(marker=dict(color=px.colors.sequential.speed))


            col3.plotly_chart(fig, use_container_width= True)

                            
            
            
            
            

        
        


        with st.container():
            st.markdown('<h3 style="text-align: center; color : #F5DEB3;">Charts of Location features</h3>', unsafe_allow_html=True)
            
            col= st.selectbox('select Location feature 🚩 to see its distribution : ',
                      ['location','listed_in(city)'], key=25)
            col5, col6 = st.columns([4,.5])
    
                  # Create bar plot
            counts = df[col].value_counts()     
            fig = px.bar(
                y=counts.index,
                x=counts.values,
                title=f"Distribution of {col}",
                text=counts.values,
                color=counts.index,height=600
                
            )

            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Count",
                showlegend=False
            )
            fig.update_traces(marker=dict(color=px.colors.sequential.speed))


            col5.plotly_chart(fig, use_container_width= True)

        


            


        






    elif sts == 'Rate':
        with st.container():
            st.markdown('<h3 style="text-align: center; color : #F5DEB3;">Charts of restorant identity features</h3>', unsafe_allow_html=True)

            col= st.selectbox('select restorant identity feature to see its distribution : ',
                      ['online_order','book_table','phone','menu_item',"success_cat","cost_category"], key=20)
            col1, col2 = st.columns([4,.5])
            

                
            # Plot
            fig = px.box(
                df,
                x=col,
                y="rate",
                title=f"{col} vs Rating Distribution",
                color=col
            )

            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Rating",
                showlegend=False
            )

            fig.update_traces(marker=dict(color=px.colors.sequential.speed[0]))

                

        
            col1.plotly_chart(fig, use_container_width= True, key=f"{col}_box")

        
        


        with st.container():
            st.markdown('<h3 style="text-align: center; color : #F5DEB3;">Charts of restorant Attributes features</h3>', unsafe_allow_html=True)
            
            col= st.selectbox('select restorant Attributes feature 👀 to see its distribution : ',
                      ['Restaurant Type','Cuisine Type', 'listed_in(type)'], key=23)
            col3,col4= st.columns([4,.5])
    
            if col == 'Cuisine Type':
                rest_columns = df.filter(regex="^Total_").columns

                long_data = []

                for col in rest_columns:
                    avg_rating = (df[col] * df['rate']).sum() / df[col].sum() if df[col].sum() != 0 else 0
                    long_data.append({'Cuisine Types': col, 'Average_Rating': avg_rating})

                rest_df_long = pd.DataFrame(long_data)

                rest_df_long = rest_df_long.sort_values("Average_Rating", ascending=False)

                fig = px.bar(
                    rest_df_long,
                    x='Cuisine Types',
                    y='Average_Rating',
                    title="Cuisine Types Rating Distribution",
                    labels={'Cuisine Types': 'Cuisine Types', 'Average Rating':'Average Rating'},
                    color='Average_Rating',
                    color_continuous_scale=px.colors.sequential.speed_r)


                fig.update_layout(xaxis_tickangle=-45)
            elif col == 'Restaurant Type':
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
            else :
                type_rate = df.groupby('listed_in(type)')['rate'].mean().reset_index()
                type_rate = type_rate.sort_values('rate', ascending=False).head(10)  # Top 10

                # رسم Bar Chart
                fig = px.bar(
                    type_rate,
                    x='listed_in(type)',
                    y='rate',
                    title="Restaurant Service Types Rating Distribution",
                    labels={'listed_in(type)': 'Restaurant Service Type', 'rate':'Average Rating'},
                    color='rate',
                    color_continuous_scale=px.colors.sequential.speed_r)


                fig.update_layout(xaxis_tickangle=-45)



            col3.plotly_chart(fig, use_container_width= True)



        with st.container():
            st.markdown('<h3 style="text-align: center; color : #F5DEB3;">Charts of Location features</h3>', unsafe_allow_html=True)
            
            col= st.selectbox('select Location feature 🚩 to see its distribution : ',
                      ['location','listed_in(city)'], key=25)
            col5,col6= st.columns([4,.5])

            if col =="listed_in(city)":
                city_rate = df.groupby('listed_in(city)')['rate'].mean().reset_index()
                city_rate = city_rate.sort_values('rate', ascending=False).head(10)  # Top 10

                # رسم Bar Chart
                fig = px.bar(
                    city_rate,
                    x='listed_in(city)',
                    y='rate',
                    title="Top 10 City Rating Distribution",
                    labels={'listed_in(city)': 'City', 'rate':'Average Rating'},
                    color='rate',
                    color_continuous_scale=px.colors.sequential.speed_r)




            else:
                location_rate = df.groupby('location')['rate'].mean().reset_index()
                location_rate = location_rate.sort_values('rate', ascending=False).head(10)  # Top 10

                # رسم Bar Chart
                fig = px.bar(
                    location_rate,
                    x='location',
                    y='rate',
                    title="Top 10 Location Rating Distribution",
                    labels={'location': 'Location', 'rate':'Average Rating'},
                    color='rate',
                    color_continuous_scale=px.colors.sequential.speed_r)


                fig.update_layout(xaxis_tickangle=-45)


            col5.plotly_chart(fig, use_container_width=True)




    elif sts == 'Approxx cost':
    
        with st.container():
            st.markdown('<h3 style="text-align: center; color : #F5DEB3;">Charts of restorant identity features</h3>', unsafe_allow_html=True)

            col= st.selectbox('select restorant identity feature to see its distribution : ',
                      ['online_order','book_table','phone','menu_item',"success_cat","cost_category"], key=50)
            col1, col2 = st.columns([4,.5])
            
                

                
            # Plot
            fig = px.box(
                df,
                x=col,
                y="approx_cost(for two people)",
                title=f"{col} vs Approxx cost Distribution",
                color=col
            )

            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Approxx cost",
                showlegend=False
            )

            fig.update_traces(marker=dict(color=px.colors.sequential.speed[0]))

                

        
            col1.plotly_chart(fig, use_container_width= True)

        
        


        with st.container():
            st.markdown('<h3 style="text-align: center; color : #F5DEB3;">Charts of restorant Attributes features</h3>', unsafe_allow_html=True)
            
            col= st.selectbox('select restorant Attributes feature 👀 to see its distribution : ',
                      ['Restaurant Type','Cuisine Type', 'listed_in(type)'], key=55)
            col3,col4= st.columns([4,.5])
    
            if col == 'Cuisine Type':
                rest_columns = df.filter(regex="^Total_").columns

                long_data = []

                for col in rest_columns:
                    avg_rating = (df[col] * df['approx_cost(for two people)']).sum() / df[col].sum() if df[col].sum() != 0 else 0
                    long_data.append({'Cuisine Types': col, 'Average_Approxx cost': avg_rating})

                rest_df_long = pd.DataFrame(long_data)

                rest_df_long = rest_df_long.sort_values("Average_Approxx cost", ascending=False)

                fig = px.bar(
                    rest_df_long,
                    x='Cuisine Types',
                    y='Average_Approxx cost',
                    title="Cuisine Types Average Approxx cost Distribution",
                    labels={'Cuisine Types': 'Cuisine Types', 'Average_Approxx cost':'Average Approxx cost'},
                    color='Average_Approxx cost',
                    color_continuous_scale=px.colors.sequential.speed_r)


                fig.update_layout(xaxis_tickangle=-45)
            elif col == 'Restaurant Type':
                rest_columns = df.filter(regex="^Rest_").columns

                long_data = []

                for col in rest_columns:
                    avg_rating = (df[col] * df['approx_cost(for two people)']).sum() / df[col].sum() if df[col].sum() != 0 else 0
                    long_data.append({'Restaurant Type': col, 'Average_Approxx cost': avg_rating})
                rest_df_long = pd.DataFrame(long_data)

                rest_df_long = rest_df_long.sort_values("Average_Approxx cost", ascending=False)

                fig = px.bar(
                    rest_df_long,
                    x='Restaurant Type',
                    y='Average_Approxx cost',
                    title="Restaurant Types Average Approxx cost Distribution",
                    labels={'Restaurant Type': 'Restaurant Type', 'Average_Approxx cost':'Average Approxx cost'},
                    color='Average_Approxx cost',
                    color_continuous_scale=px.colors.sequential.speed_r)


                fig.update_layout(xaxis_tickangle=-45)
            else :
                type_rate = df.groupby('listed_in(type)')['approx_cost(for two people)'].mean().reset_index()
                type_rate = type_rate.sort_values('approx_cost(for two people)', ascending=False).head(10)  # Top 10

                # رسم Bar Chart
                fig = px.bar(
                    type_rate,
                    x='listed_in(type)',
                    y='approx_cost(for two people)',
                    title="Restaurant Service Types Average Approxx cost Distribution",
                    labels={'listed_in(type)': 'Restaurant Service Type', 'approx_cost(for two people)':"Average Approxx cost"},
                    color='approx_cost(for two people)',
                    color_continuous_scale=px.colors.sequential.speed_r)


                fig.update_layout(xaxis_tickangle=-45)



            col3.plotly_chart(fig, use_container_width= True)



        with st.container():
            st.markdown('<h3 style="text-align: center; color : #F5DEB3;">Charts of Location features</h3>', unsafe_allow_html=True)
            
            col= st.selectbox('select Location feature 🚩 to see its distribution : ',
                      ['location','listed_in(city)'], key=52)
            col5,col6= st.columns([4,.5])
            if col =="listed_in(city)":
                city_rate = df.groupby('listed_in(city)')['approx_cost(for two people)'].mean().reset_index()
                city_rate = city_rate.sort_values('approx_cost(for two people)', ascending=False).head(10)  # Top 10

                # رسم Bar Chart
                fig = px.bar(
                    city_rate,
                    x='listed_in(city)',
                    y='approx_cost(for two people)',
                    title="Top 10 City Average Approxx cost Distribution",
                    labels={'listed_in(city)': 'City', 'approx_cost(for two people)':"Average Approxx cost"},
                    color='approx_cost(for two people)',
                    color_continuous_scale=px.colors.sequential.speed_r)


                fig.update_layout(xaxis_tickangle=-45)


            else:
                location_rate = df.groupby('location')['approx_cost(for two people)'].mean().reset_index()
                location_rate = location_rate.sort_values('approx_cost(for two people)', ascending=False).head(10)  # Top 10

                # رسم Bar Chart
                fig = px.bar(
                    location_rate,
                    x='location',
                    y='approx_cost(for two people)',
                    title="Top 10 Location Average Approxx cost Distribution",
                    labels={'location': 'Location', 'approx_cost(for two people)':"Average Approxx cost"},
                    color='approx_cost(for two people)',
                    color_continuous_scale=px.colors.sequential.speed_r)


                fig.update_layout(xaxis_tickangle=-45)


            col5.plotly_chart(fig, use_container_width=True)











































                
    st.write("📌 **Statistics for Categorical Columns**")
    st.dataframe(df.describe(include="O").T)



# ------------------------------
# 🟧 Numerical
# ------------------------------
with tab2:
    st.markdown(
        '<h3 style="text-align: center; color : #F5DEB3;">📊 Charts of Numerical Features</h3>',
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown(
            '<h4 style="text-align: center; color : #F5DEB3;">📈 Distribution of Numerical Features</h4>',
            unsafe_allow_html=True
        )

        col = st.selectbox(
            'Select a Numerical Feature to See Its Distribution:',
            ["approx_cost(for two people)", "rate", "votes"],
            key=30
        )


        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            histnorm='density',
            template="plotly_white",
        )

        fig.update_layout(
            title=f'Distribution  {col}',
            xaxis_title=col,
            yaxis_title='Density'
        )
        fig.update_xaxes(tickangle=45)
        fig.update_traces(marker=dict(color=px.colors.sequential.speed[0]))

        st.plotly_chart(fig, use_container_width=True)


        fig = px.box(
            df,
            x=col,  
            template="plotly_white",
            title=f'Box Plot of {col}'
            )

        fig.update_layout(
            xaxis_title=col,
            )

        fig.update_xaxes(tickangle=45)  
        fig.update_traces(marker=dict(color=px.colors.sequential.speed[0]))

        st.plotly_chart(fig, use_container_width=True)


        fig = px.violin(
            df,
            x=col,  
            template="plotly_white",
            title=f' Violin Plot of {col}'
            )

        fig.update_layout(
            xaxis_title=col,
            )
        fig.update_traces(marker=dict(color=px.colors.sequential.speed[0]))

        fig.update_xaxes(tickangle=45)  

        st.plotly_chart(fig, use_container_width=True)
        

        
        

    st.write("📌 **Statistics For Numerical Feature**")
    st.dataframe(df.describe().T)







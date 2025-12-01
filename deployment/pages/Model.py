import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys

# Fix path
sys.path.append(r"H:\ML Intern\zomato\app")

# Load model & data
pipeline = joblib.load("https://raw.githubusercontent.com/MohamedHeshamrg/Zomato/main/models/success_model.pkl")
location_List = joblib.load("https://raw.githubusercontent.com/MohamedHeshamrg/Zomato/main/Deployment/location_List.h5")
type_list = joblib.load('https://raw.githubusercontent.com/MohamedHeshamrg/Zomato/main/Deployment/type_list.h5')
cost_list = joblib.load('https://raw.githubusercontent.com/MohamedHeshamrg/Zomato/main/Deployment/cost_list.h5')
inputs = joblib.load("https://raw.githubusercontent.com/MohamedHeshamrg/Zomato/main/Deployment/input.h5")


def predict(online_order, book_table, phone, location, approx_cost, menu_item, type_rest, cost_category):

    test_df = pd.DataFrame([{
        'online_order': online_order,
        'book_table': book_table,
        'phone': phone,
        'location': location,
        'approx_cost(for two people)': approx_cost,
        'menu_item': menu_item,
        'listed_in(type)': type_rest,
        'cost_category': cost_category
    }])

    predicted = pipeline.predict(test_df)[0]
    predicted = "Success" if predicted == 1 else "Fail"
    return predicted

def main():
    st.title('👨‍🍳🍔 Restaurant Success Prediction')

    approx_cost = st.number_input('approx_cost(for two people)', min_value=40, max_value=6000, value=1500)

    online_order = st.selectbox('Online order available?', ['Yes', 'No'])
    online_order = 1 if online_order.lower() == "yes" else 0

    book_table = st.selectbox('Book table available?', ['Yes', 'No'])
    book_table = 1 if book_table.lower() == "yes" else 0

    phone = st.selectbox('How many phone lines?', ['One', 'Two'])
    phone = 1 if phone.lower() == "one" else 2

    location = st.selectbox('Location', sorted(location_List))

    menu_item = st.selectbox('Is there a menu?', ['Yes', 'No'])
    menu_item = 1 if menu_item.lower() == "yes" else 0

    type_rest = st.selectbox('Restaurant Type', sorted(type_list))
    cost_category = st.selectbox('Cost Category', sorted(cost_list))

    if st.button('🔮 Predict'):
        pred = predict(online_order, book_table, phone, location, approx_cost,
                    menu_item, type_rest, cost_category)

        if pred == "Success":
            st.markdown(f"<h3 style='color: green;'>✅ The restaurant will: {pred}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color: red;'>❌ The restaurant will: {pred}</h3>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()




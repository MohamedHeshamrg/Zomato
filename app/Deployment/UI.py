import streamlit as st


def heading():
    st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
    <div class="container-fluid">
        <div class="card shadow p-3 mb-3 bg-white rounded">
            <h2 style="text-align:center; color:#343a40; font-weight:bold;">
                📊 Analytics Dashboard 📈
            </h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

 
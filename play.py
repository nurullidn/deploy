import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier

# Load dataset from CSV file
df = pd.read_csv("NFLX.csv")

# Convert 'Date' column to datetime data type
df['Date'] = pd.to_datetime(df['Date'])

# Add a new column 'Year' containing the year from the date
df['Year'] = df['Date'].dt.year

# Aggregate data by year and take the mean value for each year
df_yearly = df.groupby('Year').mean().reset_index()

# Define function to load AQI model
def load_aqi_data(data_path):
    try:
        df_aqi = pd.read_csv(data_path)
        return df_aqi
    except Exception as e:
        st.error(f"Error loading the AQI data: {e}")

# Load AQI data from CSV file
aqi_data_path = 'model.csv'
df_aqi = load_aqi_data(aqi_data_path)

# Define function for 'Dashboard' tab
def display_dashboard():
    # Main content area
    st.title("Prediksi Harga Saham Netflix ")
    selected_tab = st.sidebar.selectbox("Pilih Tab:", ["Dashboard", "EDA", "Modelling"])  # Set index=1 untuk menampilkan tab 'Dashboard' secara default

    # Conditional rendering based on selected tab
    if selected_tab == "EDA":
        display_EDA()
    elif selected_tab == "Dashboard":
        st.write("### Dashboard Harga Saham Netflix:")
        st.image("https://i.pcmag.com/imagery/reviews/05cItXL96l4LE9n02WfDR0h-5..v1582751026.png", use_column_width=True)
        
        # Display descriptive statistics
        st.write("### Statistik Deskriptif untuk Harga Saham Netflix:")
        st.write(df.describe())
    elif selected_tab == "Modelling":
        display_Modelling()


# Define function for '4 Pilar Visualisasi' tab
def display_EDA():
    st.write("## 4 Pilar Visualisasi")

    # Display historical line chart of Netflix stock prices
    st.write("### Grafik Historis Harga Saham Netflix")
    fig_line = px.line(df, x='Date', y='Close', title='Grafik Historis Harga Saham Netflix')
    st.plotly_chart(fig_line)

    # Display scatter plot between 'Open' and 'Close' stock prices
    st.write("### Scatter Plot: Harga Open vs. Close")
    fig_scatter = px.scatter(df, x='Open', y='Close', trendline='ols', title='Scatter Plot: Harga Open vs. Close')
    st.plotly_chart(fig_scatter)

    # Display Pie Chart based on yearly stock prices
    st.write("### Pie Chart: Harga Tahunan Netflix")
    fig_pie = px.pie(df_yearly, names='Year', values='Close', title='Pie Chart: Harga Tahunan Netflix')
    st.plotly_chart(fig_pie)

    # Display trading volume distribution
    st.write("### Distribusi Volume Perdagangan")
    fig_hist = px.histogram(df, x='Volume', title='Distribusi Volume Perdagangan')
    st.plotly_chart(fig_hist)


# Define function for 'EDA' tab
def display_Modelling():
    st.write("## Modelling")

    # Display data head
    st.write("### Data Head:")
   

# Call the function to display the dashboard
display_dashboard()



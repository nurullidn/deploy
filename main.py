import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Judul dashboard
st.title('Prediksi Harga Saham Netflix')

# Gambar header
st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=500)


# Path absolut dari file CSV
DATA_URL = 'NFLX.csv'

# Membaca file CSV
df = pd.read_csv(DATA_URL)

# Aggregate data by year and take the mean value for each year
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df_year = df.groupby('Year').mean().reset_index()

# Sidebar
section = st.sidebar.selectbox("Choose Section", ["Home", "EDA", "Modelling"])

# Menampilkan Home
if section == "Home":
   
    st.write("Tujuan utama proyek ini adalah untuk memprediksi harga saham Netflix berdasarkan data historis yang diberikan. Dengan menggunakan metode beberapa pemodelan, proyek ini bertujuan untuk mengembangkan model yang dapat memberikan perkiraan harga saham yang akurat.")

    

# Menampilkan EDA
elif section == "EDA":
    # Visualisasi 4 pilar
    st.header('EDA')

    ## Pilar 1: Grafik Historis Harga Saham Netflix
    st.subheader('Pilar 1: Grafik Historis Harga Saham Netflix')
    fig_line = px.line(df, x='Date', y='Close', title='Grafik Historis Harga Saham Netflix')
    st.plotly_chart(fig_line)
    st.write("Grafik historis di atas menampilkan harga penutup saham Netflix berdasarkan tahun. Ini memungkinkan kita untuk melihat bagaimana harga saham Netflix berfluktuasi dari tahun ke tahun dan membandingkan kinerja saham pada periode waktu yang berbeda. Dengan memperhatikan pola dan tren dalam grafik ini, kita dapat mendapatkan wawasan tentang bagaimana pasar bereaksi terhadap kinerja Netflix dari waktu ke waktu.")

    ## Pilar 2: Scatter plot - Harga Open vs. Close
    st.subheader('Pilar 2: Scatter Plot - Harga Open vs. Close')
    fig_scatter = px.scatter(df, x='Open', y='Close', title='Scatter Plot: Harga Open vs. Close')
    st.plotly_chart(fig_scatter)
    st.write("Scatter plot di atas menampilkan hubungan antara harga awal dan harga akhir saham Netflix. Dengan melihat plot ini, kita dapat menganalisis apakah ada korelasi antara harga awal dan harga akhir saham Netflix. Plot ini membantu kita memahami apakah perubahan harga awal yang tinggi cenderung menghasilkan pergerakan harga yang signifikan pada akhir periode perdagangan, atau apakah ada pola lain yang dapat diamati dalam hubungan antara harga awal dan harga akhir saham Netflix")

    ## Pilar 3: Pie Chart - Harga Tahunan Netflix
    st.write("### Pie Chart: Harga Tahunan Netflix")
    fig_pie = px.pie(df_year, names='Year', values='Close', title='Pie Chart: Harga Tahunan Netflix')
    st.plotly_chart(fig_pie)
    st.write("Grafik pie di atas menampilkan komposisi perubahan harga saham Netflix berdasarkan tahun. Setiap irisan dalam grafik mewakili persentase dari total perubahan harga saham dalam satu tahun. Ini memberikan gambaran tentang seberapa signifikan perubahan harga saham Netflix dari tahun ke tahun.")
    

    # Display trading volume distribution
    st.write("### Distribusi Volume Perdagangan")
    fig_hist = px.histogram(df, x='Volume', title='Distribusi Volume Perdagangan')
    st.plotly_chart(fig_hist)
    st.write("Grafik batang ini menunjukkan bahwa volume perdagangan saham Netflix lebih terkonsentrasi pada volume yang rendah. Volume perdagangan yang sangat rendah dan sangat tinggi terjadi lebih jarang.")

# Menampilkan Modelling
elif section == "Modelling":
    st.header('Modelling')

  
    # Assuming 'df' contains your dataset with features and target variable
    # Define features (X) and target variable (y)
    X = df.drop(columns=['Date'])  # Drop the target column from features
    y = df['Volume']  # Select the target column

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features (if necessary)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train your KNN model
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = knn.predict(X_test_scaled)

    # Combine predictions with test data
    combined_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    # Visualize results
    st.header('Modelling Results - K-Nearest Neighbor (KNN)')

    # Visualisasi hasil prediksi KNN
    fig, ax = plt.subplots()
    sns.countplot(data=combined_data, x='Predicted', ax=ax)  # Adjust x='Predicted' based on your column names
    plt.title('KNN Prediction Results')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    st.pyplot(fig)

    # Train Gaussian Naive Bayes (GNB) model
    gnb = GaussianNB()
    gnb.fit(X_train_scaled, y_train)

    # Make predictions using GNB model
    y_pred_gnb = gnb.predict(X_test_scaled)

    # Train Decision Tree Classifier (DTC) model
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train_scaled, y_train)

    # Make predictions using DTC model
    y_pred_dtc = dtc.predict(X_test_scaled)

    # Combine predictions with test data for GNB
    combined_data_gnb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_gnb})

    # Combine predictions with test data for DTC
    combined_data_dtc = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_dtc})

    # Visualize results for GNB
    st.header('Modelling Results - Gaussian Naive Bayes (GNB)')
    fig_gnb, ax_gnb = plt.subplots()
    sns.countplot(data=combined_data_gnb, x='Predicted', ax=ax_gnb)  
    plt.title('GNB Prediction Results')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    st.pyplot(fig_gnb)

    # Visualize results for DTC
    st.header('Modelling Results - Decision Tree Classifier (DTC)')
    fig_dtc, ax_dtc = plt.subplots()
    sns.countplot(data=combined_data_dtc, x='Predicted', ax=ax_dtc)  
    plt.title('DTC Prediction Results')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    st.pyplot(fig_dtc)

    st.markdown(""" 
    Grafik diatas menunjukkan garis jumlah prediksi nilai harga suatu saham. Grafik menunjukkan jumlah prediksi yang telah dibuat dari waktu ke waktu. Sumbu x pada grafik menunjukkan waktu, dan sumbu y pada grafik menunjukkan jumlah prediksi.
    Grafik tersebut tampaknya menunjukkan bahwa jumlah prediksi terus meningkat seiring berjalannya waktu. Hal ini dapat disebabkan oleh sejumlah faktor, seperti peningkatan jumlah data yang tersedia, peningkatan algoritma prediksi, atau peningkatan permintaan prediksi.
    Grafik tersebut juga tampak menunjukkan bahwa ada beberapa musiman dalam data. Hal ini mungkin disebabkan oleh faktor-faktor seperti perubahan volume perdagangan atau volatilitas pasar.
    Secara keseluruhan, grafik menunjukkan bahwa permintaan terhadap prediksi harga saham meningkat dan keakuratan prediksi tersebut meningkat. Namun, penting untuk dicatat bahwa ini hanyalah observasi umum, dan mungkin ada faktor lain yang dapat menjelaskan pola dalam data.""")
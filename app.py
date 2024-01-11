import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import pickle

#import model
svm = pickle.load(open('knn.pkl','rb'))

#load dataset
data = pd.read_csv('/content/drive/MyDrive/UAS/Heart Dataset.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Heart')


html_layout1 = """
<br>
<iframe width="560" height="315" src="https://www.youtube.com/embed/FEcOIJbhGHI?si=4LFrAJ-AOoMan6uB" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
<div style="background-color:rgb(255, 99, 71); padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Diagnosa Penyakit Jantung</b></h2>
</div>
<br>
<br>
"""

st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['KNN','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset yang memprediksi penyakit jantung</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X = data.drop('output',axis=1)
y = data['output']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():

    age = st.sidebar.number_input('Masukkan usia Anda: ',(0))
    sex  = st.sidebar.selectbox('Jenis Kelamin',(0,1))
    cp = st.sidebar.selectbox('Jenis Sakit Di Dada',(0,1,2,3))
    trtbps = st.sidebar.slider('Tekanan Darah Saat Istirahat: ', 0,140,40)
    chol = st.sidebar.slider('Cholestoral Serum mg/dl: ', 0,240,108)
    fbs = st.sidebar.slider('Gula Darah Saat Beristirahat',0,200,110)
    restecg = st.sidebar.number_input('Hasil Elektrokardiografi')
    thalachh = st.sidebar.slider('Detak Jantung Maksimum Dicapai: ', 0,180,60)
    exng = st.sidebar.selectbox('Latihan Diinduksi Angina ',(0,1))
    oldpeak = st.sidebar.number_input('Oldpeak ')
    slp = st.sidebar.number_input('kemiringan segmen ST latihan puncak:  ')
    caa = st.sidebar.selectbox('Banyak Nadi Utama',(0,1,2,3))
    thall = st.sidebar.selectbox('Kondisi',(0,1,2))

    user_report_data = {
        'age':age,
        'sex':sex,
        'cp':cp,
        'trtbps':trtbps,
        'chol':chol,
        'fbs':fbs,
        'restecg':restecg,
        'thalachh':thalachh,
        'exng':exng,
        'oldpeak':oldpeak,
        'slp':slp,
        'caa':caa,
        'thall':thall
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = svm.predict(user_data)
knn_score = accuracy_score(y_test,svm.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena penyakit jantung'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(knn_score*100)+'%')
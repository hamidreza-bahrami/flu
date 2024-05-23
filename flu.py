import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
st.set_page_config(page_title='تشخیص آنفولانزا - RoboAi', layout='centered', page_icon='🤖')

def load_model():
    with open('saved.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

rfc = data['model']
x = data['x']

def show_page():
    st.write("<h1 style='text-align: center; color: blue;'>مدل تشخیص آنفولانزا</h1>", unsafe_allow_html=True)
    st.write("<h2 style='text-align: center; color: gray;'>علائم خود را وارد کنید</h2>", unsafe_allow_html=True)
    st.write("<h4 style='text-align: center; color: gray;'>True = بله , False = خیر</h4>", unsafe_allow_html=True)
    st.write("<h4 style='text-align: center; color: gray;'>Robo-Ai.ir طراحی شده توسط</h4>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")
    with st.sidebar:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image('img.png')
        with col3:
            st.write(' ')
        st.divider()
        st.write("<h4 style='text-align: right; color: gray;'>تخمین قیمت نقره با دقت 98 درصد</h>", unsafe_allow_html=True)
        st.write("<h4 style='text-align: right; color: gray;'>ساخته شده با جمع آوری داده 15 سال بازار سهام</h>", unsafe_allow_html=True)
        st.divider()
        st.write('Developed & Designed by')
        st.write('Hamidreza Bahrami')
    
    kid = (True , False)
    kid = st.selectbox('سن زیر 20 سال', kid)

    adult = (True , False)
    adult = st.selectbox('سن بالای 20 سال', adult)

    Symptom_speed = (True , False)
    Symptom_speed = st.selectbox('عود ناگهانی علائم', Symptom_speed)

    Fever = (True , False)
    Fever = st.selectbox('تب', Fever)

    Aches = (True , False)
    Aches = st.selectbox('بدن درد', Aches)

    Chills = (True , False)
    Chills = st.selectbox('لرز', Chills)

    weakness = (True , False)
    weakness = st.selectbox('ضعف', weakness)

    Sneezing = (True , False)
    Sneezing = st.selectbox('عطسه', Sneezing)

    Chest_discomfort = (True , False)
    Chest_discomfort = st.selectbox('ناراحتی سینه', Chest_discomfort)

    Dry_cough = (True , False)
    Dry_cough = st.selectbox('سرفه خشک', Dry_cough)

    Stuffy_nose = (True , False)
    Stuffy_nose = st.selectbox('بینی گرفته', Stuffy_nose)

    Sore_throat = (True , False)
    Sore_throat	 = st.selectbox('گلودرد', Sore_throat)

    Headache = (True , False)
    Headache = st.selectbox('سردرد', Headache)

    trouble_breathing = (True , False)
    trouble_breathing = st.selectbox('تنگی نفس', trouble_breathing)

    Bluish_lips_face = (True , False)
    Bluish_lips_face = st.selectbox('رنگ پریدگی', Bluish_lips_face)

    Ribs_pulling_each_breath = (True , False)
    Ribs_pulling_each_breath = st.selectbox('درد دنده ها هنگام تنفس', Ribs_pulling_each_breath)

    Chest_pain = (True , False)
    Chest_pain = st.selectbox('درد سینه', Chest_pain)

    Severe_muscle_pain = (True , False)
    Severe_muscle_pain = st.selectbox('درد شدید عضلانی', Severe_muscle_pain)

    Dehydration = (True , False)
    Dehydration = st.selectbox('بی آبی بدن', Dehydration)

    Worsening_conditions = (True , False)
    Worsening_conditions = st.selectbox('علائم مزمن', Worsening_conditions)
    
    
    button = st.button('معاینه و تشخیص بیماری')
    if button:
        with st.chat_message("assistant"):
                with st.spinner('''درحال بررسی لطفا صبور باشید'''):
                    time.sleep(2)
                    st.success(u'\u2713''تحلیل انجام شد')
                    x = np.array([[kid, adult, Symptom_speed, Fever, Aches, Chills, weakness, Sneezing,
                                   Chest_discomfort, Dry_cough, Stuffy_nose, Sore_throat, Headache, trouble_breathing,
                                   Bluish_lips_face, Ribs_pulling_each_breath, Chest_pain, Severe_muscle_pain, Dehydration, Worsening_conditions]])
        x = x.astype(float)

        y = rfc.predict(x)
        if y == True:
            text1 = 'بر اساس تحلیل من ، شما به آنفولانزا مبتلا شده اید'
            text2 = 'برای درمان به پزشک مراجعه کنید'
            text3 = 'Based on my analysis, you have got Flu'
            text4 = 'Please visit a doctor as soon as possible'
            def stream_data1():
            for word in text1.split(" "):
                yield word + " "
                time.sleep(0.09)
            st.write_stream(stream_data1)
            def stream_data2():
                for word in text2.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data2)
            def stream_data3():
                for word in text3.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data3)
            def stream_data4():
                for word in text4.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data3)

        elif y == False:
            text1 = 'بر اساس تحلیل من ، شما به سرماخوردگی مبتلا شده اید'
            text2 = 'Based on my analysis, you have got cold'
            def stream_data1():
                for word in text1.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data1)
            def stream_data2():
                for word in text2.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data2)
show_page()

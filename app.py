import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import json
import streamlit.components.v1 as components
st.cache(suppress_st_warning=True)



st.title("Speech Emotions Recognition")

def extract_mfcc(filename):
    """extract mel frequency cepstral coefficients"""
    y,sr =  librosa.load(filename,duration=3,offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfcc

def file_upload(uploaded_files):  
    if uploaded_files is not  None:
        st.sidebar.success("successfully uploaded!")
        audio_bytes = uploaded_files.read()   
        saved_file=open('uploaded_files/'+uploaded_files.name,'wb')
        saved_file.write(audio_bytes)
        saved_file.close()
        
    return ['uploaded_files/'+uploaded_files.name]

st.sidebar.header("upload vocal files to specify emotions 3s duration at most")

uploaded_files= st.sidebar.file_uploader('upload text files',type=['.wav','.mp3'])
if uploaded_files is not  None:    
    btn = st.sidebar.button("Save")
    if btn :
        paths=file_upload(uploaded_files)
        #print(paths)
        if len(paths) > 1:
             data = []
             for path in paths:
                 data+=[extract_mfcc(path)]
             data=np.row_stack(data)     
        else:     
             data =extract_mfcc(paths[-1])
             data=np.array(data).reshape(1,40)
        
        selected=st.sidebar.selectbox('Select Language',[' ','Arabic','English'])   
        btn2 = st.sidebar.button('Emotions :)')
        print(btn2)
        if btn2: 
            if selected == 'Arabic':
                # get the data of strings of labels saved in dictionary for example high,low,neutral,etc
                fLabels = open('Data/Arabic/Arlabels_dict.json','r')
                Arlabels_dict=json.load(fLabels)
                # prepare the model and load it 
                XgbC=xgb.XGBClassifier(n_estimators=1024)
                XgbC.load_model('models/Arabic/XgbArSpeechEmotions.json')
                # check if uploaded data is one sample or more if it's more
                # use row stack otherwise predic
                #print(np.array(data).shape)
                emotions_labels =XgbC.predict(np.array(data)) 
                # sometimes the predicted values is probabilities so choose the index of the highest probability 
                # in each sample for example [1.4,2.3, 3.5] the highest probability in the index 2   
                emotions_labels = np.argmax(emotions_labels,axis=1)                
                if(len(emotions_labels)>1):
                      for label in emotions_labels:     
                             st.sidebar.text('the emotion dominates the record is '+Arlabels_dict[label])
                else:
                             st.sidebar.text('the emotion dominates the record is '+Arlabels_dict[emotions_labels])                                   
            elif selected=='English':
                # get the data of strings of labels saved in dictionary for example sadness,anger,fear,etc
                print(1)
                fLabels = open('Data/English/Enlabels_dict.json','r')
                Enlabels_dict=json.load(fLabels)
                LSTMEnglish_Speech=load_model('models/English/LSTMEnSpeechEmotions.h5')
                emotions_labels=LSTMEnglish_Speech.predict(np.expand_dims(data,-1))
                print(emotions_labels)
                if(len(emotions_labels)>1):
                      for label in emotions_labels:     
                             st.sidebar.text('the emotion dominates the record is '+Enlabels_dict[label])
                else:
                             st.sidebar.text('the emotion dominates the record is '+Enlabels_dict[emotions_labels])       
    
        



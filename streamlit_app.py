
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

def  select_box_event_handler():
        if not "initialized" in st.session_state:
                   st.session_state.return_values = []
                   st.session_state.change_values = []
                   st.session_state.initialized = True
                   st.session_state.first_return = True

        def capture_change_value():
                st.session_state.change_values.append(st.session_state.select_box)

        def capture_return_value(select_box):
               if st.session_state.first_return:
                  capture_change_value()
                  st.session_state.first_return = False
               st.session_state.return_values.append(select_box)
               

        capture_return_value(st.selectbox("Select Languages",[' ','Arabic','English'], key="select_box", on_change=capture_change_value))

 
select_box_event_handler()

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
       
        
    
        if st.session_state.change_values[-1] == 'Arabic':
                # get the data of strings of labels saved in dictionary for example high,low,neutral,etc
                fLabels = open('Data/Arabic/Arlabels_dict.json','r')
                Arlabels_dict=json.load(fLabels)
                # prepare the model and load it 
                XgbC=xgb.XGBClassifier(n_estimators=1024)
                XgbC.load_model('models/Arabic/74_Arabic_Speech_model.json')
                # check if uploaded data is one sample or more if it's more
                # use row stack otherwise predic
                #print(np.array(data).shape)
                emotions_labels =XgbC.predict(np.array(data)) 
                # sometimes the predicted values is probabilities so choose the index of the highest probability 
                # in each sample for example [1.4,2.3, 3.5] the highest probability in the index 2
                if(len(emotions_labels) > 1): 
                    emotions_labels = np.argmax(emotions_labels,axis=1)   
                #print(Arlabels_dict.keys())             
                if(len(emotions_labels)>1):
                      for label in emotions_labels:     
                             st.sidebar.text('the emotion is '+Arlabels_dict[label])
                elif(len(emotions_labels) == 1):
                             st.sidebar.text('the emotion is '+Arlabels_dict[str(emotions_labels[-1])])                                   
        elif st.session_state.change_values[-1]=='English':
                # get the data of strings of labels saved in dictionary for example sadness,anger,fear,etc
                fLabels = open('Data/English/Enlabels_dict.json','r')
                Enlabels_dict=json.load(fLabels)
                LSTMEnglish_Speech=load_model('models/English/LSTMEnSpeechEmotions.h5')
                emotions_labels=LSTMEnglish_Speech.predict(np.array(data))
                emotions_labels = np.argmax(emotions_labels,axis=1)
                emotions_labels=emotions_labels.tolist()
                if(len(emotions_labels)>1):
                      for label in emotions_labels:      
                             st.sidebar.text('the emotion is '+Enlabels_dict[label])
                elif(len(emotions_labels) == 1):
                             st.sidebar.text('the emotion is '+Enlabels_dict[str(emotions_labels[-1])])       
    

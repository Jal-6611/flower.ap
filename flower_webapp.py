# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:27:22 2021

@author: DHARMESH MISTRY
"""
import streamlit as st
import pickle
import numpy as np
DT_model=pickle.load(open('DT.pkl', 'rb'))
KNN_model=pickle.load(open('KNN.pkl', 'rb'))
NB_model=pickle.load(open('NB.pkl', 'rb'))
SVM_model=pickle.load(open('SVM.pkl', 'rb'))
def main():
     st.title("created by jal mistry")
     html_temp = """
     <div style="backgroud-color:teal ;padding:10px">
     <h2 style="color:white;text-align:center;">flower prediction system</h2>
     </div>
     """
     st.markdown(html_temp, unsafe_allow_html=True)
     activities=['decision tree','KNN','NB','SVM']
     option=st.sidebar.selectbox('which model you use?',activities)
     st.subheader(option)
     s1=st.slider('select sepal lenght',0.0,10.0)
     sw=st.slider('select sepal width',0.0,5.0)
     p1=st.slider('select petal lenght',0.0,10.0)
     pw=st.slider('select petal lenght',0.0,5.0)
     feature_list=[s1,sw,p1,pw]
     single_pred = np.array(feature_list).reshape(1,-1)
     clas=['setosa','versicolor','virginica']
     if st.button('predict'):
         if option=='decision tree':
             st.success(clas[int(DT_model.predict(single_pred))])
         elif option=='KNN':
             st.success(clas[int(KNN_model.predict(single_pred))])
         elif option=='NB':
             st.success(clas[int(NB_model.predict(single_pred))])
         else:
             st.success(clas[int(SVM_model.predict(single_pred))])
if __name__=='__main__':
    main()
       



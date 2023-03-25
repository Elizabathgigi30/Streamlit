import streamlit as st
import numpy as np
from prediction import predict
st.title('iris flower classification')
sl=st.slider('SepalLengthCm',1.0,10.0,0.5 ) 
sw=st.slider('SepalWidthCm',1.0,10.0,0.5 )
pl=st.slider('PetalLengthCm',1.0,10.0,0.5 )
pw=st.slider('PetalWidthCm',1.0,10.0,0.5)

if st.button('predict'):
    result=predict(np.array([[sl,sw,pl,pw]]))

    st.success(result[0])

import streamlit as st
import numpy as np
import pandas as pd



st.header('Hello World!')

st.subheader('this is our first streamlit app')

# st.write("Here's our first attempt at using streamlit")


with st.sidebar:
    st.header('This is the sidebar, choose something!')
    button_images = st.button("Show Images")
    button_videos = st.button("Show Videos")



col1, col2 = st.columns(2)


if button_images:

    st.subheader('Here are 2 totally different cats, find the difference if you can ;) ')

    with col1:
        st.image('https://static.streamlit.io/examples/cat.jpg', caption='Sunset Love.', use_column_width=True)
        st.write('This is a cat!')

    with col2:
        st.image('https://static.streamlit.io/examples/cat.jpg', caption='Sunset Love.', use_column_width=True)
        st.write('This is another cat!')



if button_videos:

    with col1:
        st.video('https://www.youtube.com/watch?v=jfKfPfyJRdk')
        st.write('This is a video')

    with col2:
        st.video('https://www.youtube.com/watch?v=jfKfPfyJRdk')
        st.write('This is another video')
    



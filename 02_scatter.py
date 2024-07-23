import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def chart(df, column1, column2):
    
    # st.scatter_chart(df, x=column1, y=column2)

    plt.scatter(df[column1], df[column2])
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title(f'Scatter plot of {column1} vs {column2}')
    plt.show()


st.header('Visualize Data!')

st.subheader('Please upload your csv file')

uploaded_file = st.file_uploader("Choose a file")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(5))

    st.subheader('Select the columns you want to plot against each other in a scatter plot')
    columns = df.columns.tolist()
    column1 = st.selectbox('Select column 1', columns, key='column1')
    column2 = st.selectbox('Select column 2', columns, key='column2')

    if st.button('Plot'):
        chart(df, column1, column2)


    


    st.write(f'Scatter plot of {column1} vs {column2}')




    # if st.button('Plot'):
    #     st.scatter_chart(df, x=column1, y=column2)

    #     plt.scatter(df[column1], df[column2])
    #     plt.xlabel(column1)
    #     plt.ylabel(column2)
    #     plt.title(f'Scatter plot of {column1} vs {column2}')
    #     st.pyplot(plt)







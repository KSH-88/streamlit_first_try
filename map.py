import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import random
import pickle
import datetime as dt
import plotly.express as px

API_KEY = 'bl8Xal3REkDsAbnccP2F0w==05KzhhPpuBvZsTNu'  # Replace with your actual API key

filename = "city_data.csv"

def get_city_data(city):
    url = "https://api.api-ninjas.com/v1/city"
    params = {'name': city}
    headers = {'X-Api-Key': API_KEY}
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            return data[0]  # Return the first result
    return None


def check_and_add_ctiy(city_name, df):
    
    if city_name in df['City'].values:
        city = df[df['City'] == city_name][['City', 'Latitude', 'Longitude', 'Population', 'Country']]
        
        # Display city data
        # st.write(city)
        
        return df

    else:
        city_d =  get_city_data(city_name)
        if city_d:
            city_data = pd.DataFrame({
                'City': city_d['name'],
                'Latitude': city_d['latitude'],
                'Longitude': city_d['longitude'],
                'Population': city_d['population'],
                'Country': city_d['country']
            }, index=[0])

            df = pd.concat([df,city_data], ignore_index=True)

            # st.write(city_data)
            df.to_csv('city_data.csv', index=False)
            return df
        # else:
        #     st.write(f"Could not fetch data for {city_name}")
        #     raise ValueError(f"Could not fetch data for {city_name}")




inputs = {'reanalysis_specific_humidity_g_per_kg': (11.7157142857, 20.46142857),
 'reanalysis_dew_point_temp_k': (1.5846198604989278e+151,
  2.3487169318087537e+152),
 'station_min_temp_c': (0.0, 25.6),
 'station_avg_temp_c': (10.0, 35.0714285714),
 'ndvi_average': (-0.0126, 0.5523)} 


def GenerateData(range_allowed, number_of_samples):
    samples = []
    for i in range(number_of_samples):
        samples.append(random.uniform(range_allowed[0], range_allowed[1]))
    return samples



today = dt.datetime.today()

year = today.year
weekofyear = today.isocalendar()[1] + 1

week_numbers = [(w % 53) for w in range(weekofyear - 1 , weekofyear + 53)]

weeks_sin = [np.sin(2 * np.pi * w / 53) for w in week_numbers]
weeks_cos = [np.cos(2 * np.pi * w / 53) for w in week_numbers]

year_range = []
for week in week_numbers:
    if week == 1:
        year += 1
    year_range.append(year)



# Load the trained model
with open('sj_model.pkl', 'rb') as f:
    sj_model = pickle.load(f)

with open('sj_model.pkl', 'rb') as j:
    iq_model = pickle.load(j)




def generate_data_for_the_chart(city_name, home_city):

        inputs_dic = {'year': year_range, 'weekofyear_sin': weeks_sin, 'weekofyear_cos': weeks_cos, 'weekofyear': week_numbers}

        inputs_dic_city= inputs_dic.copy()
        inputs_dic_home= inputs_dic.copy()
        for feature in inputs.keys():
            inputs_dic_city[feature] = GenerateData(inputs[feature], number_of_samples=len(week_numbers))
            inputs_dic_home[feature] = GenerateData(inputs[feature], number_of_samples=len(week_numbers))

        # Create a DataFrame with year, sin, cos as the index and keys of inputs_dic as columns
        chart_df_home = pd.DataFrame(inputs_dic_home).set_index(['year', 'weekofyear_sin', 'weekofyear_cos'])
        chart_df_city = pd.DataFrame(inputs_dic_city).set_index(['year', 'weekofyear_sin', 'weekofyear_cos'])

        prediction_home = sj_model.predict(chart_df_home)

        prediction_city = iq_model.predict(chart_df_city)


        chart_df_home['Home_city'] = prediction_home

        chart_df_home['city'] = prediction_city

        chart_df_home['city vs home'] = round(100*chart_df_home['city'] / chart_df_home['Home_city'], 1)

        chart_df_home.reset_index(inplace=True)

        chart_df_home['first_monday'] = pd.to_datetime(chart_df_home['year'].astype(str) + '-W' + chart_df_home['weekofyear'].astype(str) + '-1', format='%Y-W%W-%w')

        chart_df_home.reset_index(inplace=True)

        chart_df_home.set_index(['first_monday'], inplace=True, drop=True)

        return chart_df_home








def generate_data_for_the_map(city_name, home_city, df):
    # Load the data
    

    # Find city in the data and get data if it doesn't exist
    df = check_and_add_ctiy(city_name, df)
    df = check_and_add_ctiy(home_city, df)

    df['Color'] ='Blue'
    df.loc[df['City'] == city_name, 'Color'] = 'Red'
    df.loc[df['City'] == home_city, 'Color'] = 'Green'

    
    inputs_dic = {'year': year_range[0], 'weekofyear_sin': weeks_sin[0], 'weekofyear_cos': weeks_cos[0], 'weekofyear': week_numbers[0]}

    inputs_dic_city= inputs_dic.copy()
    df['Value'] = np.nan

    for city in df['City']:
        inputs_dic_city = inputs_dic.copy()
        for feature in inputs.keys():
            inputs_dic[feature] = GenerateData(inputs[feature], number_of_samples=len(week_numbers))

        df_prediction = pd.DataFrame(inputs_dic).set_index(['year', 'weekofyear_sin', 'weekofyear_cos'])

        value = iq_model.predict(df_prediction).reset_index(drop=True)

        df.loc[df['City'] == city, 'Value'] = value[0]

        home_city_value = df[df['City'] == home_city]['Value'].iloc[0]

    df['odds vs home'] = round(100*df['Value'] / home_city_value, 1)

    return df







# Streamlit app



def main():
    st.markdown("<h1 style='text-align: center;'>Odds of catching Dengue around the world compared to your city</h1>", unsafe_allow_html=True)


    st.subheader(f'Today is {today.strftime("%Y-%m-%d")}')

    # User home city
    home_city = st.text_input("Enter your city name:")

    # home_city = 'Rome'
    
    # User input
    city_name = st.text_input("Enter a city name:")
    # city_name = 'Paris'

    if home_city != '' and city_name != '':

        df = pd.read_csv('city_data.csv')

        df = check_and_add_ctiy(city_name, df)
        df = check_and_add_ctiy(home_city, df)

        df = generate_data_for_the_map(city_name, home_city, df)

        df['Color'] ='Rest of the world'
        df.loc[df['City'] == city_name, 'Color'] = 'Desired city'
        df.loc[df['City'] == home_city, 'Color'] = 'Home city'



        column_dict =  {'City': False, 'Latitude': False, 'Longitude': False, 'Population': True, 'Country': True, 'Value': False, 'Color': False, 'odds vs home': True}

        #  Display the map

        st.markdown("<h1 style='text-align: center;'>Odds of catching Dengue in different cities compared to your home city</h1>", unsafe_allow_html=True)
        



        fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Color", hover_name="City", hover_data=column_dict, size="odds vs home",
                                mapbox_style="carto-positron", zoom=0, width=1200, height=800)

    

        fig.update_layout(legend_title_text='Cities')
    

        fig.update_layout(mapbox=dict(center=dict(lat=df.loc[df['City'] == home_city, 'Latitude'].values[0],
                                                lon=df.loc[df['City'] == home_city, 'Longitude'].values[0])),
                        margin=dict(l=0, r=0, t=0, b=0))



        st.plotly_chart(fig, use_container_width=False)
    
        
        st.markdown("<h1 style='text-align: center;'>Odds of catching Dengue in the choosen city compared to your home city for 1 year ahead</h1>", unsafe_allow_html=True)

                
        # Generate the chart
        chart_df_home = generate_data_for_the_chart(city_name, home_city)
        st.line_chart(chart_df_home, y='city vs home')
        # st.text(f'odds of catching Dengue in your city: {home_city} againtst {city_name} in % for 1 year ahead starting from {today.strftime("%Y-%m-%d")}')


    else:
        st.write("Please enter the cities names")






if __name__ == '__main__':
    main()
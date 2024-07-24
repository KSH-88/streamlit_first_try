import csv
import random
import requests
import time

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

def check_and_add_ctiy(city_name):

    df = pd.read_csv('city_data.csv')
    if city_name in df['City'].values:
        city = df[df['City'] == city_name][['City', 'Latitude', 'Longitude', 'Population', 'Country']]
        
        # Display city data
        st.write(city)
        
        return df


    else:
        city_d =  get_city_data(city_name)
        if city_d:
            city_data = city_data.append({
                'City': city_d['name'],
                'Latitude': city_d['latitude'],
                'Longitude': city_d['longitude'],
                'Population': city_d['population'],
                'Country': city_d['country']
            }, ignore_index=True)

            df = df.append(city_d, ignore_index=True)

            st.write(city_data)
            df.to_csv('city_data.csv', index=False)
            return df
        else:
            st.write(f"Could not fetch data for {city_name}")


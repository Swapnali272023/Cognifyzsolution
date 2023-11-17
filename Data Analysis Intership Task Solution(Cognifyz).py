#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install folium')
get_ipython().system('pip install vaderSentiment')
get_ipython().system('pip install wordcloud')


# In[2]:


# import the needed libraries
import numpy as np
import pandas as pd
import ast 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 10,6
plt.xkcd()
get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud
from folium.plugins import HeatMap
import folium
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import r2_score


# ## Level 1	Where Data Meets Intelligence

# ### Task 1: City Analysis
# 
# 

# In[3]:


#load data to pandas dataframe
data=pd.read_csv(r"C:\Users\admin\Downloads\Dataset.csv")


# In[4]:


data.head()


# In[5]:


data.tail(5)


# In[6]:


data.shape


# In[7]:


data.columns


# In[8]:


data.count()


# In[9]:


# get some info about data
data.info()


# In[10]:


data.describe()


# In[11]:


data.dtypes


# In[12]:


data.isnull().sum()


# In[13]:


data.dropna(how='any')


# In[14]:


data.head(5)


# In[15]:


#check for duplicated values
data.duplicated().sum()


# In[16]:


# check for null values
((data.isna().sum()/data.shape[0])*100).round(2)


# In[17]:


data.columns


# In[18]:


data = data.rename(columns={'Number ': 'Number'})


# In[19]:


data.columns = data.columns.str.strip()


# In[20]:


# drop null values for aggregate_rating and average_cost as they would be compared later
data.dropna(subset=['Aggregate rating', 'Average Cost for two'],inplace=True)


# In[21]:


data.shape


# In[22]:


data= data[data['Has Online delivery'].astype('str').map(len) < 4]


# In[23]:


# check for percentage of null values 
((data.isna().sum()/data.shape[0])*100).round(2)


# In[24]:


# check for values
data.Average_cost_for_two.unique()


# In[ ]:


bins = [0,200, 500, 1000, 3000, 6000]
labels = ['very cheap','cheap','moderate','expensive','very expensive']
data['cost_range']= pd.cut(data.Average_cost_for_two, bins=bins,labels=labels)


# In[ ]:


data.loc[:,['Average_cost','cost_range']]


# ### Task 2:1)Identify the city with the highest number of restaurants in the dataset.
# ###             2)Calculate 	the 	average 	rating 	for restaurants in each city.

# In[ ]:


city_counts = data.groupby('City').count()


# In[ ]:


sorted_cities = city_counts.sort_values('Restaurant Name', ascending=False)


# In[ ]:


city_with_most_restaurants = sorted_cities.index[0]


# In[ ]:


print(f"The city with the highest number of restaurants is: {city_with_most_restaurants}")


# In[ ]:


city_ratings = data.groupby('City')['Aggregate rating'].mean()


# In[ ]:


print(city_ratings)


# In[ ]:


data.columns


# In[ ]:


sns.heatmap(data[['Index([]].corr(),cmap="YlGnBu",  annot=True);


# ### Task3:Price Range Distribution
# ### 1)Determine the city with the highest average rating.
# ### 2)Create a histogram or bar chart to visualize the distribution of price ranges among the restaurants.
# ### 3)Calculate the percentage of restaurants in each price range category.

# In[ ]:


city_ratings = data.groupby('City')['Aggregate rating'].mean()


# In[ ]:


city_with_highest_rating = city_ratings.idxmax()


# In[ ]:


print(f"The city with the highest average rating is: {city_with_highest_rating}")


# In[ ]:


plt.hist(data['Price range'], bins=4, align='left', rwidth=0.75)
plt.xlabel('Price Range')
plt.ylabel(' Restaurants ID')
plt.title('Distribution of Price Ranges Among Restaurants')
plt.show()



# In[ ]:


# group the restaurants by price range and count the number of restaurants in each group

restaurant_count = data.groupby('Price range').count()['Restaurant Name']


# In[ ]:


# calculate the total number of restaurants
total_restaurants = restaurant_count.sum()


# In[ ]:


# calculate the percentage of restaurants in each price range category
restaurant_percentage = (restaurant_count / total_restaurants) * 100


# In[ ]:


# print the results
print(restaurant_percentage)


# ### Task 4: Online Delivery
# ### 1)Determine the percentage of restaurants that offer online delivery.
# ### 2)Compare the average ratings of restaurants with and without online delivery.
# 

# In[ ]:


# count the number of restaurants that offer online delivery
delivery_count = data['Has Online delivery'].value_counts()['Yes']

# calculate the total number of restaurants
total_restaurants = len(data)

# calculate the percentage of restaurants that offer online delivery
delivery_percentage = (delivery_count / total_restaurants) * 100

# print the results
print("Percentage of restaurants that offer online delivery: {:.2f}%".format(delivery_percentage))


# In[ ]:


# calculate the average rating of restaurants with online delivery
with_delivery_avg_rating = data[data['Has Online delivery'] == 'Yes']['Aggregate rating'].mean()

# calculate the average rating of restaurants without online delivery
without_delivery_avg_rating = data[data['Has Online delivery'] == 'No']['Aggregate rating'].mean()

# print the results
print("Average rating of restaurants with online delivery: {:.2f}".format(with_delivery_avg_rating))
print("Average rating of restaurants without online delivery: {:.2f}".format(without_delivery_avg_rating))


# In[ ]:





# In[ ]:


data.columns


# ### Level 2:

# ### Task 1: Restaurant Ratings
# ### 1) Analyze the distribution of aggregate ratings and determine the most common rating range.
# ### 2)Calculate the average number of votes received by restaurants.
# 

# In[ ]:


plt.hist("Aggregate rating", bins=10)
plt.xlabel('Aggregate rating')
plt.ylabel('Frequency')
plt.title('Distribution of Aggregate Ratings')
plt.show()


# In[ ]:


total_votes = sum(data.Votes)


# In[ ]:


num_restaurants = len(data.Votes)


# In[ ]:


avg_votes = total_votes / num_restaurants


# In[ ]:


print("The average number of votes per restaurant is:", avg_votes)


# ### Task 2:

# ### Task: Cuisine Combination
# ### 1) Identify the most common combinations of cuisines in the dataset.
# ### 2)Determine if certain cuisine combinations tend to have higher ratings.
# 
# 
# 

# In[ ]:


from collections import defaultdict
import pandas as pd

# Assuming the dataset is stored in a pandas DataFrame called "data"
cuisine_combinations = defaultdict(int)

for index, row in data.iterrows():
    cuisines = pd.Series(row['Cuisines'])
    cuisines = cuisines.sort_values().reset_index(drop=True)  # Sort cuisines to ensure consistent ordering
    key = tuple(cuisines)  # Convert list of cuisines to a tuple for use as dictionary key
    cuisine_combinations[key] += 1

# Sort the cuisine_combinations dictionary by the number of restaurants and print out the top N combinations
N = 10  # Top N combinations to print
sorted_combinations = sorted(cuisine_combinations.items(), key=lambda x: x[1], reverse=True)
for i in range(N):
    combination = sorted_combinations[i][0]
    count = sorted_combinations[i][1]
    print(f"{i+1}. {', '.join(combination)} ({count} restaurants)")


# In[ ]:


# create a column for cuisine combinations
data['Cuisine Combination'] = data['Cuisines'].str.split(', ')

# calculate average rating for each cuisine combination
ratings = data.explode('Cuisine Combination').groupby('Cuisine Combination')['Aggregate rating'].mean()

# sort ratings in descending order
sorted_ratings = ratings.sort_values(ascending=False)

# print the top 10 cuisine combinations with highest ratings
for i, cuisine_combination in enumerate(sorted_ratings.index[:10]):
    print(f"{i+1}. {cuisine_combination}: {sorted_ratings.loc[cuisine_combination]}")


# In[ ]:


data['Cuisine Combination'] = data['Cuisines'].str.split(', ')


# In[ ]:


ratings = data.explode('Cuisine Combination').groupby('Cuisine Combination')['Aggregate rating'].mean()


# In[ ]:


sorted_ratings = ratings.sort_values(ascending=False)


# In[ ]:


for i, cuisine_combination in enumerate(sorted_ratings.index[:10]):
    print(f"{i+1}. {cuisine_combination}: {sorted_ratings.loc[cuisine_combination]}")


# In[ ]:


data.columns


# ### Task 3: Geographic Analysis
# ### 1)Plot the locations of restaurants on a map using longitude and latitude coordinates.
# ### 2)Identify any patterns or clusters of restaurants in specific areas.

# In[ ]:


import folium

# create map centered on the first restaurant in the data
lat, lon = data.loc[0, ['Latitude', 'Longitude']]
map = folium.Map(location=[lat, lon], zoom_start=12)

# add markers for each restaurant
for i, row in data.iterrows():
    lat, lon = row['Latitude'], row['Longitude']
    name = row['Restaurant Name']
    marker = folium.Marker([lat, lon], popup=name)
    marker.add_to(map)

# show the map
map


# In[ ]:


import folium
from sklearn.cluster import KMeans

# create map centered on the first restaurant in the data
lat, lon = data.loc[0, ['Latitude', 'Longitude']]
map = folium.Map(location=[lat, lon], zoom_start=12)

# create KMeans clusters
X = data[['Latitude', 'Longitude']]
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
data['Cluster'] = kmeans.labels_

# add markers for each restaurant, color coded by cluster
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i, row in data.iterrows():
    lat, lon = row['Latitude'], row['Longitude']
    name = row['Restaurant Name']
    cluster = row['Cluster']
    marker = folium.Marker([lat, lon], popup=name, icon=folium.Icon(color=colors[cluster]))
    marker.add_to(map)

# show the map
map


# ### Task 4: Restaurant Chains
# ### 1)Identify if there are any restaurant chains present in the dataset.
# ### 2)Analyze the ratings and popularity of different restaurant chains.

# In[ ]:


# create a new column for restaurant chains
data['Chain'] = data['Name'].apply(lambda x: x.split()[0])

# count the number of restaurants in each chain
chain_counts = data['Chain'].value_counts()

# print the chains with more than one restaurant
print("Restaurant chains present in the dataset:")
print(chain_counts[chain_counts > 1].index.tolist())


# In[25]:


data.columns


# ### Level 3
# 

# ### Task 1: Restaurant Reviews
# ### Analyze the text reviews to identify the most common positive and negative keywords.
# ### Calculate the average length of reviews and explore if there is a relationship between review length and rating
# 
# 
# 

# In[31]:


data['Review_Length'] = data['Review'].apply(lambda x: len(x.split()))  # Assuming words are separated by spaces

# Calculate the average review length
average_review_length = data['Review_Length'].mean()
print("Average Review Length:", average_review_length)

# Explore the relationship between review length and rating
sns.scatterplot(data=data, x='Review_Length', y='Aggregate rating')
plt.title('Relationship between Review Length and Rating')
plt.xlabel('Review Length')
plt.ylabel('Aggregate rating')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ### Task 2: Votes Analysis
# ### Identify the restaurants with the highest and lowest number of votes.
# ### Analyze if there is a correlation between the number of votes and the rating of a restaurant.
# 
# 
# 

# In[ ]:


data.columns


# In[32]:


# Identify the restaurant with the highest number of votes
restaurant_highest_votes = data[data['Votes'] ==data['Votes'].max()]
print("Restaurant with the highest number of votes:")
print(restaurant_highest_votes)

# Identify the restaurant with the lowest number of votes
restaurant_lowest_votes = data[data['Votes'] ==data['Votes'].min()]
print("Restaurant with the lowest number of votes:")
print(restaurant_lowest_votes)


# In[34]:


# Create a scatter plot to visualize the relationship between the number of votes and the rating
sns.scatterplot(data=data, x='Votes', y='Aggregate rating')
plt.title('Number of Votes vs. Aggregate rating')
plt.xlabel('Number of Votes')
plt.ylabel('Aggregate rating')
plt.show()


# In[33]:


data.columns


# In[ ]:





# In[ ]:





# ### Task 3: Price Range vs. Online Delivery and Table Booking
# ### Analyze if there is a relationship between the price range and the availability of online delivery and table booking.
# ### Determine if higher-priced restaurants are more likely to offer these services.
# 
# 
# 
# 
# 

# In[35]:


data.columns


# In[36]:


# Create a count plot to visualize the availability of online delivery and table booking based on different price ranges
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Price range', hue='Has Online delivery')
plt.title
plt.xlabel('Price range')
plt.ylabel('Count')
plt.title('Availability of Online Delivery by Price Range')
plt.legend(title='Online Delivery', labels=['Not Available', 'Available'])
plt.show()

# Create a count plot to visualize the availability of table booking based on different price ranges
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Price range', hue='Has Table booking')
plt.xlabel('Price range')
plt.ylabel('Count')
plt.title('Availability of Table Booking by Price Range')
plt.legend(title='Table Booking', labels=['Not Available', 'Available'])
plt.show()



# In[37]:


# Calculate the proportion of restaurants offering online delivery within each price range category
online_delivery_proportions = data.groupby('Price range')['Has Online delivery'].value_counts(normalize=True).unstack()

# Calculate the proportion of restaurants offering table booking within each price range category
table_booking_proportions = data.groupby('Price range')['Has Table booking'].value_counts(normalize=True).unstack()

print("Proportion of Restaurants Offering Online Delivery by Price Range:")
print(online_delivery_proportions)

print("Proportion of Restaurants Offering Table Booking by Price Range:")
print(table_booking_proportions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





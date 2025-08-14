#!/usr/bin/env python
# coding: utf-8

# # Indicators of Anxiety or Depression Based on Report Frequency Of Symptoms During The Last 7 Days
# 
# Natalie Dume and Ryleigh Harvey

# The dataset was derived from a 20-minute online survey, The Household Pulse Survey, given to people in various age groups (19-80+ years old) by the National Center for Health Statistics (NCHS) and the US Census Bureau. It was designed to provide information on the impact of COVID-19. The data collecting began in April 23, 2020.
# 
# The Household Pulse Survey was designed to gauge the impact of the pandemic on employment status, consumer spending, food security, housing, education disruptions, and dimensions of physical and mental wellness.
# 
# The dataset focuses on the frequency of depression and anxiety symptoms through the course a 7 day period. The questions are a modified version of 2 other surveys: the Patient Health Questionnaire (PHQ-2) and the Generalized Anxiety Disorder (GAD-2) scale on the Household Pulse Survey, collecting information on symptoms over the last 7 days (rather than the typical 14 days). Beginning in Phase 3.2 (July 21, 2021) of data collection and reporting, the question reference period changed from the ‘last 7 days’ to the ‘last two weeks’.
# 
# The main features tested was the "Indicator", which indicated if the individual was experiencing Symptoms of Depressive Disorder, Symptoms of Anxiety Disorder or Both and the "Subgroup" which determined whether a Male or Female completed the survey
# 
# To further understand this data, the K-Nearest Neighbor was the most appropiate approach, as it was used to make accurate predictions of the frequencies of these symptoms for males and females.

# In[1]:


#Add necessary Libraries and mods
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from statistics import mode

# Read in a CSV file
data = pd.read_csv('data_sets/Indicators_of_Anxiety_or_Depression_Based_on_Reported_Frequency_of_Symptoms_During_Last_7_Days (1).csv')
data.head()


# Here, certain columns deemed unnecessary are dropped from the DataFrame data.

# In[2]:


# Removing unecessary columns from the data
columns_to_drop = ["Quartile Range", "Confidence Interval","Phase","Time Period","Time Period Label"]

# Drop the specified columns
data.drop(columns=columns_to_drop, inplace=True)


# In[3]:


# Remove rows where non-existent values occur in specific columns
cleaned_data = data.dropna(subset=['Value'])
cleaned_data.head(50)


# In[4]:


# Make a copy of the DataFrame slice
gender_data = cleaned_data[cleaned_data['Group'] == "By Sex"].copy()
# Drop columns from the copied DataFrame
not_needed = ["Time Period Start Date", "Time Period End Date"]
gender_data.drop(columns=not_needed, inplace=True)


# In[5]:


gender_data.head()


# In[6]:


gender_data.shape


# In[7]:


Train = gender_data[:195].copy()


# A new column 'Distance' is added to the training data DataFrame Train, initialized with a value of 9999.

# In[8]:


Train.loc[:, 'Distance'] = 9999
Train.head(70)


# Separate DataFrames for male and female data are created from the training data.

# In[9]:


Male_Train = Train[Train["Subgroup"] ==  "Male"].copy()
Female_Train = Train[Train["Subgroup"]=="Female"].copy()


# Target values for males and females are defined as pandas Series and determined by the mean of Value, Low CI and High CI.

# In[10]:


Male_Train["Value"].mean()


# In[11]:


Female_Train["Value"].mean()


# In[12]:


Male_Train["Low CI"].mean()


# In[13]:


Female_Train["Low CI"].mean()


# In[14]:


Male_Train["High CI"].mean()


# In[15]:


Female_Train["High CI"].mean()


# In[16]:


Male_Target = pd.Series([27.89,26.75,29.07])
Male_Target


# In[17]:


Female_Target = pd.Series([34.56,33.61,35.52])
Female_Target


# Predicted values for males and females are calculated using the mean of the 'Value' column for the top k rows of each gender's training data.
# 

# In[18]:


# For male:
k = 30
predicted_values_male = Male_Train.head(k)['Value'].mean()

# For female:
k2 = 30
predicted_values_female = Female_Train.head(k2)['Value'].mean()

# Display the predicted values
print("Predicted Value for Males:", predicted_values_male)
print("Predicted Value for Females:", predicted_values_female)


# In[19]:


# Calculate the Euclidean distance for each row
Male_Train["Distance"] = ((Male_Train["Value"] - Male_Target[0])**2 + 
                          (Male_Train["Low CI"] - Male_Target[1])**2 + 
                          (Male_Train["High CI"] - Male_Target[2])**2)**0.5

# Display the result
Male_Train.head(10)


# In[20]:


# Calculate the Euclidean distance for each row
Female_Train["Distance"] = ((Female_Train["Value"] - Female_Target[0])**2 + 
                          (Female_Train["Low CI"] - Female_Target[1])**2 + 
                          (Female_Train["High CI"] - Female_Target[2])**2)**0.5

# Display the result
Female_Train.head(10)


# In[21]:


k = 30
Male_Train = Male_Train.sort_values("Distance", ascending=True)
knn = list(Male_Train.head(k).Indicator)
knn


# In[22]:


k2 = 30
Female_Train = Female_Train.sort_values("Distance", ascending=True)
knn2 = list(Female_Train.head(k2).Indicator)
knn2


# the mode (most common value) of the lists knn and knn2 is calculated and printed to determine the predicted class (indicator) for males and females

# In[23]:


print(mode(knn))


# In[24]:


print(mode(knn2))


# ### Male Scatter Plot For Anxiety and Depression Data

# In[25]:


colors = {'Symptoms of Anxiety Disorder': 'red',
          'Symptoms of Depressive Disorder': 'blue',
          'Symptoms of Anxiety Disorder or Depressive Disorder': 'purple'}

# Scatter plot for Male_Train data
for indicator, color in colors.items():
    # Shorten the label if it's too long
    label = 'Anxiety or Depressive' if indicator == 'Symptoms of Anxiety Disorder or Depressive Disorder' else indicator
    subset = Male_Train[Male_Train['Indicator'] == indicator]
    plt.scatter(subset['Low CI'], subset['High CI'], c=color, label=label, s=20) # Adjust s value for marker size

# Scatter plot for Male_Target data
plt.scatter(Male_Target[1], Male_Target[2], c="lightgreen", label='Target Data', s=20) # Adjust s value for marker size

plt.xlabel('Low CI')
plt.ylabel('High CI')
plt.title('Male Data Scatter Plot')

# Create legend and move it to the bottom right corner
plt.legend(loc='lower right')

plt.show()


# ### Female Scatter Plot For Anxiety and Depression Data

# In[26]:


colors = {'Symptoms of Anxiety Disorder': 'red',
          'Symptoms of Depressive Disorder': 'blue',
          'Symptoms of Anxiety Disorder or Depressive Disorder': 'purple'}

# Scatter plot for Female_Train data
for indicator, color in colors.items():
    # Shorten the label if it's too long
    label = 'Anxiety or Depressive' if indicator == 'Symptoms of Anxiety Disorder or Depressive Disorder' else indicator
    subset = Female_Train[Female_Train['Indicator'] == indicator]
    plt.scatter(subset['Low CI'], subset['High CI'], c=color, label=label, s=20) # Adjust s value for marker size

# Scatter plot for Female_Target data
plt.scatter(Female_Target[1], Female_Target[2], c="lightgreen", label='Target Data', s=20) # Adjust s value for marker size

plt.xlabel('Low CI')
plt.ylabel('High CI')
plt.title('Female Data Scatter Plot')

# Create legend and move it to the bottom right corner
plt.legend(loc='lower right')

plt.show()


# ## Comparative Dataset: Mental Health Care in the Last Four Weeks

# In[27]:


data2 = pd.read_csv('data_sets/Mental_Health_Care_in_the_Last_4_Weeks.csv')
data2.head()


# In[28]:


#Removing Unecessary columns from the data2
columns_drops = ["Suppression Flag","Quartile Range","Confidence Interval","Phase","Time Period","Time Period Label","Time Period Start Date","Time Period End Date"]
data2.drop(columns=columns_drops,inplace=True)


# In[29]:


# Remove rows where non-existent values occur in specific columns
cleaned_data2 = data2.dropna(subset=['Value'])
cleaned_data2.head(50)


# In[30]:


# Make a copy of the DataFrame slice
gender_data2 = cleaned_data2[cleaned_data2['Group'] == "By Sex"].copy()
gender_data2.head(40)


# In[31]:


gender_data2.shape


# In[32]:


Train2 = gender_data2[:132].copy()


# In[33]:


# Set the 'Distance' column to 99 for all rows in the DataFrame 'Train'
Train2.loc[:, 'Distance'] = 9999
Train2.head(70)


# In[34]:


Male_Train2 = Train2[Train2["Subgroup"] ==  "Male"].copy()
Female_Train2 = Train2[Train2["Subgroup"]=="Female"].copy()


# In[35]:


Male_Train2["Value"].mean()


# In[36]:


Male_Train2["LowCI"].mean()


# In[37]:


Male_Train2["HighCI"].mean()


# In[38]:


Male_Target2 = pd.Series([12.31,11.56,13.08])
Male_Target2


# In[39]:


Female_Train2["Value"].mean()


# In[40]:


Female_Train2["LowCI"].mean()


# In[41]:


Female_Train2["HighCI"].mean()


# In[42]:


Female_Target2 = pd.Series([20.06,19.34,20.78])
Female_Target2


# In[43]:


# Calculate the Euclidean distance for each row
Male_Train2["Distance"] = ((Male_Train2["Value"] - Male_Target2[0])**2 + 
                          (Male_Train2["LowCI"] - Male_Target2[1])**2 + 
                          (Male_Train2["HighCI"] - Male_Target2[2])**2)**0.5

# Display the result
Male_Train2.head(10)


# In[44]:


# Calculate the Euclidean distance for each row
Female_Train2["Distance"] = ((Female_Train2["Value"] - Female_Target2[0])**2 + 
                          (Female_Train2["LowCI"] - Female_Target2[1])**2 + 
                          (Female_Train2["HighCI"] - Female_Target2[2])**2)**0.5

# Display the result
Female_Train2.head(10)


# In[45]:


k02 = 30
Male_Train2 = Male_Train2.sort_values("Distance", ascending=True)
knn02 = list(Male_Train2.head(k).Indicator)
knn02


# In[46]:


k002 = 30
Female_Train2 = Female_Train2.sort_values("Distance", ascending=True)
knn002 = list(Female_Train2.head(k).Indicator)
knn002


# In[47]:


print(mode(knn02))


# In[48]:


print(mode(knn002))


# ### Male Scatter Plot For Mental Health Data

# In[49]:


colors = {'Received Counseling or Therapy, Last 4 Weeks': 'red',
          'Took Prescription Medication for Mental Health And/Or Received Counseling or Therapy, Last 4 Weeks': 'blue',
          'Needed Counseling or Therapy But Did Not Get It, Last 4 Weeks': 'purple',
          'Took Prescription Medication for Mental Health, Last 4 Weeks':'pink'}

# Scatter plot for Male2_Train data
for indicator, color in colors.items():
    # Shorten the label if it's too long
    label = 'Medication and/or Therapy' if indicator == 'Took Prescription Medication for Mental Health And/Or Received Counseling or Therapy, Last 4 Weeks' else indicator
    label = 'Needed Therapy but Not Received' if indicator =="Needed Counseling or Therapy But Did Not Get It, Last 4 Weeks" else label
    label = 'Prescription Medication' if indicator == "Took Prescription Medication for Mental Health, Last 4 Weeks" else label
    label = 'Recieved Therapy' if indicator == 'Received Counseling or Therapy, Last 4 Weeks' else label
    subset = Male_Train2[Male_Train2['Indicator'] == indicator]
    plt.scatter(subset['LowCI'], subset['HighCI'], c=color, label=label, s=20) # Adjust s value for marker size

# Scatter plot for Male_Target data
plt.scatter(Male_Target[1], Male_Target[2], c="lightgreen", label='Target', s=20) # Adjust s value for marker size

plt.xlabel('Low CI')
plt.ylabel('High CI')
plt.title('Male Data Scatter Plot')

# Create legend and move it to the bottom right corner
plt.legend(loc='upper left')

plt.show()


# ### Female Scatter Plot For Mental Health Data

# In[50]:


colors = {'Received Counseling or Therapy, Last 4 Weeks': 'red',
          'Took Prescription Medication for Mental Health And/Or Received Counseling or Therapy, Last 4 Weeks': 'blue',
          'Needed Counseling or Therapy But Did Not Get It, Last 4 Weeks': 'purple',
          'Took Prescription Medication for Mental Health, Last 4 Weeks':'pink'}

# Scatter plot for Female_Train2 data
for indicator, color in colors.items():
    # Shorten the label if it's too long
    label = 'Medication and/or Therapy' if indicator == 'Took Prescription Medication for Mental Health And/Or Received Counseling or Therapy, Last 4 Weeks' else indicator
    label = 'Needed Therapy but Not Received' if indicator =="Needed Counseling or Therapy But Did Not Get It, Last 4 Weeks" else label
    label = 'Prescription Medication' if indicator == "Took Prescription Medication for Mental Health, Last 4 Weeks" else label
    label = 'Recieved Therapy' if indicator == 'Received Counseling or Therapy, Last 4 Weeks' else label
    subset = Female_Train2[Female_Train2['Indicator'] == indicator]
    plt.scatter(subset['LowCI'], subset['HighCI'], c=color, label=label, s=20) # Adjust s value for marker size

# Scatter plot for Male_Target data
plt.scatter(Female_Target[1], Female_Target[2], c="lightgreen", label='Target', s=20) # Adjust s value for marker size

plt.xlabel('Low CI')
plt.ylabel('High CI')
plt.title('Female Data Scatter Plot')

# Create legend and move it to the bottom right corner
plt.legend(loc='upper left')

plt.show()


# ### Comparison

# In[ ]:





# # Conclusion

# After using K-Nearest Neighbor algorithm to train and test the data, and comparing both the Low Confidence Interval and the High Confidence Interval for males and females, it is evident by the scatterplots that Symptoms of Depressive Disorder had a much lower chance of being inaccurately predicted. These symptoms produced lower and more narrow Confidence Interval ranges. 
# In contrast, both groups expressed higher frequencies of experiencing symptoms from both anxiety and depression, a direct result of the mental affects of the coronavirus pandemic.
# 
# It's also evident that females reported higher frequencies of these symptoms than males, making for higher confidence intervals. The target data points seem to correlate more closely with the Symptoms of Anxiety Disorder as the data points are clustered the closest to the target data point, meaning the predictions are slightly more accurate and accounting for less error. By choosing a K-value of 30 and using a Euclidean distance metric (the data is continuous), it was much easier to test and train the data according to Gender.
# 
# Males reported more about experiencing symptoms of both Anxiety and Depression whereas females seemed to experience more symptoms in relation to Anxiety overall.
# 
# This data and model analysis can better allocate mental health resources for people during a traumatic event such as the pandemic as well as develop more effective programs. It can also aid in early detection and intervention which can prevent the progression of mild symptoms into more severe disorders. Knowing which populations are at risk allows healthcare systems to integrate mental health services with primary care, promoting holistic treatment approaches.

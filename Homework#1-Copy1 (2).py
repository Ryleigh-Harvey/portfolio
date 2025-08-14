#!/usr/bin/env python
# coding: utf-8

# Description and Title

# In[1]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


# In[2]:


#read data from csv file
df = pd.read_csv('data_sets/Deaths_in_122_U.S._cities_-_1962-2016._122_Cities_Mortality_Reporting_System.csv')


# In[3]:


# identifying 13 columns
ndf = df.iloc[:,:13]


# In[4]:


ndf.head()


# In[5]:


ndf.tail()


# In[6]:


ndf.describe()


# In[7]:


ndf.info()


# In[8]:


ndf.shape


# In[9]:


len(ndf)


# In[10]:


ndf.columns =['year','week','week_ending','region','state','city','Pn_Flu_Deaths','all_deaths','less_than_one','one_to_twenty-four','twenty-five_forty-four','forty-five_sixty-four','sixty-five_plus']


# In[11]:


print(ndf.dtypes)


# showing lot

# In[12]:


plt.figure()
ndf.plot()


# In[13]:


plt.figure(figsize= (12, 6))
ndf['all_deaths'].plot(color='purple')
ndf['Pn_Flu_Deaths'].plot(color='violet')
plt.grid(True, color='gray')
#comparing all death and death by flu or Pn


# In[14]:


group_by_year = ndf.loc[:, ['year', 'all_deaths', 'Pn_Flu_Deaths']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y2 = avgs.Pn_Flu_Deaths
y1 = avgs.all_deaths
def plot(x, y1,y2, title, y_label):
    plt.figure(figsize= (12, 6))
    plt.rcParams["figure.autolayout"] = True
    fig, ax1 = plt.subplots()
    line1 =ax1.plot(x,y1, color="magenta")
    ax2 = ax1.twinx()
    line2= ax2.plot(y2)
    ax1.set_title(title)
    ax1.set_ylabel(y_label)
    #ax1.margins(x=0, y=0)
    ax1.legend('First Line', loc='upper center')
    ax2.legend('Second Line', loc="lower center")
    fig.tight_layout()
    plt.grid(True, color='gray')
    plt.show()
    #comparing all deaths and flu deaths


# comparing total deaths and flu and pn deaths

# In[15]:


plot(x,y1,y2,"Death over the years", "Deaths")


# In[16]:


ndf[ndf['year'] == 1962]


# In[17]:


ndf[ndf['year'] == 2016]


# description: compare the two graphs which are showing the deaths by age good in a a soecific year

# In[18]:


ndf_1962 = ndf[ndf['year'] == 1962]

grouped_data = ndf_1962.groupby('week')[['less_than_one', 'one_to_twenty-four', 'twenty-five_forty-four', 'forty-five_sixty-four', 'sixty-five_plus']].sum()

plt.figure(figsize=(12,7))
plt.plot(grouped_data.index, grouped_data['less_than_one'], label='< 1', color='darkblue',lw= 3)
plt.plot(grouped_data.index, grouped_data['one_to_twenty-four'], label=' 1 - 24', color='darkgreen',lw= 3)
plt.plot(grouped_data.index, grouped_data['twenty-five_forty-four'], label='25 - 44', color='red',lw= 3)
plt.plot(grouped_data.index, grouped_data['forty-five_sixty-four'], label=' 45 - 64', color='darkorange',lw= 3)
plt.plot(grouped_data.index, grouped_data['sixty-five_plus'], label='65+', color='violet', lw = 3)

plt.xlabel('Number of Weeks')
plt.ylabel('Frequency of Deaths')
plt.title('Frequency of Deaths in 1962 by Age Group(ALL CITIES)')
plt.legend()
plt.grid(True, color='black')
plt.show()


# In[19]:


ndf_2016 = ndf[ndf['year'] == 2016]

grouped_data = ndf_2016.groupby('week')[['less_than_one', 'one_to_twenty-four', 'twenty-five_forty-four', 'forty-five_sixty-four', 'sixty-five_plus']].sum()

plt.figure(figsize=(12,7))
plt.plot(grouped_data.index, grouped_data['less_than_one'], label='< 1', color='darkblue',lw= 1.75)
plt.plot(grouped_data.index, grouped_data['one_to_twenty-four'], label=' 1 - 24', color='darkgreen',lw= 2)
plt.plot(grouped_data.index, grouped_data['twenty-five_forty-four'], label='25 - 44', color='red',lw= 3)
plt.plot(grouped_data.index, grouped_data['forty-five_sixty-four'], label=' 45 - 64', color='darkorange',lw= 3)
plt.plot(grouped_data.index, grouped_data['sixty-five_plus'], label='65+', color='violet', lw = 3)

plt.xlabel('Number of Weeks')
plt.ylabel('Frequency of Deaths')
plt.title('Frequency of Deaths in 2016 by Age Group (ALL CITIES)')
plt.legend()
plt.grid(True, color='black')
plt.show()


# In[20]:


week_ending = ndf['week_ending'].tail(40)
pn_flu_deaths = ndf['Pn_Flu_Deaths'].tail(40)
fig,ax = plt.subplots(figsize=(16,9))

ax.barh(week_ending, pn_flu_deaths)
 

for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
 
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
ax.grid(True, color ='black',linestyle ='-.', linewidth = 0.5,alpha = 0.2)
 
# Show top values 
ax.invert_yaxis()
 
ax.set_title('Pneumonia and Influenza Deaths in Tacoma (2016)',loc ='left')

# Show Plot
plt.show()


# In[21]:


Tacoma_data = ndf[ndf['city'] == 'Tacoma']

plt.figure(figsize=(10, 6))
plt.scatter(Tacoma_data['year'], Tacoma_data['all_deaths'], color='hotpink', alpha=0.5)

plt.title('All Deaths in Tacoma (1962-2016)')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')

plt.grid(True)
plt.show()


# In[22]:


bridgeport_data = ndf[ndf['city'] == 'Bridgeport']

plt.figure(figsize=(10, 6))
plt.scatter(bridgeport_data['year'], bridgeport_data['all_deaths'], color='blue', alpha=0.5)

plt.title('All Deaths in Bridgeport (1962-2016)')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')

plt.grid(True)
plt.show()


# In[28]:


cambridge_data = ndf[ndf['city'] == 'Cambridge']

plt.figure(figsize=(10, 6))
plt.scatter(cambridge_data['year'], cambridge_data['all_deaths'], color='green', alpha=0.5)

plt.title('All Deaths in Cambridge (1962-2016)')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')

plt.grid(True)
plt.show()


# In[24]:


fr_data = ndf[ndf['city'] == 'Fall River']

plt.figure(figsize=(10, 6))
plt.scatter(fr_data['year'], fr_data['all_deaths'], color='salmon', alpha=0.5)

plt.title('All Deaths in Fall River (1962-2016)')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')

plt.grid(True)
plt.show()


# In[25]:


ny_data = ndf[ndf['city'] == 'New York']

plt.figure(figsize=(10, 6))
plt.scatter(ny_data['year'], ny_data['all_deaths'], color='gray', alpha=0.5)

plt.title('All Deaths in New York (1962-2016)')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')

plt.grid(True)
plt.show()


# In[26]:


h_data = ndf[ndf['city'] == 'Houston']

plt.figure(figsize=(10, 6))
plt.scatter(h_data['year'], h_data['all_deaths'], color='purple', alpha=0.5)

plt.title('All Deaths in Houston (1962-2016)')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')

plt.grid(True)
plt.show()


# In[27]:


austin_data = ndf[ndf['city'] == 'Austin']

plt.figure(figsize=(10, 6))
plt.scatter(austin_data['year'], austin_data['all_deaths'], color='tan', alpha=0.5)

plt.title('All Deaths in Austin (1962-2016)')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')

plt.grid(True)
plt.show()


# In[ ]:





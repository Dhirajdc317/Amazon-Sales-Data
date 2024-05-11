#!/usr/bin/env python
# coding: utf-8

# # Project title : Analysing Amazon Sales Data 

# 1. To analyze the Sales-trend -> month-wise, year-wise, yearly_month-wise

# 2. Find key metrics and factors and show the meaningful relationships between attributes.

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("Amazon Sales data.csv")


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


df.duplicated().sum()


# # TO PERFORM MONTH WISE, YEAR WISE, YEARLY_MONTH WISE SALES TREND

# In[13]:


# Extract date 
df["Order Date"]= pd.to_datetime(df["Order Date"])
df["Order Date"]


# In[15]:


# To extract year and month
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month


# In[16]:


df["Year"]


# In[17]:


df["Month"]


# In[19]:


df["Total Sales"] = df["Units Sold"]* df["Unit Price"]


# In[21]:


df["Total Sales"].sum()


# In[22]:


df.head()


# # Analyze sales trend

# In[24]:


Sales_Trends = df[['Order Date', 'Year', 'Month', 'Total Sales']]


# In[26]:


Sales_Trends.head(10)


# # Month wise sales trend

# In[27]:


monthwise_sales = Sales_Trends.groupby(['Month'])['Total Sales'].sum()
monthwise_sales


# In[33]:


plt.figure(figsize = (10,6))
monthwise_sales.plot(kind = 'bar', color = 'orange', edgecolor = 'black')
plt.title("Month Wise Sales Trends")
plt.xlabel('Months')
plt.ylabel('Sales')
plt.show()


# # Year wise sales trend

# In[34]:


yearwise_sales = Sales_Trends.groupby(['Year'])['Total Sales'].sum()
yearwise_sales


# In[36]:


plt.figure(figsize = (10,6))
yearwise_sales.plot(kind = 'bar', color = 'skyblue', edgecolor = 'black')
plt.title("Year Wise Sales Trends")
plt.xlabel('Years')
plt.ylabel('Sales')
plt.show()


# # Yearly_Month_wise Sales

# In[37]:


yearly_month_wise = Sales_Trends.groupby(['Year', 'Month'])['Total Sales'].sum()
yearly_month_wise


# In[38]:


sales_2010 = yearly_month_wise[yearly_month_wise.index.get_level_values('Year') == 2010]
sales_2011 = yearly_month_wise[yearly_month_wise.index.get_level_values('Year') == 2011]
sales_2012 = yearly_month_wise[yearly_month_wise.index.get_level_values('Year') == 2012]
sales_2013 = yearly_month_wise[yearly_month_wise.index.get_level_values('Year') == 2013]
sales_2014 = yearly_month_wise[yearly_month_wise.index.get_level_values('Year') == 2014]
sales_2015 = yearly_month_wise[yearly_month_wise.index.get_level_values('Year') == 2015]
sales_2016 = yearly_month_wise[yearly_month_wise.index.get_level_values('Year') == 2016]
sales_2017 = yearly_month_wise[yearly_month_wise.index.get_level_values('Year') == 2017]


# # plt.figure(figsize=(20, 10))
# 
# plt.subplot(2,4,1)
# plt.bar(sales_2010.index.get_level_values('Month'), sales_2010, color = 'red')
# plt.title('Monthly Sales for the Year 2010')
# plt.xlabel('Month')
# plt.ylabel('Sales')
# 
# plt.subplot(2,4,2)
# plt.bar(sales_2011.index.get_level_values('Month'), sales_2011, color = 'blue')
# plt.title('Monthly Sales for the Year 2011')
# plt.xlabel('Month')
# plt.ylabel('Sales')
# 
# plt.subplot(2,4,3)
# plt.bar(sales_2012.index.get_level_values('Month'), sales_2012, color = 'pink')
# plt.title('Monthly Sales for the Year 2012')
# plt.xlabel('Month')
# plt.ylabel('Sales')
# 
# plt.subplot(2,4,4)
# plt.bar(sales_2013.index.get_level_values('Month'), sales_2013, color = 'yellow')
# plt.title('Monthly Sales for the Year 2013')
# plt.xlabel('Month')
# plt.ylabel('Sales')
# 
# plt.subplot(2,4,5)
# plt.bar(sales_2014.index.get_level_values('Month'), sales_2014, color = 'skyblue')
# plt.title('Monthly Sales for the Year 2014')
# plt.xlabel('Month')
# plt.ylabel('Sales')
# 
# plt.subplot(2,4,6)
# plt.bar(sales_2015.index.get_level_values('Month'), sales_2015, color = 'violet')
# plt.title('Monthly Sales for the Year 2015')
# plt.xlabel('Month')
# plt.ylabel('Sales')
# 
# plt.subplot(2,4,7)
# plt.bar(sales_2016.index.get_level_values('Month'), sales_2016, color = 'purple')
# plt.title('Monthly Sales for the Year 2016')
# plt.xlabel('Month')
# plt.ylabel('Sales')
# 
# plt.subplot(2,4,8)
# plt.bar(sales_2017.index.get_level_values('Month'), sales_2017, color = 'grey')
# plt.title('Monthly Sales for the Year 2017')
# plt.xlabel('Month')
# plt.ylabel('Sales')
# 
# 
# plt.tight_layout(pad=3.0)
# plt.show()

# # Observations:
# 

# Each bar shows monthly sales per year

# # Visualise the relationship between attributes

# Create following plots

# 1. A scatter plot of Units sold and Total Revenue, Unit price and Total revenue, Unit price and total profit
# 2. A count plot of Item Type and channels

# In[46]:


# Relationship between Units sold and total revenue 
plt.figure(figsize=(8, 6))
plt.scatter(df['Units Sold'], df['Total Revenue'], color='purple',alpha = 0.8)
plt.xlabel('Units Sold')
plt.ylabel('Total Revenue')
plt.title('Relationship between Units Sold and Total Revenue')
plt.grid(True)
plt.tight_layout()
plt.show()


# # Observations 

# 1. We observe that there is a relationship between Units Sold and Total Revenue 
# 2. As Unit Sales increases, Revenue also increases

# In[47]:


# Relationship between Unit Price and Total Revenue

plt.figure(figsize = (10,6))
plt.scatter(df['Unit Price'], df['Total Revenue'], color = 'red', alpha = 0.7)
plt.title("Relationship between Unit price and Total Revenue")
plt.xlabel('Unit Price')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.tight_layout()
plt.show()


# # Observations

# 1. We observe that there is a relationship between unit price and total revenue
# 2. As per the observation higher unit prices lead to higher total revenue.

# In[48]:


# Relationship between Unit Price and Total Profit

plt.figure(figsize = (11,6))
plt.scatter(df['Unit Price'], df['Total Profit'], color = 'green', alpha = 0.7)
plt.title("Relationship between Unit Price and Total Profit ")
plt.xlabel('Unit Price')
plt.ylabel('Total Profit')
plt.grid(True)
plt.tight_layout()
plt.show()


# # Observations

# 1. We observe that there is relationship between Unit Price and total profit
# 2. As per the observation unit price increases, the profit also increase
# 3. The profit is maximun between 400 and 500 unit price

# In[53]:


# Count plot of the "Item Type" feature

plt.figure(figsize=(15, 6))
colors = sns.color_palette('husl', len(df['Item Type'].unique()))
sns.countplot(data=df, x='Item Type', edgecolor='linen', alpha=0.7, palette=colors)
plt.title("Count Plot of Item Type")
plt.xlabel('Item Type')
plt.ylabel('Count')
plt.show()


# # Observations

# We see that most customers choose clothes or cosmetics, having maximum purchases

# In[54]:


# Count plot of channel

plt.figure(figsize = (15,6))
colors = sns.color_palette('husl', len(df['Item Type'].unique()))

sns.countplot(data = df ,x = 'Sales Channel', edgecolor = 'linen', alpha = 0.7, palette = colors)

plt.title("Count Plot of Sales Channel")
plt.xlabel('Sales Channel')
plt.ylabel('Count')
plt.show()


# # Observations

# We have seen that online and offline both are equally distributed

# # Conclusion

# In[ ]:






# coding: utf-8

# # Analysing BOM Solar Exposure Data
# 

# This notebook imports, rearranges and plots solar exposure data provided by the BOM. The data is collected from the BOM website as a csv file, and consists of daily global solar exposure data. This is the total amount of solar energy hitting the surface of the ground at a weather station over a 24 hour period, in MegaJoules per square metre. More information can be found [here](http://www.bom.gov.au/climate/austmaps/solar-radiation-glossary.shtml#globalexposure).
# 
# The aim is to bring the data in, reformat it using Pandas DataFrames, and mathematically model the change in solar exposure throughout the year using SciPy.

# ## Read in the file, format the data
# We will read in the file using the pandas method read_csv, which will automatically create a DataFrame with the csv headers as column names.

# In[1]:


# Import modules
from math import ceil
import pandas # DataFrames
import matplotlib.pyplot as plt # Plotting
from scipy.optimize import curve_fit # Nonlinear regression
import numpy as np # Arrays
# Show plots inline in the notebook
get_ipython().magic('matplotlib inline')

# Read in the csv file
csv_name = 'Brisbane-Daily.csv'
df = pandas.read_csv(csv_name)
df.head()


# We have successfully ingested the data. We need to remove the first two columns, as they are irrelevant to us.

# In[2]:


# Remove the 'Product Code' and 'Station Number' columns
df.drop(df.columns[[0,1]], axis=1, inplace=True)
# Shorten the name of the last column
df.rename(columns={df.columns[-1]:'MJsqm'}, inplace=True)
df.head()


# We now need to pad the month and day columns, to make them equal width. This is because we will concatenate the month and day later, pre-empting a table pviot on year. Padding can be done easily with zfill(), a string method designed to pad with zeros.

# In[3]:


# Pad the months and days with zeros to make them all 2 chars wide
df['Month'] = df['Month'].apply(lambda x : str(x).zfill(2))
df['Day'] = df['Day'].apply(lambda x : str(x).zfill(2))
df.head()


# We now concatenate the month and day using string formatting. There should be 366 unique values in the 'Mon-Day' column.

# In[4]:


# Create a DateTime column - the concat of month and day
df['Mon-Day'] = df[['Month','Day']].apply(lambda x : '{}-{}'.format(x[0], x[1]), axis=1)
df.head()


# ## Pivot the table on year
# We will pivot based on year, to get one column per year, with one row for each day of the year. This will allow us to easily compute statistics over each row. Pandas DataFrames have a very easy to use pivot method.

# In[5]:


# Pivot the table to have years as columns
byYear = df.pivot(index='Mon-Day', columns='Year', values='MJsqm')
byYear.head()


# ## Calculate statistics over day
# We will keep it simple and calculate simple statistics for each day. We will get the mean, the extremes, and calculate simple 95% confidence intervals for each day, across years. Because there are only 17 years of data (and therefore only 17 records per day of the year), the data is quite rough. We will use mean for the upcoming mathematical regression.

# In[6]:


# Calculate statistics for each day
byYear['Mean'] = byYear.mean(axis=1)
byYear['Min'] = byYear.min(axis=1)
byYear['Max'] = byYear.max(axis=1)
byYear['Stdv'] = byYear.std(axis=1)
byYear['+2sd'] = byYear['Mean'] + 1.96*byYear['Stdv']
byYear['-2sd'] = byYear['Mean'] - 1.96*byYear['Stdv']
byYear.head()


# ## Plot the Data
# We will plot the mean of the data over the year. As expected, we get a periodic function. Again, as we only have a limited amount of data, the confidence intervals and the mean itself are still quite rough.

# In[7]:


# Set up the month ticks for the x axis
daysInMonth = [31,29,31,30,31,30,31,31,30,31,30,31]
xticks = [sum(daysInMonth[0:i]) for i in range(len(daysInMonth))]

# Plot the data
ax = byYear.plot(y='Mean', 
                 title='Daily Solar, 17 year mean, Bruce',
                 legend=True,
                 style='r-',
                 xticks=xticks,
                 ylim=(0, 5*round(float(byYear['+2sd'].max())/5))
                )
byYear.plot(y='+2sd', 
            style='b-',
            ax=ax,
            grid=True
           )
byYear.plot(y='-2sd', 
            style='b-',
            ax=ax,
            grid=True
           )
ax.set_xlabel('Date')
ax.set_ylabel('MJ/sqm')
setp = plt.setp(ax.xaxis.get_majorticklabels(),rotation=45)


# ## Non-linear regression
# The data is clearly a periodic function. We can use curve fitting methods (the most common implementation of which is least squares) from the scipy module to get the parameters of the periodic function.

# In[8]:


# Define the function to be optimised
# This function will be passed to the curve fitter
def nonlinear_func(x, a, b, c, d):
    return a*np.sin(b*x + c) + d


# When regressing, you must enter starting guesses for each of the parameters you wish to optimise. In this case, we can very easily approximate each parameter.

# In[9]:


# Initial estimates of parameters for regression

# Amplitude ~ half of max - min
a_guess = (byYear['Mean'].max() - byYear['Mean'].min())/2.0
# B = 2pi / period. Period ~ 366 days
b_guess = 2*np.pi / 366
# Phase Shift = pi/2 (starts at a maximum)
c_guess = np.pi/2
# Equilibrium ~ average of max, min
d_guess = (byYear['Mean'].max() + byYear['Mean'].min())/2.0

print("Initial guesses:\nA: {}\nB: {}\nC: {}\nD: {}".format(a_guess, b_guess, c_guess, d_guess))


# We now run the regression using the mean data.

# In[10]:


# Perform the regression using scipy
xdata = np.arange(1, 367)
ydata = byYear['Mean'].as_matrix()
popt, pcov = curve_fit(nonlinear_func, xdata, ydata, p0=(a_guess, b_guess, c_guess, d_guess))
print("Optimised Values:\nA: {}\nB: {}\nC: {}\nD: {}".format(*popt))


# ## Evaluate Model
# We can plot the models against each other in order to evaluate the goodness of fit.

# In[11]:


curvey = nonlinear_func(xdata, *popt)
byYear['fitcurve'] = curvey

# Plot the data

ax2 = byYear.plot(y='fitcurve', style='k-', linewidth=2)
byYear.plot(y='Mean',
            ax=ax2,
            title='Daily Solar, raw data vs regression',
            legend=True,
            grid=True,
            style='r-',
            xticks=xticks,
            ylim=(0, 5*ceil(float(byYear['Mean'].max())/5))
           )
ax2.set_xlabel('Date')
ax2.set_ylabel('MJ/sqm')
setp = plt.setp(ax2.xaxis.get_majorticklabels(),rotation=45)


# The regression was successful. The fit is great.
# The nature of the BOM data is such that (theoretically) this notebook can be used to analyse and regress any weather station's daily Global Solar Exposure.

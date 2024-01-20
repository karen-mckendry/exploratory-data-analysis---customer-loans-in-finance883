# Exploratory Data Analysis - Customer Loans in Finance

## Description

The aim of this project is to conduct analysis to aid decisions on loan approvals, interest rates and risk for a business. Steps include extracting a dataset on loans using a class, and conducting some exploratory data analysis with the help of 4 additional classes which clean, analyse, visualise and transform the data.

## Data Extraction

Initially I created a db_utils.py file to retrieve the data from an AWS RDS using sqlalchemy, then stored it as a csv file. Next I familiarised myself with the columns, looking at nulls, range of values, datatypes (considering whether they were appropriate), unique values for categorical columns, and had a look at one or two columns to check why values might be nulls. 

## Cleaning the data

The next task was to deal with missing values, dropping rows or columns as appropriate depending on the number of nulls, or imputing missing values with the median or mean. 

With nulls removed, I used the Plotter class to visualise skewed columns along with the df_stats method of the DataFrameInfo class to view their statistical info, and made a decision to transform them using either a log transform or a Yeo-Johnson transform.

Once the skewed columns were transformed, I looked at outliers in each column using both box plots and z-scores, and in each case decided to remove outliers either outside the upper/lower fences, or with a z-score over 3, depending on how much data would be lost.

Finally I produced a heatmap of the columns and dropped columns with a correlation over 0.9.

## Analysis and conclusions

To start off the analysis phase, I looked at the current position of the loans in terms of how much has been recovered and then calculated a projection of the position in 6 months' time. I looked at losses from charged off loans and calculated a projection of that loss over the outstanding months. I moved on to look at loans where the repayments were late, and calculated potential losses should those loans be charged off. Lastly I looked at each column in the dataset and compared statistical information and visualisations to find patterns by considering subsets of the data ie fully paid versus charged off and late versus charged off, and made conclusions.


### Classes

Brief overview of classes and methods:

class DataTransform:  
A class to transform datatypes in a DataFrame, with the following methods:  
    &emsp;transform_date(cols)  
    &emsp;transform_str(cols)  
    &emsp;transform_int(cols)  
    &emsp;transform_bool(cols)  
    &emsp;str_to_int(cols)  

class DataFrameInfo:  
A class to return information on a DataFrame, with methods:  
    &emsp;df_info():   
        &emsp;&emsp;&emsp;Displays column names, number of non-nulls, and datatype  
    &emsp;df_stats(cols):   
        &emsp;&emsp;&emsp;Calculates the mean, median, mode, standard deviation, variance, maximum and minimum for each column in the list.  
    &emsp;df_shape():   
        &emsp;&emsp;&emsp;Gives the number of rows and columns of the DataFrame.  
    &emsp;count_distinct():   
        &emsp;&emsp;&emsp;Prints number of unique values for each categorical column in the DataFrame.  
    &emsp;count_nulls(cols):  
        &emsp;&emsp;&emsp;Count number of nulls in each column passed to the method.  
    &emsp;list_unique_values():  
        &emsp;&emsp;&emsp;Lists unique values for each categorical column in the DataFrame.  
    &emsp;normal_test():  
        &emsp;&emsp;&emsp;D'Agostino's K^2 Test to test for normal distribution.  
    &emsp;crosstab_table(col1, col2):  
        &emsp;&emsp;&emsp;Creates two crosstab DataFrames of 2 categorical columns, one with the first column values adding to 100% and one with the  
        &emsp;&emsp;&emsp;second adding to 100%.  


class Plotter:  
A class to produce visualisations, with the following methods:  
    &emsp;qq_plot(col):   
        &emsp;&emsp;&emsp;Prints a scatter plot of actual column values (y) against theoretical values from normal distribution (x).  
    &emsp;log_transform(col):  
        &emsp;&emsp;&emsp;Plot of log of column values.  
    &emsp;box_cox_transform(col):  
        &emsp;&emsp;&emsp;Plot of the Box-Cox transform of column values.  
    &emsp;yeo_johnson(col):  
        &emsp;&emsp;&emsp;Plot of Yeo-Johnson transform of column values.  
    &emsp;box_and_whisker(col):  
        &emsp;&emsp;&emsp;Box and Whisker plot of column values.  
    &emsp;correlation_matrix(num_cols):  
        &emsp;&emsp;&emsp;Heatmap showing correlation between columns.  
    &emsp;regression_plot(col_1, col_2):  
        &emsp;&emsp;&emsp;Regression plot.  
    &emsp;scatter_plot(col_1, col_2):  
        &emsp;&emsp;&emsp;Scatter plot of col_1 against col_2.  
    &emsp;bar_chart(col, x_vals, y_vals):  
        &emsp;&emsp;&emsp;Bar chart of categorical column with percentage of each value  
    &emsp;pie_chart(col):  
        &emsp;&emsp;&emsp;A pie chart with percentage labels.  
    &emsp;bar_chart(col, x_vals, y_vals):  
        &emsp;&emsp;&emsp;A bar chart for use with crosstab_table method of DataFrameInfo.  
    &emsp;hist_kde(cols):  
        &emsp;&emsp;&emsp;Histogram with KDE curve.  

class DataFrameTransform:  
A class to transform data, with the following methods:  
    &emsp;impute_mean(cols_to_impute_mean):  
        &emsp;&emsp;&emsp;Replaces nulls with the mean of the column.  
    &emsp;impute_median(cols_to_impute_mean):  
        &emsp;&emsp;&emsp;Replaces nulls with the median of the column.  
    &emsp;apply_log_transforms(cols):  
        &emsp;&emsp;&emsp;Transform the columns by applying the log transform.  
    &emsp;apply_yeo_transforms(cols):  
        &emsp;&emsp;&emsp;Transform the columns by applying the Yeo-Johnson transform.  
     &emsp;calc_z_scores(col):  
        &emsp;&emsp;&emsp;Adds a column with the z-score for each value in column.  
     &emsp;calc_outliers_IQR(col):  
        &emsp;&emsp;&emsp;Calculate outliers beyond the lower and upper fences and returns rows from the DataFrame where those outliers occur.  
    &emsp;variation_inflation_factor():  
        &emsp;&emsp;&emsp;Calculates the r^2 and VIF for each column.  
    &emsp;remove_outliers(outliers):  
        &emsp;&emsp;&emsp;Removes outliers identified in either calc_outliers_IQR or calc_z_scores.  
    &emsp;future_position():  
        &emsp;&emsp;&emsp;Adds columns containing the oustanding principal for the next 6 months for a loans DataFrame.  
    &emsp;loss_by_month(start_month):  
        &emsp;&emsp;&emsp;Calculates the projected loss month by month for the outstanding months of a charged off loan.  


## Installation instructions

`git clone https://github.com/karen-mckendry/exploratory-data-analysis---customer-loans-in-finance883`


The following should be installed:  
pandas as pd  
numpy as np  
scipy import stats  
statsmodels.graphics.gofplots import qqplot  
statsmodels.formula.api as smf  
matplotlib.pyplot as plt  
seaborn as sns  
plotly.express as px  
numpy_financial as npf  

## Usage instructions

Enter `python project.ipynb` in your terminal.

## License information

This project is licensed under the GNU GPLv3 License.

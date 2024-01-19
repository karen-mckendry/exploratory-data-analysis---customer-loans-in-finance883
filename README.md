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
    transform_date(cols)  
    transform_str(cols)  
    transform_int(cols)  
    transform_bool(cols)  
    str_to_int(cols)  

class DataFrameInfo:  
A class to return information on a DataFrame, with methods:  
    df_info():   
        Displays column names, number of non-nulls, and datatype  
    df_stats(cols):   
        Calculates the mean, median, mode, standard deviation, variance, maximum and minimum for each column in the list.  
    df_shape():   
        Gives the number of rows and columns of the DataFrame.  
    count_distinct():   
        Prints number of unique values for each categorical column in the DataFrame.  
    count_nulls(cols):  
        Count number of nulls in each column passed to the method.  
    list_unique_values():  
        Lists unique values for each categorical column in the DataFrame.  
    normal_test(self):  
        D'Agostino's K^2 Test to test for normal distribution.  
    crosstab_table(col1, col2):
        Creates two crosstab DataFrames of 2 categorical columns, one with the first column values adding to 100% and one with the second adding to 100%.


class Plotter:  
A class to produce visualisations, with the following methods:  
    qq_plot(col):   
        Prints a scatter plot of actual column values (y) against theoretical values from normal distribution (x).  
    log_transform(col):  
        Plot of log of column values.  
    box_cox_transform(col):  
        Plot of the Box-Cox transform of column values.  
    yeo_johnson(col):  
        Plot of Yeo-Johnson transform of column values.  
    box_and_whisker(col):  
        Box and Whisker plot of column values.  
    correlation_matrix(num_cols):  
        Heatmap showing correlation between columns.  
    regression_plot(col_1, col_2):  
        Regression plot.  
    scatter_plot(col_1, col_2):  
        Scatter plot of col_1 against col_2.  
    bar_chart(col, x_vals, y_vals):  
        Bar chart of categorical column with percentage of each value  
    pie_chart(col):  
        A pie chart with percentage labels.  
    bar_chart(col, x_vals, y_vals):  
        A bar chart for use with crosstab_table method of DataFrameInfo.  

    hist_kde(cols):
        Histogram with KDE curve.

class DataFrameTransform:  
A class to transform data, with the following methods:  
    impute_mean(cols_to_impute_mean):  
        Replaces nulls with the mean of the column.  
    impute_median(cols_to_impute_mean):  
        Replaces nulls with the median of the column.  
    apply_log_transforms(cols):  
        Transform the columns by applying the log transform.  
    apply_yeo_transforms(cols):  
        Transform the columns by applying the Yeo-Johnson transform.  
     calc_z_scores(col):  
        Adds a column with the z-score for each value in column.  
     calc_outliers_IQR(col):  
        Calculate outliers beyond the lower and upper fences and returns rows from the DataFrame where those outliers occur.  
    variation_inflation_factor():  
        Calculates the r^2 and VIF for each column.  
    remove_outliers(outliers):  
        Removes outliers identified in either calc_outliers_IQR or calc_z_scores.  
           future_position():  
        Adds columns containing the oustanding principal for the next 6 months for a loans DataFrame.  
    loss_by_month(start_month):  
        Calculates the projected loss month by month for the outstanding months of a charged off loan.  


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
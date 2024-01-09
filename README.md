# Exploratory Data Analysis - Customer Loans in Finance

## Description

The aim of this project is to conduct analysis to aid decisions on loan approvals, interest rates and risk for a business. Steps include extracting a dataset on loans using a class I set up for this purpose, and conducting some exploratory data analysis with the help of 4 additional classes which clean, analyse, visualise and transform the data.

Initially I created a db_utils.py file to retrieve the data from an AWS RDS using sqlalchemy, then stored it as a csv file. Next I familiarised myself with the columns, looking at nulls, range of values, datatypes (considering whether they were appropriate), unique values for categorical columns, and had a look at one or two columns to check why values might be nulls. 

The next task was to deal with missing values, dropping rows or columns as appropriate depending on the number of nulls, or imputing missing values with the median or mean. 

With nulls removed, I used the Plotter class to visualise skewed columns along with the df_stats method of the DataFrameInfo class to view their statistical info, and made a decision to transform them using either a log transform or a Yeo-Johnson transform.

Once the skewed columns were transformed, I looked at outliers in each column using both box plots and z-scores, and in each case decided to remove outliers either outside the upper/lower fences, or with a z-score over 3, depending on how much data would be lost.

Finally I produced a heatmap of the columns and dropped columns with a correlation over 0.9.


## Classes

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

class Plotter:
A class to produce visualisations, with the following methods:
    qq_plot(col): 
        Prints a scatter plot of actual column values (y) against theoretical values from normal distribution (x)
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


## Installation instructions

`git clone https://github.com/karen-mckendry/exploratory-data-analysis---customer-loans-in-finance883`

The following should be installed:
pandas as pd
numpy as np
scipy.stats
statsmodels.graphics.gofplots.qqplot
statsmodels.formula.api as smf
matplotlib.pyplot as plt
seaborn as sns
plotly.express as px

## Usage instructions

Enter `python project.ipynb` in your terminal.

## License information

This project is licensed under the GNU GPLv3 License.
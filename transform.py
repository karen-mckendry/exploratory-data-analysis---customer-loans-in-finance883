import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 80)

class DataTransform:
    """
    A class to transform datatypes in a DataFrame.

    Attributes
    ----------
    df : the DataFrame whose datatypes are to be changed.

    Methods
    -------
    transform_date(date_cols): 
        Changes datatype from object to datetime.

    transform_str(str_cols):
        Changes datatype from object to string.
    
    transform_int(int_cols):
        Changes datatype from float to int.

    transform_bool(bool_cols):
        Changes datatype from object to bool.

    str_to_int(str_to_int_cols):
        Removes text following integer then change datatype to int.
    """

    def __init__(self, df):
        """ Assigns DataFrame as an attribute """
        self.df = df

    def transform_date(self, date_cols):
        """
        Changes datatype from object to datetime.

        Parameters
        ----------
        date_cols : list of columns whose datatypes are to be changed to datetime

        Returns
        -------
        Updated DataTransform object with column datatypes changed to datetime.
        """
        
        for date_col in date_cols:
            self.df[date_col] = pd.to_datetime(self.df[date_col], format="%b-%Y")
        return self
    
    def transform_str(self, str_cols):
        """
        Changes datatype from object to string.

        Parameters
        ----------
        str_cols : list of columns whose datatypes are to be changed to str.

        Returns
        -------
        Updated DataTransform object with column datatypes changed to str.
        """

        for str_col in str_cols:
            self.df[str_col] = self.df[str_col].astype('string')

    def transform_int(self, int_cols):
        """
        Changes datatype from float to integer.

        Parameters
        ----------
        int_cols : list of columns whose datatypes are to be changed to int.

        Returns
        -------
        Updated DataTransform object with column datatypes changed to int.
        """

        for int_col in int_cols:
            self.df[int_col] = self.df[int_col].apply(lambda x: x if pd.isnull(x) else int(x))
        return self

    def transform_bool(self, bool_cols):
        """
        Changes datatype from object to boolean.

        Parameters
        ----------
        bool_cols : list of columns whose datatypes are to be changed to bool.

        Returns
        -------
        Updated DataTransform object with column datatypes changed to bool.
        """
        for bool_col in bool_cols:
            self.df[bool_col] = self.df[bool_col].replace({'y': True, 'n': False})
        return self
    
    def str_to_int(self, str_to_int_cols):
        """
        Changes datatype from object to integer.

        Parameters
        ----------
        str_to_int_cols : list of columns whose datatypes are to be stripped of text and changed to int.

        Returns
        -------
        Updated DataTransform object with column datatypes changed to int.
        """
        for str_to_int_col in str_to_int_cols:
            self.df[str_to_int_col] = self.df[str_to_int_col].apply(lambda x: x.split()[0] if pd.isnull(x) == False else x)
        self.transform_int(str_to_int_cols)
        return self


class DataFrameInfo:
    """
    A class to return information on a DataFrame.

    Attributes
    ----------
    df : the DataFrame whose datatypes are to be changed.
    cat_cols : list of categorical columns in the DataFrame.
    num_cols : list of numerical columns in the DataFrame.

    Methods
    -------
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

    def normal_test(self):
        D'Agostino's K^2 Test to test for normal distribution.
    """

    def __init__(self, df):
        """ Assigns DataFrame as an attribute, and creates list of categorical and numerical columns """
        self.df = df
        self.cat_cols = self.df.select_dtypes(include=['object','string','bool']).columns
        self.num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
  
    def df_info(self):
       """ Displays column names, number of non-nulls, and datatype """
       print(self.df.info())
        
    def df_stats(self, cols):
        """
        Calculates the mean, median, mode, standard deviation, variance, maximum and minimum for each column in the list.

        Parameters
        ----------
        cols : list of columns whose data is to be summarised

        Returns
        -------
        DataFrame of these statistical measures.
        """
        num_cols = cols
        list(set(cols).intersection(set(self.num_cols)))
        stats_df = pd.DataFrame()
        for col in num_cols:
            col_stats = {
                'Column': col,
                'Mean': self.df[col].mean(),
                'Median': self.df[col].median(),
                'Mode': self.df[col].mode(),
                'Std Dev': self.df[col].std(),
                'Var': self.df[col].var(),
                'Max': self.df[col].max(), 
                'Min': self.df[col].min(),

            }
            col_stats_df = pd.DataFrame(col_stats)
            stats_df = pd.concat([stats_df, col_stats_df])
        return stats_df    
 
    def df_shape(self):
        print(f'\nData has {self.df.shape[0]} rows and {self.df.shape[1]} columns\n')
    
    def count_distinct(self):
       print(f'Categorical columns:\n{self.df[self.cat_cols].nunique()}')

    def count_nulls(self, cols):
        """
        Prints number of nulls in columns provided.

        Parameters
        ----------
        cols : list of columns in which to count nulls

        Returns
        -------
        None
        """
        col_width = max([len(col) for col in list(self.df[cols])])
        for col in cols:
            print(f'{col:<{col_width}}{round(self.df[col].isnull().sum(),2):>15}{round(self.df[col].isnull().sum()/len(self.df[cols])*100,2):>15} %')

    def list_unique_values(self):
        """
        Prints unique values in categorical columns.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for col in self.cat_cols:
            print(col, self.df[col].sort_values().unique())

    def normal_test(self):
        """
        D'Agostino's K^2 Test to test for normal distribution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        num_cols = set(self.df.select_dtypes(include=[np.number]).columns.tolist())
        cols_with_nulls = set(self.df[self.df.columns[self.df.isna().any()]]).intersection(num_cols)
        for col in cols_with_nulls:
            data = self.df[col]
            stat, p = stats.normaltest(data, nan_policy='omit')
            print('\n',col,': Statistics=%.3f, p=%.3f' % (stat, p))


class Plotter:
    """
    A class to produce visualisations.

    Attributes
    ----------
    df : the DataFrame

    Methods
    -------
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

    """
    
    def __init__(self, df):
        """ Assigns DataFrame as an attribute """
        self.df = df

    def col_hist(self, col):
        """
        Prints a histogram of column.

        Parameters
        ----------
        col : the column whose data is to be plotted.

        Returns
        -------
        None
        """
        self.df[col].hist(bins=40)
        print(self.df[col].skew())
    
    def qq_plot(self, col):
        """
        Prints a scatter plot of actual column values (y) against theoretical values from normal distribution (x)

        Parameters
        ----------
        col : the column whose data is to be plotted.

        Returns
        -------
        None
        """
        qqplot(self.df[col] , scale=1 ,line='q', fit=True)

    def log_transform(self, col):
        """
        Plot of log of column values.

        Parameters
        ----------
        col : the column whose data is to be plotted.

        Returns
        -------
        None
        """    
        log_col = self.df[col].map(lambda i: np.log(i) if i > 0 else 0)
        t=sns.histplot(log_col,label="Skewness: %.2f"%(log_col.skew()) )
        t.legend()

    def box_cox_transform(self, col):
        """
        Plot of the Box-Cox transform of column values.

        Parameters
        ----------
        col : the column whose data is to be plotted.

        Returns
        -------
        None
        """    
        boxcox_col = self.df[col]
        boxcox_col= stats.boxcox(boxcox_col)
        boxcox_col= pd.Series(boxcox_col[0])
        t=sns.histplot(boxcox_col,label="Skewness: %.2f"%(boxcox_col.skew()) )
        t.legend()

    def yeo_johnson (self, col):
        """
        Plot of Yeo-Johnson transform of column values.

        Parameters
        ----------
        col : the column whose data is to be plotted.

        Returns
        -------
        None
        """    
        yeojohnson_col = self.df[col]
        yeojohnson_col = stats.yeojohnson(yeojohnson_col)
        yeojohnson_col= pd.Series(yeojohnson_col[0])
        t=sns.histplot(yeojohnson_col,label="Skewness: %.2f"%(yeojohnson_col.skew()) )
        t.legend()
    
    def box_and_whisker(self, col):
        """
        Box and Whisker plot of column values.

        Parameters
        ----------
        col : the column whose data is to be plotted.

        Returns
        -------
        None
        """    
        fig = px.box(self.df, y=col)
        fig.show()

    def correlation_matrix(self, num_cols):
        """
        Heatmap showing correlation between columns.

        Parameters
        ----------
        cols : the columns whose data are to be compared for colinearity in the heatmap.

        Returns
        -------
        Correlation DataFrame containing correlation values between each pair of columns.
        """    
        # Compute the correlation matrix
        corr = self.df[num_cols].corr()
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True
        # set things up for plotting
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap
        sns.heatmap(corr, mask=mask, square=True, linewidths=.5, annot=False, cmap=cmap)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.show()
        return corr
    
    def regression_plot(self, col_1, col_2):
        """
        Regression plot of col_1 against col_2.

        Parameters
        ----------
        col_1, col_2 : the columns whose data are to be plotted.

        Returns
        -------
        None
        """    
        # fit a line to both distributions
        sns.regplot(x=self.df[col_1], y=self.df[col_2])
        # Compute the regression line parameters
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.df[col_1], self.df[col_2])
        # Predict the values using the regression line - just y = mx + c
        predicted_values = slope * self.df[col_1] + intercept
        # Compute the residuals
        residuals = self.df[col_2] - predicted_values
        # Compute the MSE
        mse = np.mean(residuals**2)
        # print the slope of the regression line
        print("slope of regression line: ", slope)
        # print the MSE of the regression line
        print("Mean Squared Error (MSE) of the regression line: ", mse)

    def scatter_plot(self, col_1, col_2):
        """
        Scatter plot of col_1 against col_2.

        Parameters
        ----------
        col_1, col_2 : the columns whose data are to be plotted.

        Returns
        -------
        None
        """
        sns.scatterplot(x = self.df[col_1], y = self.df[col_2])
    
   
class DataFrameTransform:
    """
    A class to transform data.

    Attributes
    ----------
    df : the DataFrame

    Methods
    -------
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
    """
    
    def __init__(self, df):
        """ Assigns DataFrame as an attribute """
        self.df = df
    
    def impute_mean(self, cols_to_impute_mean):
        """
        Replaces nulls with the mean of the column.

        Parameters
        ----------
        cols_to_impute_mean : the columns whose data are to have nulls imputed with the mean of the column.

        Returns
        -------
        DataFrameTransform
        """    
        for col in cols_to_impute_mean:
            self.df[col] = self.df[col].fillna(self.df[col].mean()) 
        return self

    def impute_median(self, cols_to_impute_median):
        """
        Replaces nulls with the median of the column.

        Parameters
        ----------
        cols_to_impute_median : the columns whose data are to have nulls imputed with the median of the column.

        Returns
        -------
        DataFrameTransform
        """    
        for col in cols_to_impute_median:
             self.df[col] = self.df[col].fillna(self.df[col].median()) 
        return self 
    
    def apply_log_transforms(self, cols):
        """
        Transform the columns by applying the log transform.

        Parameters
        ----------
        cols : the columns whose data are to have the log transform applied.

        Returns
        -------
        DataFrameTransform
        """    
        for col in cols:
            self.df[col] = self.df[col].map(lambda i: np.log(i) if i > 0 else 0)
        return self

    def apply_yeo_transforms(self, cols):
        """
        Transform the columns by applying the Yeo-Johnson transform.

        Parameters
        ----------
        cols : the columns whose data are to have the Yeo-Johnson transform applied.

        Returns
        -------
        DataFrameTransform
        """    
        for col in cols:
            yeojohnson_col = self.df[col]
            yeojohnson_col = stats.yeojohnson(yeojohnson_col)
            yeojohnson_col= pd.Series(yeojohnson_col[0])
            self.df[col] = yeojohnson_col
        return self

    def calc_z_scores(self, col):
        """
        Adds a column with the z-score for each value in column.

        Parameters
        ----------
        col : the column on which the z-scores are to be calculated.

        Returns
        -------
        DataFrameTransform
        """    
        #Calculate the z-scores for self.col
        mean_of_col = np.mean(self.df[col])
        std_of_col = np.std(self.df[col])
        self.df[col + '_z'] = (self.df[col] - mean_of_col) / std_of_col
        return self
    
    def calc_outliers_IQR(self, col):
        """
        Calculate outliers beyond the lower and upper fences and returns rows from the DataFrame where those outliers occur.

        Parameters
        ----------
        col : the column on which outliers are to be calculated.

        Returns DataFrame of outliers in column passed into method.
        -------
        DataFrameTransform
        """    
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask_1 = (self.df[col] < (Q1 - 1.5 * IQR))
        outlier_mask_2 = (self.df[col] > (Q3 + 1.5 * IQR))
        outliers = self.df[outlier_mask_1 | outlier_mask_2]
        return outliers
    
    def variation_inflation_factor(self):
        """
        Calculates the r^2 and VIF for each column.

        Parameters
        ----------
        None

        Returns
        -------
        DataFrame with r^2 and VIF values.
        """    
        cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cols.remove('id')
        vif_df = pd.DataFrame(columns = cols)
        
        for col in cols:
            model_col= col
            exog_cols = cols
            exog_cols.remove(model_col)
            
            #construct string to pass into model
            model_str = model_col + ' ~ '
            for i in range(len(exog_cols)):
                model_str = model_str + exog_cols[i] + ' + '
            model_str = model_str[:-3]
                
            model = smf.ols(model_str, self.df).fit()
            r2 = model.rsquared
            vif_df[col][0] = r2
            vif = 0
            if r2 != 1:
                vif = 1/(1-r2)
                vif_df[col][1] = vif
            print(col, r2, vif)
            return vif_df
      
    def remove_outliers(self, outliers):
        """
        Removes outliers identified in either calc_outliers_IQR or calc_z_scores.

        Parameters
        ----------
        outliers : DataFrame with only those rows where the column contains outliers.

        Returns
        -------
        DataFrameTransform
        """    
        self.df = self.df[~self.df.isin(outliers)].dropna(how='all')
        return self   
    

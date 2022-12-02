# -*- coding: utf-8 -*-

'''
Author: Ellie Biessek

This script  has various tools used for data exploration
'''

from scipy.stats.contingency import association
import pandas as pd
import numpy as np
from numpy import eye
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

class Matrix:
    
    def __init__(self, dataframe, copy = True):
        '''
        Class initialization, it serves as a base class for all the metrics
        Parameters
        ----------
        dataframe : pandas.DataFrame
            Pandas dataframe containing the variables
            of interest to measure the degree of association.
        Returns
        -------
        None.
        '''
        if isinstance(dataframe, pd.DataFrame):
        
            if copy:
                self.data = dataframe.copy()
            else:
                self.data = dataframe
            self.data = self.data.apply(lambda x: x.astype("category") if x.dtype == "O" else x)    
                
        else:
            raise TypeError("dataframe must be an instance of a pd.DataFrame")

class AssociationMatrix(Matrix):
    '''
    A dataframe like object that contains all calculated associations
    between all features(columns) of a given data frame
    It works for both numerical and categorical type of data. 
    Following measurements of correlation/association have been used:
        
    - Pearson's R        for numerical-numerical cases         (Range -1:1)
    - Correlation Ratio  for categorical-numerical cases       (Range  0:1)
    - Cramer's V         for categorical-categorical cases     (Range  0:1)       
   
    This code has been created by adjusting the code found in the link below 
    https://github.com/HeberTU/association_metrics/blob/main/association_metrics/categorical.py  
    '''   
    
    
    def __init__(self, dataframe):
        Matrix.__init__(self, dataframe)
        self.matrix = None
    
    def select_categorical(self):
        '''
        Selects all category variables

        Returns:
            None
        '''
        self.cat_columns = self.data.select_dtypes(
            include=['category']).columns
            
    def select_numerical(self):
        '''
        Selects all category variables

        Returns:
            None
        '''
        self.num_columns = self.data.select_dtypes(
            exclude =['category']).columns    
        
    def columns(self):
        '''
        Selects all category variables

        Returns:
            None
        '''
        self.columns = self.data.columns    

    def initialise_matrix(self):
        '''
        init a square matrix n x n fill with zeros,
        where n is the total number of columns
        found in the pd.DataFrame
        
        Returns:
            None
        '''
        n = len(self.columns)
        self.matrix = pd.DataFrame(
            eye(n),
            columns=self.columns,
            index=self.columns
        )


    def fill_in_associations(self):
        '''
        fills the square matrix with chosen association method
        
        - Pearson's R for continuous-continuous cases
        - Correlation Ratio for categorical-continuous cases
        - Cramer's V for categorical-categorical cases
        
        Returns:
            None
        '''
        
        all_combinations = combinations(self.columns, r=2)

        for comb in all_combinations:
            i = comb[0]
            j = comb[1]   
            
            if (i in self.cat_columns and j in self.cat_columns):   # cat_cat (Cramer's V)
                input_tab = pd.crosstab(self.data[i], self.data[j])
                res = association(input_tab, method ='cramer')
                
            elif (i in self.num_columns and j in self.num_columns): # num_num (Pearson's R)
                res = self.data[i].corr(self.data[j])
                
            else:
                if (i in self.cat_columns and j in self.num_columns): # cat_num (Correlation Ratio)
                    categories = np.array(self.data[i])
                    values = np.array(self.data[j])
                elif (i in self.num_columns and j in self.cat_columns): # num_cat                   
                    categories = np.array(self.data[j])
                    values = np.array(self.data[i])  
                ssw = 0
                ssb = 0
                for category in set(categories):
                    subgroup = values[np.where(categories == category)[0]]
                    ssw += sum((subgroup-np.mean(subgroup))**2)
                    ssb += len(subgroup)*(np.mean(subgroup)-np.mean(values))**2
                res = (ssb / (ssb + ssw))**.5
                
            self.matrix[i][j], self.matrix[j][i] = res, res  

    def fit(self):
        '''
        Creates a matrix filled with calculated associations,
        where columns and index are the categorical
        variables of the passed pandas.DataFrame
        
        Returns
            Association Matrix.
        '''
        self.select_categorical()
        self.select_numerical()
        self.columns()
        self.initialise_matrix()
        self.fill_in_associations()
        
        return self.matrix

def show_hitmap(df, **kwargs):
        '''
        creates pyplot object to visualise hitmap
        Because of the fact that different ranges are being used:
        Pearson (-1 : 1),  Cramer's & Correlation Ratio (0, 1)
        the hitmap has been designed in such a way that should help with result interpretation:
       
        - Strong green colour means strong correlation/association
        - light green colour means weak correlation/association
        
        Args:
            df - dataframe
            ** kwargs
            
        Returns
            None
        '''
        sns.set(font_scale=2)
        plt.figure(figsize=(20,20))#kwargs.get('figsize',None))
        diverging_palette = sns.diverging_palette(150, 160, as_cmap=True)
        sns.heatmap(df, annot=kwargs.get('annot',True), fmt=kwargs.get('fmt','.2f'), annot_kws={"fontsize":14}, cmap=diverging_palette)
        plt.show()    


def summary_table(df):
    '''
    Creates a summary info table about given dataframe
    
    Args:
        df: dataframe
        
    Returns:
        summary: info data frame
    '''
    summary = pd.DataFrame(df.dtypes, columns = ['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    return summary

def get_cat_features(df):
    '''
    Returns a list of column names that contain categorical type of data 
    
    Args:
        df: dataframe
        
    Returns:
        catcol: list of categorical columns names
    '''    
    
    catcols = []
    df = df.apply(lambda x: x.astype("category") if x.dtype == "O" else x)
    
    for feature in df.columns:
        if df[feature].dtype == 'category':
            catcols.append(feature)
    return catcols

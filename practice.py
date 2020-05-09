# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:15:01 2020

@author: jfran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


w_d = 'D:/Dropbox/Fimee/7mo semestre/Mineria de Datos/Practica Individual/Data/'
i_f = w_d + 'survey_results_public.csv'
data = pd.read_csv(i_f, encoding = 'utf-8')
column = 'ConvertedComp'
data.head()


"""
****************************
*       Help Functions     *
****************************
"""

def sub_string(string, sub_string):
    return sub_string in string
    
def sub_string_tokens(string, sub_string):
    sub_string = string.replace(r'++', r'\+\+')
    reg = re.compile('(;|^)'+sub_string+'(;|$)')
    return bool(reg.findall(string))

def uniques(col:pd.Series):
    lista = list(col.unique())
    lista = ';'.join(lista).split(';')
    return list(set(lista))

def filter_not_nulls(data, *cols):
    filt = data[cols[0]].notnull()
    for col in cols[1:]:
        filt = filt & data[col].notnull()
    return data[filt]

def five_numbers_summary(col):
    minimo = col.min()
    maximo = col.max()
    q1   = col.quantile(0.25)
    median   = col.quantile(0.5)
    q3   = col.quantile(0.75)
    return minimo, maximo, q1, median, q3

def std_mean(col):
    return col.std(),col.mean()

def histogram(data, col_a, col_b, nrows=1, ncols=None, xlabel=None, ylabel=None):
        new_data = filter_not_nulls(data, col_b, col_a)
        uniques_ = uniques(new_data[col_b])
        if not ncols:
            ncols = len(uniques_)
        if not ylabel is None:
            ylabel = "Amount"
        if not xlabel:
            xlabel = col_a
        for i, unique in enumerate(uniques_):
            filter_data = new_data[col_b].apply(sub_string, args=(unique,))
            f_data = new_data[filter_data]
            plt.subplot(nrows,ncols,i+1)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.title(f'{unique[:10]}')
            plt.hist(f_data[col_a], bins=10)
            plt.tight_layout()
        plt.show()

edlevels ={
     'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 2,
     'Professional degree (JD, MD, etc.)': 4,
     'Associate degree': 3,
     'Primary/elementary school': 1,
     'Bachelor’s degree (BA, BS, B.Eng., etc.)':4,
     'Some college/university study without earning a degree': 5,
     'Other doctoral degree (Ph.D, Ed.D., etc.)': 7,
     'I never completed any formal education':0,
     'Master’s degree (MA, MS, M.Eng., MBA, etc.)': 6
 }

def edlevel(item):
    return edlevels[item]


"""
****************************
*   Functions Problems     *
****************************
"""

def p1():   #5 number summary, the boxplot, the mean, and the standard deviation for the annual salary per gender.
    print('Problema #1')
    new_data = filter_not_nulls(data, 'Gender', 'ConvertedComp') 
    uniques_ = uniques(new_data['Gender']) 

    values = '|Min = {0}, Max = {1}, Q1 = {2}, Median = {3}, Q3 = {4}, standard deviation = {5}, Mean = {6}|'
    i = 1
    plt.figure(figsize=(8, 8), dpi=80, facecolor='w')
    for i, uni in enumerate(uniques_):
        filter_data = new_data['Gender'].apply(sub_string, args=(uni,))
        g_data = new_data[filter_data]
        print(uni[:10], values.format(*five_numbers_summary(g_data['ConvertedComp']), *std_mean(g_data['ConvertedComp'])),'\n')
        plt.subplot(3, 3, i+1)
        plt.boxplot(g_data['ConvertedComp'], sym='')
        plt.ylabel('Amount')
        plt.xlabel(uni[:10])
        plt.title(f'{uni[:10]}')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
        plt.tight_layout()
    plt.show()
    plt.clf()
        
        
def p2():   #5 number summary, the boxplot, the mean, and the standard deviation for the annual salary per ethnicity.
    print('\n Problema #2')
    new_data = filter_not_nulls(data, 'Ethnicity', 'ConvertedComp') 
    uniques_ = uniques(new_data['Ethnicity']) 

    values = '|Min = {0}, Max = {1}, Q1 = {2}, Median = {3}, Q3 = {4}, standard deviation = {5}, Mean = {6}|'
    i = 1
    plt.figure(figsize=(8, 8), dpi=80, facecolor='w')
    for i, uni in enumerate(uniques_):
        filter_data = new_data['Ethnicity'].apply(sub_string, args=(uni,))
        g_data = new_data[filter_data]
        print(uni[:10], values.format(*five_numbers_summary(g_data['ConvertedComp']), *std_mean(g_data['ConvertedComp'])),'\n')
        plt.subplot(3, 3, i+1)
        plt.boxplot(g_data['ConvertedComp'], sym='')
        plt.ylabel('Amount')
        plt.xlabel(uni[:10])
        plt.title(f'{uni[:10]}')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
        plt.tight_layout()
    plt.show()
    plt.clf()
    
def p3(): #5 number summary, the boxplot, the mean, and the standard deviation for the annual salary per developer type.
    print('\n Problema #3')
    new_data = filter_not_nulls(data, 'DevType', 'ConvertedComp') 
    uniques_ = uniques(new_data['DevType']) 

    values = '|Min = {0}, Max = {1}, Q1 = {2}, Median = {3}, Q3 = {4}, standard deviation = {5}, Mean = {6}|'
    i = 1
    plt.figure(figsize=(14, 8), dpi=80, facecolor='w')
    for i, uni in enumerate(uniques_):
        filter_data = new_data['DevType'].apply(sub_string, args=(uni,))
        g_data = new_data[filter_data]
        print(uni[:10], values.format(*five_numbers_summary(g_data['ConvertedComp']), *std_mean(g_data['ConvertedComp'])),'\n')
        plt.subplot(5, 5, i+1)
        plt.boxplot(g_data['ConvertedComp'], sym='')
        plt.ylabel('Amount')
        plt.xlabel(uni[:10])
        plt.title(f'{uni[:10]}')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
        plt.tight_layout()
    plt.show()
    plt.clf()
    

def p4():   #Compute the median, mean and standard deviation of the annual salary per country.
    print('\n Problema #4')
    values = '{0}: Mean = {1}, Median = {2}, standard deviation = {3}'
    new_data = filter_not_nulls(data, 'Country', 'ConvertedComp') 
    countries = uniques(new_data['Country'])
    for country in countries:
        filter_data = new_data['Country'] == country  
        f_data = new_data[filter_data]                 
        print(values.format(country, f_data['ConvertedComp'].mean(), f_data['ConvertedComp'].median(), f_data['ConvertedComp'].std()),'\n')   


def p5():   #Obtain a bar plot with the frequencies of responses for each developer type.
    print('\n Problema #5')
    new_data   = filter_not_nulls(data, 'DevType') 
    devtypes = uniques(new_data['DevType'])
    freqs = {} 
    for devtype in devtypes:
        freq = sum(new_data['DevType'].apply(sub_string, args=(devtype,)))
        freqs[devtype] = freq
    x = np.arange(len(freqs))
    plt.figure(figsize=(12,10), facecolor='w')
    plt.bar(height=freqs.values(), x=x)
    plt.xticks(x, freqs.keys(), rotation=90)


def p6():   #Plot histograms with 10 bins for the years of experience with coding per gender.
    print('\n Problema #6')
    data['YearsCode'].replace('Less than 1 year', '0.5', inplace=True)
    data['YearsCode'].replace('More than 50 years', '51', inplace=True)
    data['YearsCode'] =  data['YearsCode'].astype('float64')
    plt.figure(figsize=(12,16), facecolor='w')
    histogram(data, 'YearsCode', 'Gender', xlabel='Experience', ylabel='',nrows=8, ncols=4)


def p7():   #Plot histograms with 10 bins for the average number of working hours per week, perdeveloper type.
    print('\n Problema #7')
    new_data = filter_not_nulls(data, 'WorkWeekHrs', 'DevType') 
    filter_data = (new_data['WorkWeekHrs'] < 84) & (new_data['WorkWeekHrs'] > 21) 
    new_data = new_data[filter_data]  
    plt.figure(figsize=(8,16), facecolor = 'w')
    histogram(data, 'WorkWeekHrs', 'DevType', nrows=8, ncols=3, xlabel='WorkWeekHrs', ylabel='')


def p8():   #Plot histograms with 10 bins for the age per gender.
    print('\n Problema #8')
    new_data = filter_not_nulls(data, 'Age', 'Gender') 
    filter_data = (new_data['Age'] < 90) & (new_data['Age'] > 0) 
    new_data = new_data[filter_data]  
    plt.figure(figsize=(10, 12), facecolor='w')
    histogram(new_data,'Age','Gender', xlabel='Age', ylabel='', nrows=4, ncols=3)


def p9(): #Compute the median, mean and standard deviation of the age per programming language.
    print('\n Problema #9')
    data['LanguageWorkedWith'].replace('')
    new_data = filter_not_nulls(data, 'Age', 'LanguageWorkedWith') 
    languages = uniques(new_data['LanguageWorkedWith'])
    values = '{0}: Mean = {1}, Median = {2}, standard deviation = {3}'
    for language in languages:
        filter_data = new_data['LanguageWorkedWith'].apply(sub_string, args=(language,))
        f_data = new_data[filter_data]['Age']  
        print(values.format(language, f_data.mean(), f_data.median(), f_data.std()), '\n')


def p10(): # Compute the correlation between years of experience and annual salary.
    print('\n Problema #10')
    new_data = filter_not_nulls(data, 'ConvertedComp', 'YearsCode')
    x = new_data['ConvertedComp'].to_numpy()
    y = new_data['YearsCode'].to_numpy()
    corr = np.corrcoef(x=x, y=y)
    print(corr)
    plt.figure(figsize=(9,9), facecolor='w')
    plt.scatter(x=x, y=y)
    plt.title('Salary and Years of experience')
    plt.xlabel('salary')
#    plt.yticks(list(edlevels.values()), list(edlevels.keys()), rotation=90)
    plt.ylabel('experience')
    

def p11(): #Compute the correlation between the age and the annual salary.
    print('\n Problema #11')
    new_data = filter_not_nulls(data, 'ConvertedComp', 'Age')
    x = new_data['ConvertedComp'].to_numpy()
    y = new_data['Age'].to_numpy()
    corr = np.corrcoef(x=x, y=y)
    print(corr)
    plt.figure(figsize=(9,9), facecolor='w')
    plt.scatter(x=x, y=y)
    plt.title('Salary and Age correlation')
    plt.xlabel('salary')
#    plt.yticks(list(edlevels.values()), list(edlevels.keys()), rotation=90)
    plt.ylabel('experience')


def p12():  #Compute the correlation between educational level and annual salary.
    print('\n Problema #12')
    new_data = filter_not_nulls(data, 'EdLevel', 'ConvertedComp')
    x = new_data['EdLevel'].apply(edlevel)
    y = new_data['ConvertedComp'].to_numpy()
    corr = np.corrcoef(x=x, y=y)
    print(corr)
    plt.figure(figsize=(9,9), facecolor='w')
    plt.scatter(x=x, y=y)
    plt.title('Salary and education level')
    plt.xlabel('Education level')
    plt.xticks(list(edlevels.values()), list(edlevels.keys()), rotation=90)
    plt.ylabel('Salary')
    

def p13():  #Obtain a bar plot with the frequencies of the different programming languages.
    print('\n Problema #13')
    new_data = filter_not_nulls(data,'LanguageWorkedWith')
    devtypes = uniques(new_data['LanguageWorkedWith'])
    freqs={}
    for devtype in devtypes:
        freq = sum(new_data['LanguageWorkedWith'].apply(sub_string,args=(devtype,)))
        freqs[devtype] = freq
    freqs = sorted(freqs.items(), key= lambda item: item[1], reverse=True)
    freqs = {k:v for k, v in freqs}
    x = np.arange(len(freqs))
    height = np.arange(len(freqs))
    plt.figure(figsize=(12,10))
    plt.bar(height=list(freqs.values()), x = height)
    plt.xticks(x, freqs.keys(), rotation=90)



def main():
    p1()
    p2()
    p3()
    p4()
    p5()
    p6()
    p7()
    p8()
    p9()
    p10()
    p11()
    p12()
    p13()

main()


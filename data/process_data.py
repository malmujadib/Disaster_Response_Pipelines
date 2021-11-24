import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load_data function: It loads data from csv files, by using csv files path as parmater, 
    Input: csv files path
    Output: merged dataframe from two dataframes, one for messages, the other one for categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on = 'id')
    
    return df
    


def clean_data(df):

    """
    clean_data function makes preprocessing in the dataframe to make it more interpretable to Machine Learning Model
    Input: Uncleand dataframe
    Output: Cleaned dataframe 
    """
    categories = df['categories'].str.split(';', expand=True)
    categories.rename(columns=categories.iloc[0], inplace = True)
    categories.rename(columns = lambda x : str(x)[:-2], inplace = True)
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x[-1])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop('categories', inplace = True, axis = 1)
    
    df = pd.concat([df,categories],sort=True, axis = 1)
    
    df.loc[df['related'] > 1,'related'] = 0
    
    df.dropna(inplace = True)
    
    df.drop_duplicates(inplace = True)

    
def save_data(df, database_filename):
    """
    save_data function, it saves the dataframe as (database)
    Input: dataframe, and the database path
    Output: database converted from the dataframe
    """
    engine = create_engine('sqlite:///' + database_filename + '.db')
    df.to_sql('DisasterResponse', engine, index=False, if_exists = 'replace')  
    engine.dispose()


def main():
    """
    main function triggers all functions to do it function
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename + '.db')
    df.to_sql('table1', engine, index=False, if_exists = 'replace')
    engine.dispose()  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
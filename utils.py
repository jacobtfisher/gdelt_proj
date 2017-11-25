'''
Frederic R. Hopp 
Media Neuroscience Lab
Oct 2017
'''

import pickle
import pandas as pd
from sklearn.preprocessing import Imputer

CODEBOOK = 'GCAM-MASTER-CODEBOOK.csv'
THEMES = 'gdelt_themes.csv'
#PATHOGENS = '/srv/lab/gdelt/pathogen/pathogen_data.csv'
TERROR = '''TODO'''
#EVENTS = '/srv/lab/gdelt/events/'
#MENTIONS = '/srv/lab/gdelt/mentions/'
#FOP = '/home/fhopp/gdelt/datasets/press_freedom/fotp_1980_2017.csv'

def preprocess(df): # Pass in a single dataframe, best for single year 
    
    '''Drop duplicates, rename columns, 
    set index to date'''
    
    print('starting preprocessing, this may take some minutes')
    
    df = df.apply(pd.to_numeric, errors = 'ignore')
    df = df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
    df = df.loc[df.index.dropna()]
    df = df.dropna(axis=1, how='all')
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
 

    print('dropping successful, moving on...')
    
    gkg_ids = df.index.values
    df['date'] = gkg_ids
    df['date'] = df['date'].map(lambda x: x[:14])
    df['date'] = pd.to_datetime(df['date'], format = '%Y%m%d%H%M%S')
    df = df.set_index(df['date'], drop = False)

    print("reindexing successfull, continuing...")

    '''Rename variables'''
    gcam = pd.read_csv(CODEBOOK)
    
    mft_vars = ['c25.{}'.format(i) for i in range(1,12)]
    mft_names = ['Care','Harm','Fairness','Cheating','Loyalty','Betrayal','Authority','Subversion','Purity','Degradation', 'MoralityGeneral']
    mft_ind = ['Care','Harm','Fairness','Cheating']
    mft_bind = ['Loyalty','Betrayal','Authority','Subversion','Purity','Degradation']
    newcols = dict(zip(mft_vars, mft_names))
    df.rename(columns=newcols, inplace=True)

    liwc_vars = ['c5.{}'.format(i) for i in range(1,63)]
    liwc_names = gcam[gcam.Variable.isin(liwc_vars)].DimensionHumanName.values
    newcols = dict(zip(liwc_vars, liwc_names))
    df.rename(columns=newcols, inplace=True)
    
    wordnet_vars = ['c14.{}'.format(i) for i in range(1,12)]
    wordnet_delete = ['c14.{}'.format(i) for i in range(12,281)]
    df.drop(wordnet_delete, axis=1, inplace=True)
    wordnet_names = gcam[gcam.Variable.isin(wordnet_vars)].DimensionHumanName.values
    newcols = dict(zip(wordnet_vars, wordnet_names))
    df.rename(columns=newcols, inplace=True)

    df['source_location'] = df['source_location'].map({'US': 'United States', 'AU': 'Austria', 'UK': 'United Kingdom', 'RP': 'Philippines', 'IN': 'India', 'AS': 'Australia',
                                                       'SF': 'South Africa','NZ':'New Zealand', 'CA':'Canada','CH':'China PRC','TW':'Taiwan','KE':'Kenya', 'EI': 'Ireland',
                                                       'GM':'Germany','EG':'Egypt','NP': 'Nepal','IL':'Israel','AR': 'Argentina','JA':'Japan','KS':'South Korea','SP':'Spain' ,
                                                       'SZ':'Switzerland','IT':'Italy','FR':'France','NL': 'Netherlands'})
    
    print("renaming successfull, continuing...")

    '''Type conversions'''
    df['themes'] = df['themes'].astype(str)
    df['entities'] = df['entities'].astype(str)

    '''Computations'''
    '''MFT'''
    
    df['mft_sum'] = df[mft_names].sum(axis=1)
    df['mft_sum_percent'] = df[mft_names].sum(axis=1) / df.wordcount
    
    df['mft_ind_sum'] = df[mft_ind].sum(axis=1)
    df['mft_ind_sum_percent'] = df[mft_ind].sum(axis=1) / df.wordcount
    
    df['mft_bind_sum'] = df[mft_bind].sum(axis=1)
    df['mft_bind_sum'] = df[mft_bind].sum(axis=1) / df.wordcount

    df['care_harm'] = df[['Care','Harm']].sum(axis=1)
    df['fairness_cheating'] = df[['Fairness', 'Cheating']].sum(axis=1)
    df['loyalty_betrayal'] = df[['Loyalty','Betrayal']].sum(axis=1)
    df['authority_subversion'] = df[['Authority', 'Subversion']].sum(axis=1)
    df['purity_degradation'] = df[['Purity', 'Degradation']].sum(axis=1)

    df['care-valence'] = df.Care - df.Harm
    df['fairness-valence'] = df.Fairness - df.Cheating
    df['loyalty-valence'] = df.Loyalty - df.Betrayal 
    df['authority-valence'] = df.Authority - df.Subversion
    df['purity-valence'] = df.Purity - df.Degradation

    print('preprocessing successfull, happy analyzing.')
    
    return df

def clean_themes(df):
    
    '''cleans and removes unwanted themes from dataframe'''
    
    gdelt_themes = pd.read_csv(THEMES, delimiter=',')
    themes = list(pd.Series(gdelt_themes.Name))
   
    df.themes = df.themes.apply(lambda x: x.replace("{",''))
    df.themes = df.themes.apply(lambda x: x.replace("}",''))
    df.themes = df.themes.apply(lambda x: x.replace("None",''))
    df.themes = df.themes.apply(lambda x: x.replace("\'",''))
    df.themes = df.themes.apply(lambda x: x.replace("\''",''))
    df.themes = df.themes.apply(lambda x: x.replace(" ",''))
    df.themes = df.themes.apply(lambda x: x.strip())
    df.themes = df.themes.apply(lambda x: x.split(","))
    df['themes'] = df['themes'].apply(lambda x : [w for w in x if w in themes])
    df = df[df['themes'].map(len) > 1]
    
    df['themes'] = df['themes'].astype(str)
    
    '''
    for theme in themes:
        df[theme] = df.themes.str.count(theme)
    
    '''
    return df                              
    
def imputation(df):
    
    df = df.apply(pd.to_numeric, errors='ignore')
    df_numeric = df.select_dtypes(include=['float64']) # only select float types
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0) #axis=0 will take the column mean
    imr = imr.fit(df.select_dtypes(include=['float64'])) # fit learns the parameters
    imputed_data = imr.transform(df.select_dtypes(include=['float64']).values) # uses the learned parameters to transform thedata

    df_imputed = pd.DataFrame(imputed_data, index=df_numeric.index, columns=df_numeric.columns)
    #df_string = df.select_dtypes(include=['object']) #omit strings for now (entities, themes, country)
    df_int = df.select_dtypes(include=['int64'])
    
    df = pd.concat([df_imputed, df_int], axis=1)
    
    return df
    
    
def load_pathogens(PATHOGENS):
    df_pat = pd.DataFrame.from_csv(PATHOGENS)
    df_pat = df_pat.apply(pd.to_numeric, errors='coerce')
   
    return df_pat

#TODO: Preprocess events and mentions, get the sharing data 

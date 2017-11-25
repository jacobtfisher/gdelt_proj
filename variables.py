
'''Defines variables'''

import pandas as pd
import re 

'''Codebooks and Datasets'''

CODEBOOK = 'GCAM-MASTER-CODEBOOK.csv'
THEME_CODES = 'gdelt_themes.csv'
gdelt_themes = pd.read_csv(THEME_CODES, delimiter=',')
themes = list(pd.Series(gdelt_themes.Name))

#CENTROIDS = '/home/fhopp/gdelt/datasets/country_centroids_all.csv'

gcam = pd.read_csv(CODEBOOK)
#df_centroids = pd.read_csv(CENTROIDS, sep='\t', error_bad_lines=False)
#fips = set(df_centroids.FIPS10.values)


'''GDELT Specific Definitions'''

THEME_LIST = ['AFFECT', 'ARMEDCONFLICT','CYBER_ATTACK','EXTREMISM','FREESPEECH', 'IMMIGRATION', 'JIHAD', 'KILL','LEGISLATION',
                                     'LGBT','MOVEMENT_SOCIAL','PROTEST','REBELLION', 'RELIGION', 'SOVEREIGNTY', 'SECURITY_SERVICES', 'TERROR', 'UNEMPLOYMENT', 'WOUND']
 

ENTITY_LIST = ['donald trump', 'trump', 'donald j. trump', 'ted cruz', 'hillary clinton', 'clinton', 'bernie sanders', 'sanders', 'barack obama', 'obama']


def gkg_filter(df):

    '''Only select predefined US sources'''

    SOURCE_LIST = ['go.com', 'dallasnews.com','slate.com','breitbart.com','foxnews.com', 'huffingtonpost.com', 'nytimes.com',
                   'nbcnews.com', 'washingtonpost.com','cbsnews.com','cnn.com','newsweek.com','latimes.com', 'nydailynews.com',
                   'nypost.com','wsj.com', 'voanews.com','reuters.com','time.com','usatoday.com']

    df = df[df.source.isin(SOURCE_LIST)]
    
    df = df.replace('go.com','ABC')
    df = df.replace('dallasnews.com','Dallas News')
    df = df.replace('slate.com','Slate')
    df = df.replace('breitbart.com','Breitbart')
    df = df.replace('foxnews.com','Fox News')
    df = df.replace('huffingtonpost.com','The Huffington Post')
    df = df.replace('nytimes.com','The New York Times')
    df = df.replace('nbcnews.com','NBC')
    df = df.replace('washingtonpost.com','The Washington Post')
    df = df.replace('cbsnews.com','CBS')
    df = df.replace('cnn.com','CNN')
    df = df.replace('newsweek.com','Newsweek')
    df = df.replace('latimes.com','The LA Times')
    df = df.replace('nydailynews.com','NY Daily news')
    df = df.replace('nypost.com','The New York Post')
    df = df.replace('wsj.com','The Wallstreet Journal')
    df = df.replace('voanews.com','VOA News')
    df = df.replace('reuters.com','Reuters')
    df = df.replace('time.com','Time')
    df = df.replace('usatoday.com','USA Today')

    return df

def entities_themes(df, THEME_LIST, ENTITY_LIST):

    '''Sets and Counts entities and themes'''
    
    df = df[df['themes'].notnull()]
    df = df[df['entities'].map(len) != 0]

    for x in df['entities']:
            if '' in x:
                x.remove('')
    
  
    
    for t in THEMES:
            df[t] = df.themes.str.count(t)

    
    for e in ENTITIES:
            df[e] = df.entities.str.count(e)

    themes = [c for c in df.columns if c.isupper()]
    #entities = [c for c in df.columns if c == 'donald trump' or c == 'ted cruz' or c == 'hillary clinton' or c == 'bernie sanders' or c == 'gop' or c == 'republican party' or c == 'democratic party' or c=='barack obama']
    
    
    return df, themes, entities


mft_names = ['Care','Harm','Fairness','Cheating','Loyalty','Betrayal','Authority','Subversion','Purity','Degradation', 'MoralityGeneral']
mft_ind = ['Care','Harm','Fairness','Cheating']
mft_bind = ['Loyalty','Betrayal','Authority','Subversion','Purity','Degradation']
mft_valence = ['care-valence','fairness-valence','loyalty-valence','authority-valence','purity-valence']

liwc_vars = ['c5.{}'.format(i) for i in range(1,63)]
liwc_names = gcam[gcam.Variable.isin(liwc_vars)].DimensionHumanName.values

wordnet_vars = ['c14.{}'.format(i) for i in range(1,12)]
wordnet_names = gcam[gcam.Variable.isin(wordnet_vars)].DimensionHumanName.values


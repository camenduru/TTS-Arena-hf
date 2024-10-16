from .config import *
from .db import *
from .models import *

import pandas as pd
def get_leaderboard(reveal_prelim = False, hide_battle_votes = False):
    conn = get_db()
    cursor = conn.cursor()
    
    if hide_battle_votes:
        sql = '''
        SELECT m.name, 
               SUM(CASE WHEN v.username NOT LIKE '%_battle' AND v.vote = 1 THEN 1 ELSE 0 END) as upvote, 
               SUM(CASE WHEN v.username NOT LIKE '%_battle' AND v.vote = -1 THEN 1 ELSE 0 END) as downvote
        FROM model m
        LEFT JOIN vote v ON m.name = v.model
        GROUP BY m.name
        '''
    else:
        sql = '''
        SELECT name, 
               SUM(CASE WHEN vote = 1 THEN 1 ELSE 0 END) as upvote, 
               SUM(CASE WHEN vote = -1 THEN 1 ELSE 0 END) as downvote
        FROM model
        LEFT JOIN vote ON model.name = vote.model
        GROUP BY name
        '''
    
    cursor.execute(sql)
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['name', 'upvote', 'downvote'])
    df['name'] = df['name'].replace(model_names)
    df['votes'] = df['upvote'] + df['downvote']

    # Filter out rows with insufficient votes if not revealing preliminary results
    if not reveal_prelim:
        df = df[df['votes'] > 500]

    ## ELO SCORE
    df['score'] = 1200
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                expected_a = 1 / (1 + 10 ** ((df['score'][j] - df['score'][i]) / 400))
                expected_b = 1 / (1 + 10 ** ((df['score'][i] - df['score'][j]) / 400))
                actual_a = df['upvote'][i] / df['votes'][i] if df['votes'][i] > 0 else 0.5
                actual_b = df['upvote'][j] / df['votes'][j] if df['votes'][j] > 0 else 0.5
                df.at[i, 'score'] += 32 * (actual_a - expected_a)
                df.at[j, 'score'] += 32 * (actual_b - expected_b)
    df['score'] = round(df['score'])
    ## ELO SCORE
    df = df.sort_values(by='score', ascending=False)
    df['order'] = ['#' + str(i + 1) for i in range(len(df))]
    df = df[['order', 'name', 'score', 'votes']]
    return df

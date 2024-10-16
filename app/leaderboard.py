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
                try:
                    expected_a = 1 / (1 + 10 ** ((df['score'].iloc[j] - df['score'].iloc[i]) / 400))
                    expected_b = 1 / (1 + 10 ** ((df['score'].iloc[i] - df['score'].iloc[j]) / 400))
                    actual_a = df['upvote'].iloc[i] / df['votes'].iloc[i] if df['votes'].iloc[i] > 0 else 0.5
                    actual_b = df['upvote'].iloc[j] / df['votes'].iloc[j] if df['votes'].iloc[j] > 0 else 0.5
                    df.iloc[i, df.columns.get_loc('score')] += 32 * (actual_a - expected_a)
                    df.iloc[j, df.columns.get_loc('score')] += 32 * (actual_b - expected_b)
                except Exception as e:
                    print(f"Error in ELO calculation for rows {i} and {j}: {str(e)}")
                    continue
    df['score'] = round(df['score'])
    ## ELO SCORE
    df = df.sort_values(by='score', ascending=False)
    df['order'] = ['#' + str(i + 1) for i in range(len(df))]
    df = df[['order', 'name', 'score', 'votes']]
    return df

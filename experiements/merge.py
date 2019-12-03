import pandas as pd
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_path')

args = parser.parse_args()

path = args.input_path # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename)
    df = df.rename(columns = {
        'Unnamed: 0': 'Epoch', 
        # 'training_loss': 'train_{}'.format(filename).replace('.csv', '').replace('./', ''),
        # 'testing_loss': 'test_{}'.format(filename).replace('.csv', '').replace('./', ''),

    }, errors = 'raise')

    # moving average 
    df['testing_loss'] = df['testing_loss'].rolling(5).mean()

    df['Experiment'] = filename.replace('.csv', '').replace('./', '')
    df['Type'] = 'Social_LSTM' if 'social' in filename else 'Vanilla_LSTM'
    li.append(df)

frame = pd.concat(li, axis=0).dropna(axis = 0)
# frame.to_csv('merged.csv')

import plotly.express as px

fig = px.line(frame, 
    x = 'Epoch', y = 'testing_loss',
    color = 'Type',
    line_group = 'Experiment')

fig.show()
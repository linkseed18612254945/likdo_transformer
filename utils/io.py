import json
import pandas as pd

def dump_json(file_path, value, method='w'):
    with open(file_path, method) as f:
        f.write(json.dumps(value, sort_keys=True, indent=4, separators=(',', ': ')))
        if method == 'a':
            f.write('\n')

def load_json(file_path):
    with open(file_path, 'r') as f:
        json_file = json.load(f)
    return json_file

def load_csv(csv_path, without_header=False, shuffle=False):
    if without_header:
        df = pd.read_csv(csv_path, header=None, encoding='utf-8')
    else:
        df = pd.read_csv(csv_path, encoding='utf-8')
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df

def load_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return lines
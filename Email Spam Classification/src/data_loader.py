import pandas as pd

def load_data(path='/content/email_spam_detect_dataset.csv'):
    df = pd.read_csv(path)
    return df

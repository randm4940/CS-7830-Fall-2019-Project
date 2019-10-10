import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_heart_data():
    return pd.read_csv('../../heart.csv')

def print_count_of_target(target, df):
    print('\nTarget value count:')
    print(df.target.value_counts())
    sns.countplot(x=target, data=df)
    plt.show()


if __name__ == "__main__":
    df = read_heart_data()
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 120)
    print(df.head(10))

    # show how many of each target we have
    print_count_of_target('target', df)




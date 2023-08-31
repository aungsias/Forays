
import pandas as pd

def check_data_frequency(economic_data_dict):
    freq_dict = {}
    for indicator, data in economic_data_dict.items():
        freq = pd.infer_freq(data.index)
        freq_dict[indicator] = freq
    return freq_dict

def check_stock_frequency(data):
    freq = pd.infer_freq(data.index)
    if freq is None:
        print('No set frequency.')
    else:
        print(freq)

def growth(data):
    return (1 + data).cumprod() - 1
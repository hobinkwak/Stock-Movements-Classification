import warnings
import matplotlib.pyplot as plt
import pandas as pd


def global_setting():
    warnings.filterwarnings("ignore")
    plt.rcParams["font.family"] = "NanumGothic"
    pd.set_option("display.max_columns", None)

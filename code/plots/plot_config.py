import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13) 
matplotlib.rcParams.update({'font.size': 13.5})

def millions_nounit(x, pos):
    """takes as args are the value and tick pos."""
    return f'{x * 1e-6:.0f}'

def millions(x, pos):
    """takes as args are the value and tick pos."""
    return f'{x * 1e-6:.1f}M'

def decmillions_nounit(x, pos):
    """takes as args are the value and tick pos."""
    return f'{x * 1e5:.2f}'

COLOR_BLIND_PALETTE = {
    "red": "#e41a1c",
    "blue": "#377eb8",
    "orange": '#ff7f00',
    "green": '#4daf4a',
    "brown": '#a65628',
    "yellow": '#dede00',
    "black": "#000000",
    "gray": '#999999',
    "pink": '#f781bf',
    "cyan": "#00FFFF",
    "purple": '#984ea3', 
    "teal": "#009392",        
    "light_gray": "#cccccc",
    "lime": "#00FF00",
    "dark_red": "#8B0000",
}

COLORS = list(COLOR_BLIND_PALETTE.values())

MARKERS = ['o', 'D', 'X', 's', 'v', '^', '>', '<', '+', '*', 'p']

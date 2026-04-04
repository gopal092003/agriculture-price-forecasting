import matplotlib.pyplot as plt
import os

def save_plot(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
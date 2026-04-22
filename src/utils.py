import numpy as np
import matplotlib.pyplot as plt

def plot_model_history(history, metrics, figsize=(8, 4)):
    """
    Plot model training history metrics.
    
    Args:
        history: Keras model history object from model.fit()
        metrics: List of strings, e.g., ["loss", "f1_score"]
        figsize: Figure size as (width, height)
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    n_epochs = len(history.history[metrics[0]])
    epoch_numbers = np.arange(1, n_epochs + 1)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot train and validation metrics
        ax.plot(epoch_numbers, history.history[metric])
        ax.plot(epoch_numbers, history.history[f"val_{metric}"])
        
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_ylim(ymin=0)
        ax.set_xlabel('epoch')
        ax.set_xticks(epoch_numbers)  # Only show integer epochs
        ax.legend(['train', 'val'], loc='upper left')
    
    plt.tight_layout()
    plt.show()
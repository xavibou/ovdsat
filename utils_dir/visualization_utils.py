import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_and_save_proposals(image, instances, save_path, k=100):
    # Get proposal boxes and objectness scores
    proposal_boxes = instances.proposal_boxes.tensor.cpu().numpy()
    objectness_scores = instances.objectness_logits.cpu().numpy()

    # Create a figure and axes
    fig, ax = plt.subplots(1)
    
    # Convert the torch.Tensor image to a NumPy array
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    ax.imshow(image)

    # Plot proposals with top K objectness scores
    top_k = min(k, len(proposal_boxes))  # Change '10' to the desired number of top proposals to visualize
    for i in range(top_k):
        x, y, w, h = proposal_boxes[i]
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Save the figure to a file
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_classification_confusion_matrix(matrix, nc, normalize=True, save_dir='', names=()):
        import seaborn as sn
        import warnings
        import matplotlib.pyplot as plt
        from pathlib import Path

        array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close(fig)
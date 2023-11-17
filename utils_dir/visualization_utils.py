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
import matplotlib.pyplot as plt

def plot_digit(image, label=None):
    image = image.values.reshape(28, 28)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis("off")
    if label is not None:
        plt.title(f"Label: {label}")
    plt.show()

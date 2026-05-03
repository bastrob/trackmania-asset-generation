import matplotlib.pyplot as plt


def show_tensor_image(img_tensor, title=""):
    img = img_tensor.detach().cpu()
    img = img.permute(1, 2, 0)
    img = (img + 1) / 2
    img = img.clamp(0, 1)
    
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

def show_triplet(original, noisy, reconstructed):
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    show_tensor_image(original, "Original")

    plt.subplot(1, 3, 2)
    show_tensor_image(noisy, "Noisy")

    plt.subplot(1, 3, 3)
    show_tensor_image(reconstructed, "Reconstructed")

    plt.show()
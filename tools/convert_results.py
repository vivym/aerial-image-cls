import numpy as np
import torch
from PIL import Image


def main():
    outputs = torch.load("results/results.pth", map_location="cpu")

    label_map = [0, 5, 10, 30, 40, 80, 90]
    label_map = {
        i: label
        for i, label in enumerate(label_map)
    }

    for image_names, preds in outputs:
        for image_name, pred in zip(image_names, preds):
            pred = pred.numpy()

            remapped_pred = np.zeros_like(pred)
            for src, tgt in label_map.items():
                remapped_pred[pred == src] = tgt

            image = Image.fromarray(pred, "L")
            image.save(f"results/masks/{image_name}")


if __name__ == "__main__":
    main()

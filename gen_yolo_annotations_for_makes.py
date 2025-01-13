import os
from transformers import AutoImageProcessor, ConditionalDetrForObjectDetection
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50", use_fast=True)
model = ConditionalDetrForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")

def visualize_detections(image, results, output_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        box = box.tolist()
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            box[0], box[1] - 10,
            f"{model.config.id2label[label.item()]}: {score:.2f}",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="red", alpha=0.7)
        )
    plt.axis("off")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def generate_annotations(csv_file, output_dir, vis_dir=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if vis_dir and not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    with open(csv_file, 'r') as file:
        lines = file.readlines()[1:]  # Pomijamy nagłówek

    for line in tqdm(lines, desc="Processing images"):
        image_path, brand = line.strip().split(",")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        if results["scores"].shape[0] == 0:
            print(f"No detections for {image_path}")
            continue

        annotation_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
        with open(annotation_path, 'w') as f:
            for i in range(results["scores"].shape[0]):
                box = results["boxes"][i].tolist()
                label = results["labels"][i].item()
                score = results["scores"][i].item()

                # Normalizacja bounding boxów do formatu YOLO
                x_center = ((box[0] + box[2]) / 2) / image.size[0]
                y_center = ((box[1] + box[3]) / 2) / image.size[1]
                width = (box[2] - box[0]) / image.size[0]
                height = (box[3] - box[1]) / image.size[1]

                f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n")

        if vis_dir:
            vis_path = os.path.join(vis_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.jpg")
            visualize_detections(image, results, output_path=vis_path)

    print(f"Annotations saved in {output_dir}")


if __name__ == "__main__":
    csv_file = "datasets/cars_make_models/test/mini_anno.csv"
    output_dir = "datasets/cars_make_models/test/detr_annotations"
    vis_dir = "datasets/cars_make_models/test/detr_visualizations"
    generate_annotations(csv_file, output_dir, vis_dir)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import pandas as pd
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import cv2 as cv
import deepsimpson.config.config as cg

split = "train"
# Load data
major_axis_df = pd.read_csv(f"deepsimpson/output/features/simpsons_{split}.csv")
mask_df = pd.read_csv(f"deepsimpson/output/segmentation/masks/mask_coordinates_{split}.csv")
tracings = pd.read_csv(f"{cg.DATA_DIR}/VolumeTracings.csv")

major_axis_df = pd.read_csv(f"/home/eda/Desktop/DeepSimpson/deepsimpson/output/features/simpsons_ed_es.csv")
# mask_df = pd.read_csv(f"deepsimpson/output/segmentation_results/mask_coordinates_{split}.csv")
# Choose a video file to demonstrate
file_name = "0X1F97C88FCA1837FC.avi"
# frame_codes = list(tracings[tracings["FileName"] == file_name]["Frame"].unique())

frame_codes = [46,55]
# Open the video
vid_cap = cv.VideoCapture(f"{cg.DATA_DIR}/Videos/{file_name}")
total_frames = int(vid_cap.get(cv.CAP_PROP_FRAME_COUNT))


def plot_example(file, frame_codes):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.set_facecolor("white")
    fig.subplots_adjust(hspace=0.25, wspace=0.15)

    # Modern, clean, well-aligned title
    plt.suptitle(
        f"Echocardiography Visualization with Segmentation & Geometric Features\n"
        f"File: {file} | Frames → ED: {frame_codes[0]}, ES: {frame_codes[1]}",
        fontsize=15, fontweight="bold", color="#333333"
    )

    for idx, frame_num in enumerate(frame_codes):
        for ax in axs[idx]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # --- Features (left)
        axs[idx, 0].set_title("Geometric Features", fontsize=11)
        major_axis = major_axis_df[
            (major_axis_df.Frame == frame_num) & (major_axis_df.Filename == file)
        ]
        lines = [((row['Start_X'], row['Start_Y']), (row['End_X'], row['End_Y'])) for _, row in major_axis.iterrows()]
        if lines:
            lc = mc.LineCollection(lines, colors="orangered", linewidths=2, alpha=0.9, label="Features")
            axs[idx, 0].add_collection(lc)
        axs[idx, 0].set_xlim(0, 112)
        axs[idx, 0].set_ylim(112, 0)
        axs[idx, 0].legend(loc="upper right", fontsize=8, framealpha=0.7)

        # --- Ultrasound image (middle)
        axs[idx, 1].set_title("Echocardiographic Image", fontsize=11)
        vid_cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = vid_cap.read()
        if success and frame is not None:
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            axs[idx, 1].imshow(frame_gray, cmap="gray")
        axs[idx, 1].set_xlabel("Original ultrasound image", fontsize=9)

        # --- Mask + Features (right)
        axs[idx, 2].set_title("Segmentation Mask + Features", fontsize=11)
        if success and frame is not None:
            axs[idx, 2].imshow(frame_gray, cmap="gray")

        mask_points = mask_df[(mask_df["Filename"] == file) & (mask_df["Frame"] == frame_num)]
        if not mask_points.empty:
            axs[idx, 2].scatter(
                mask_points["X"], mask_points["Y"],
                color="deepskyblue", s=3, alpha=0.6, label="Segmentation Mask"
            )

        if lines:
            lc = mc.LineCollection(lines, colors="orangered", linewidths=2, alpha=0.9, label="Features")
            axs[idx, 2].add_collection(lc)

        axs[idx, 2].legend(loc="upper right", fontsize=8, framealpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("deepsimpson/output_images/echocardiography_mask_with_features_0X1F97C88FCA1837FC.png", dpi=300)
    plt.close()

# plot_example(file_name, frame_codes)
plot_example(file_name, frame_codes )


print("Echocardiography, Segmentation, and Feature Visualizations Completed!")

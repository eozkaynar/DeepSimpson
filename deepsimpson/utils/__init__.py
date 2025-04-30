import os
import numpy as np
import torch
import tqdm
from sklearn.decomposition import PCA
import csv

import matplotlib.pyplot as plt
def clip_line_to_mask(x1, y1, x2, y2, mask, num_points=100):
    """Return start and end of the total clipped segment (only one segment per line)."""
    xs = np.linspace(x1, x2, num_points)
    ys = np.linspace(y1, y2, num_points)

    # Check which points are inside the mask
    inside = [mask[int(round(y)), int(round(x))] > 0 
              if 0 <= int(round(y)) < mask.shape[0] and 0 <= int(round(x)) < mask.shape[1] 
              else False
              for x, y in zip(xs, ys)]

    # Find indices where inside == True
    inside_indices = [i for i, val in enumerate(inside) if val]

    if not inside_indices:
        return None  # No part of the line is inside the mask

    # Get start and end index of the entire masked segment
    start_idx = inside_indices[0]
    end_idx = inside_indices[-1]

    return (xs[start_idx], ys[start_idx]), (xs[end_idx], ys[end_idx])
def savemajoraxis_with_simpson(results, output, split, num_discs=20, frame_btw_ed_es=True):
    """
    Extracts and saves perpendicular line segments (discs) along the major axis of each mask
    using PCA and clips them to stay within the segmented region.
    Output is written in CSV format for Simpson's rule integration.
    """

    output_path         = os.path.join(output, f"simpsons_{split}.csv")
    buffer              = []  # Buffer to hold all rows before writing
    
    for entry in results:
        if len(entry) != 7:
            print(f"[WARNING] Unrecognized format in savemajoraxis_with_simpson():")
            continue

        filename, logit, _, systole, diastole, large_index, small_index = entry
        T = logit.shape[0]  # Number of frames in the video

        if frame_btw_ed_es:
            start_frame =  min(large_index, small_index)
            end_frame   = max(large_index, small_index) + 1  # +1 for inclusive range
        else:
            start_frame = 0
            end_frame = T
        total_frame_number  = 0   # Count unique frames saved
        for frame_idx in range(start_frame, end_frame):
            mask = logit[frame_idx] > 0
            if np.sum(mask) == 0:
                continue  # Skip empty masks
            total_frame_number += 1  # Count each saved frame
            # Determine phase label for the current frame
            if frame_idx == large_index:
                phase = "ED_GT"
            elif frame_idx == small_index:
                phase = "ES_GT"
            elif frame_idx in diastole:
                phase = "ED"
            elif frame_idx in systole:
                phase = "ES"
            else:
                phase = "ALL"

            # Perform PCA to determine the major axis of the mask
            result = find_major_axis_pca(mask)
            if result[0] is None:
                continue

            (start_x, start_y), (end_x, end_y) = result
            buffer.append([filename, phase, frame_idx, start_x, start_y, end_x, end_y, total_frame_number,"Major Axis"])
            # Compute direction and perpendicular vectors
            axis_vector = np.array([end_x - start_x, end_y - start_y])
            if not axis_vector.any():
                continue  # Skip degenerate cases earlier

            axis_length = np.linalg.norm(axis_vector)
            unit_vector = axis_vector / axis_length
            perp_vector = np.array([-unit_vector[1], unit_vector[0]])  # Rotate 90° CCW

            # Generate evenly spaced perpendicular segments along the axis
            for i in range(num_discs):
                t = i / (num_discs - 1)
                midpoint_x = start_x + t * axis_vector[0]
                midpoint_y = start_y + t * axis_vector[1]

                # Define endpoints of the perpendicular line (disc)
                line_length = axis_length * 0.5
                line_start_x = midpoint_x - perp_vector[0] * line_length
                line_start_y = midpoint_y - perp_vector[1] * line_length
                line_end_x   = midpoint_x + perp_vector[0] * line_length
                line_end_y   = midpoint_y + perp_vector[1] * line_length

                # Clip the disc to the inside of the mask
                clipped = clip_line_to_mask(line_start_x, line_start_y, line_end_x, line_end_y, mask)
                if clipped is not None:
                    (sx, sy), (ex, ey) = clipped
                    buffer.append([filename, phase, frame_idx, sx, sy, ex, ey,"","Simpson's Disc"])

    # Write everything at once after processing
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Phase", "Frame", "Start_X", "Start_Y", "End_X", "End_Y","Frames","Type"])
        writer.writerows(buffer)


def plot_size_curve(filename, size, systole, diastole, output_dir=None, show=False):
    """
    Plots the segmentation size curve over frames and marks systole/diastole peaks.

    Parameters:
        filename (str): name of the video (used for saving plot)
        size (np.ndarray): array of segmentation pixel counts per frame
        systole (set): indices of systole peaks
        diastole (set): indices of diastole peaks
        output_dir (str): folder to save the plots (optional)
        show (bool): whether to show the plot interactively (optional)
    """
    plt.figure(figsize=(10, 4))
    plt.plot(size, label="Segmentation Area (pixels)", color="gray")
    
    # Plot diastole (maxima)
    if diastole:
        plt.scatter(list(diastole), size[list(diastole)], color="blue", label="Diastole", zorder=3)

    # Plot systole (minima)
    if systole:
        plt.scatter(list(systole), size[list(systole)], color="red", label="Systole", zorder=3)

    plt.title(f"Segmentation Size Over Time\n{filename}")
    plt.xlabel("Frame")
    plt.ylabel("Area (pixels)")
    plt.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_curve.png")
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()



def get_mean_and_std(dataset: torch.utils.data.Dataset,
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4):
    """Computes mean and std from samples from a Pytorch dataset.

    Args:
        dataset (torch.utils.data.Dataset): A Pytorch dataset.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int or None, optional): Number of samples to take from dataset. If ``None'', mean and
            standard deviation are computed over all elements.
            Defaults to 128.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 8.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.

    Returns:
       A tuple of the mean and standard deviation. Both are represented as np.array's of dimension (channels,).
    """

    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    n = 0  # number of elements taken (should be equal to samples by end of for loop)
    s1 = 0.  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))
    for (x, *_) in tqdm.tqdm(dataloader):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = s1 / n  # type: np.ndarray
    std = np.sqrt(s2 / n - mean ** 2)  # type: np.ndarray

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std
    
def savemask(results, output, split):
    """Saves segmentation mask pixel coordinates for all frames with ED/ES/GT phase tags to CSV."""

    mask_csv_path = os.path.join(output, f"mask_coordinates_{split}.csv")

    with open(mask_csv_path, "w") as g:
        g.write("Filename,Phase,X,Y,Frame\n")  # CSV Header

    def save_mask_coordinates(mask, filename, phase, frame):
        """Save (X, Y) coordinates of the segmentation mask to CSV."""
        y_coords, x_coords = np.where(mask > 0)
        with open(mask_csv_path, "a") as g:
            for x, y in zip(x_coords, y_coords):
                g.write(f"{filename},{phase},{x},{y},{frame}\n")

    for entry in results:
        if len(entry) == 7:
            filename, logit, _, systole, diastole, large_index, small_index = entry

            T = logit.shape[0]  # Total frames

            for frame_idx in range(T):
                mask = logit[frame_idx] > 0
                if np.sum(mask) == 0:
                    continue

                # Label phase
                if frame_idx == large_index:
                    phase = "ED_GT"
                elif frame_idx == small_index:
                    phase = "ES_GT"
                elif frame_idx in diastole:
                    phase = "ED"
                elif frame_idx in systole:
                    phase = "ES"
                else:
                    phase = "ALL"

                save_mask_coordinates(mask, filename, phase, frame_idx)

        elif len(entry) == 9:
            # Eski formatı destekle (ED / ES ikili format)
            filename, _, logit_ed, _, ed_idx, _, logit_es, _, es_idx = entry
            for phase, idx, logit in [("ED", ed_idx, logit_ed), ("ES", es_idx, logit_es)]:
                mask = logit > 0
                if np.sum(mask) == 0:
                    continue
                save_mask_coordinates(mask, filename, phase, idx)

        else:
            print(f"[WARNING] Unrecognized format for savemask()")
            continue


def savesize(results, output, split):
    """Saves segmentation sizes to CSV for both single-phase and ED-ES pairs."""

    size_csv_path = os.path.join(output, f"size_{split}.csv")
    
    with open(size_csv_path, "w") as g:
        g.write("Filename,Phase,Frame,Size\n")

    def write_size(filename, phase, frame, size):
        with open(size_csv_path, "a") as g:
            g.write(f"{filename},{phase},{frame},{size}\n")

    for entry in results:
        if len(entry) == 7:
            filename, _, size_array, systole, diastole, large_index, small_index = entry

            for frame_idx, size in enumerate(size_array):
                if frame_idx == large_index:
                    phase = "ED_GT"
                elif frame_idx == small_index:
                    phase = "ES_GT"
                elif frame_idx in diastole:
                    phase = "ED"
                elif frame_idx in systole:
                    phase = "ES"
                else:
                    phase = "ALL"

                write_size(filename, phase, frame_idx, size)

        elif len(entry) == 9:
            # Format: (filename, "ED", logit_ed, size_ed, ed_idx, "ES", logit_es, size_es, es_idx)
            filename, _, _, size_ed, ed_idx, _, _, size_es, es_idx = entry
            write_size(filename, "ED", ed_idx, size_ed)
            write_size(filename, "ES", es_idx, size_es)

        else:
            print(f"[WARNING] Unrecognized result format savesize()")



def find_major_axis_pca(mask):
    """
    Uses PCA to determine the major axis of a binary mask.
    Returns the two endpoints (start_x, start_y) and (end_x, end_y).
    """
    # Extract the (x, y) coordinates of all nonzero pixels
    points = np.column_stack(np.where(mask))  # (row, col) 

    # Get apex point
    apex   = points[np.argmin(points[:,0])]

    if len(points) < 2:  # If there are not enough points, return None
        return None, None

    # Apply PCA to find the principal components
    pca = PCA(n_components=2)
    pca.fit(points)

    # Get the primary eigenvector (major axis direction)
    major_axis_vector = pca.components_[0]  # First principal component
    center = np.mean(points, axis=0)  # Compute centroid

    # Project points onto the major axis
    projections = np.dot(points - center, major_axis_vector)

    # Get the extreme points along the major axis
    min_proj, max_proj = projections.min(), projections.max()
    start_point = center + min_proj * major_axis_vector
    end_point = center + max_proj * major_axis_vector

    # Convert to (X, Y) format (swap row, col → x, y)
    return (apex[1], apex[0]), (end_point[1], end_point[0]) 

def savemajoraxis(results, output, split):
    """
    Saves the major axis (PCA-based) of each frame in each video,
    using both ground-truth and peak-based ED/ES labeling.
    """

    major_axis_csv_path = os.path.join(output, f"major_axis_pca_{split}.csv")

    with open(major_axis_csv_path, "w", newline="") as g:
        writer = csv.writer(g)
        writer.writerow(["Filename", "Phase", "Frame", "Start_X", "Start_Y", "End_X", "End_Y"])

        for entry in results:
            if len(entry) == 7:
                filename, logit, _, systole, diastole, large_index, small_index = entry

                T = logit.shape[0]  # Total frames

                for frame_idx in range(T):
                    mask = logit[frame_idx] > 0

                    if np.sum(mask) == 0:
                        continue

                    # Label phase
                    if frame_idx == large_index:
                        phase = "ED_GT"
                    elif frame_idx == small_index:
                        phase = "ES_GT"
                    elif frame_idx in diastole:
                        phase = "ED"
                    elif frame_idx in systole:
                        phase = "ES"
                    else:
                        phase = "ALL"

                    # PCA
                    result = find_major_axis_pca(mask)

                    if result[0] is None:
                        continue

                    (start_x, start_y), (end_x, end_y) = result
                    writer.writerow([filename, phase, frame_idx, start_x, start_y, end_x, end_y])

            elif len(entry) == 9:
                # Legacy format: (filename, "ED", logit_ed, size_ed, ed_idx, "ES", logit_es, size_es, es_idx)
                filename, _, logit_ed, _, ed_idx, _, logit_es, _, es_idx = entry
                for phase, frame_idx, logit in [("ED", ed_idx, logit_ed), ("ES", es_idx, logit_es)]:
                    mask = logit > 0
                    if np.sum(mask) == 0:
                        continue
                    result = find_major_axis_pca(mask)
                    if result[0] is None:
                        continue
                    (start_x, start_y), (end_x, end_y) = result
                    writer.writerow([filename, phase, frame_idx, start_x, start_y, end_x, end_y])
            else:
                print(f"[WARNING] Unrecognized format for savemajoraxis()")
                continue



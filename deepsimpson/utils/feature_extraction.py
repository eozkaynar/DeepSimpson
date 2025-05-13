import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from deepsimpson.datasets.EchoSegmentation import Echo
from deepsimpson.utils import get_mean_and_std, savemask, savesize, savemajoraxis , savemajoraxis_with_simpson
import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm
import os



@click.command("feature_extraction")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="/home/eda/Desktop/dynamic/EchoNet-Dynamic")
@click.option("--output", type=click.Path(file_okay=False), default="deepsimpson/output/features")
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.segmentation.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.segmentation.__dict__[name]))),
    default="deeplabv3_resnet50")
@click.option("--pretrained/--random", default=True)
@click.option("--batch_size", type=int, default=10)
@click.option(
    "--weights",
    type=click.Path(exists=True, dir_okay=False),
    default="/home/eda/Desktop/EF-Estimation-Paper-Imlementation/WorkingFolder/output/segmentation/deeplabv3_resnet50_pretrained/best.pt",
    show_default=True,
    help="Path to the pretrained model checkpoint."
)

@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,
    pretrained=True,
    weights=None,
    model_name="deeplabv3_resnet50",
    num_workers=4,
    batch_size=10,
    device=None,
    seed=0,
):
  
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation")
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.segmentation.__dict__[model_name](pretrained=pretrained, aux_loss=True)

    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    
    try:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"\n[ERROR] Failed to load the model: {e}")
        print("[WARNING] A pre-trained model checkpoint is required for segmentation.")
        print("          Please provide a valid path using the '--weights' argument.\n")
        exit(1)  # Safely terminate the program


    # Compute mean and std
    mean, std = get_mean_and_std(Echo(root=data_dir, split="train"))

 
                                    
    for split in [ "train", "val", "test"]:      
        # Saving videos with segmentations
        dataset = Echo(root=data_dir, split=split,
                                        mean=mean, std=std,  # Normalization
                                        length=None, max_length=None, period=1  # Take all frames
                                        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=num_workers, shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)

        # Save videos with segmentation


        os.makedirs(os.path.join(output, "size"), exist_ok=True)

        model.eval()

        with torch.no_grad():
            # Run segmentation model once for all videos
            results = []
            for (x, (filenames, large_index, small_index, _, _, _, _, _, _, _,_,_,_,_), length) in tqdm.tqdm(dataloader):
                y = np.concatenate([
                    model(x[i:(i + batch_size), :, :, :].to(device))["out"].detach().cpu().numpy()
                    for i in range(0, x.shape[0], batch_size)
                ])

                x = x.cpu().numpy()  # Ensure x is on CPU before converting to NumPy

                start = 0
                for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                    logit = y[start:(start + offset), 0, :, :] 

                    # Compute segmentation size per frame
                    size = (logit > 0).sum((1, 2))

                    # Identify systole and diastole frames using peak detection
                    trim_min = sorted(size)[round(len(size) ** 0.05)]
                    trim_max = sorted(size)[round(len(size) ** 0.95)]
                    trim_range = trim_max - trim_min
                    systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])
                    diastole = set(scipy.signal.find_peaks(size, distance=20, prominence=(0.50 * trim_range))[0])

                    # Store results for both CSVs
                    results.append((filename,logit, size, systole, diastole, large_index[i], small_index[i]))
                    # # Save visualizations
                    # plot_output_dir = os.path.join(output, "curves")
                    # plot_size_curve(filename, size, systole, diastole, output_dir=plot_output_dir)


                    start += offset  # Move to next video
        


                # savesize(results,output,split)
                # savemajoraxis(results,output,split)
                # savemask(results,output,split)
                savemajoraxis_with_simpson(results, output, split, num_discs=20, type="ed-es")


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i

if __name__ == "__main__":
    run()  

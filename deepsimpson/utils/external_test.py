import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from deepsimpson.datasets.EchoSegmentation import Echo
from deepsimpson.utils import get_mean_and_std, savevideo, savemajoraxis_with_simpson
from deepsimpson.models.sequence_models import LSTM, RNN
# from deepsimpson.datasets.RNNDataset import Dataset
from deepsimpson.datasets.LSTMDataset import Dataset_lstm
import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm



@click.command("feature_extraction")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="/home/eda/Desktop/dynamic/EchoNet-Dynamic")
@click.option("--external_dir", type=click.Path(exists=True, file_okay=False), default="/home/eda/Desktop/DeepSimpson/External Videos/")
@click.option("--output", type=click.Path(file_okay=False), default="deepsimpson/output_external")
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.segmentation.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.segmentation.__dict__[name]))),
    default="deeplabv3_resnet50")
@click.option("--pretrained/--random", default=True)
@click.option("--batch_size", type=int, default=10)
@click.option(
    "--weights_seg",
    type=click.Path(exists=True, dir_okay=False),
    default="/home/eda/Desktop/EF-Estimation-Paper-Imlementation/WorkingFolder/output/segmentation/deeplabv3_resnet50_pretrained/best.pt",
    show_default=True,
    help="Path to the pretrained model checkpoint."
)
@click.option(
    "--weights_reg_rnn",
    type=click.Path(exists=True, dir_okay=False),
    default="deepsimpson/output/prediction/ED_ES_RNN/best.pt",
    show_default=True,
    help="Path to the pretrained model checkpoint."
)
@click.option(
    "--weights_reg_lstm",
    type=click.Path(exists=True, dir_okay=False),
    default="deepsimpson/output/prediction/ED_ES_LSTM/best.pt",
    show_default=True,
    help="Path to the pretrained model checkpoint."
)
@click.option("--save_video", type=str, default=True)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    external_dir=None,
    output=None,
    pretrained=True,
    weights_seg=None,
    weights_reg_rnn=None,
    weights_reg_lstm=None,
    model_name="deeplabv3_resnet50",
    num_workers=4,
    batch_size=10,
    save_video=True,
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

    # Set up segmentation  model
    seg_model = torchvision.models.segmentation.__dict__[model_name](pretrained=pretrained, aux_loss=True)

    seg_model.classifier[-1] = torch.nn.Conv2d(seg_model.classifier[-1].in_channels, 1, kernel_size=seg_model.classifier[-1].kernel_size)  # change number of outputs to 1
    if device.type == "cuda":
        seg_model = torch.nn.DataParallel(seg_model)
    seg_model.to(device)
    
    try:
        checkpoint = torch.load(weights_seg)
        seg_model.load_state_dict(checkpoint['state_dict'])
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"\n[ERROR] Failed to load the model: {e}")
        print("[WARNING] A pre-trained model checkpoint is required for segmentation.")
        print("          Please provide a valid path using the '--weights' argument.\n")
        exit(1)  # Safely terminate the program

    # Load prediction model 

    model_lstm = LSTM(input_size=21)
    model_lstm = torch.nn.DataParallel(model_lstm)
    model_lstm.load_state_dict(torch.load(weights_reg_lstm)["state_dict"])

    model_rnn = RNN(input_size=21)
    model_rnn = torch.nn.DataParallel(model_rnn)
    model_rnn.load_state_dict(torch.load(weights_reg_rnn)["state_dict"])

    model_lstm.to(device).eval()
    model_rnn.to(device).eval()
    # Compute mean and std
    # mean, std = get_mean_and_std(Echo(root=data_dir, split="ext", external_data_dir=external_dir))
    mean = np.array([97.24017664, 95.95814051, 93.87780209])      # RGB sıfır ortalama
    std = np.array([55.88998819,59.40321542, 59.31306297]) # RGB için eşit standart sapma


    # dataset & dataloader 
    split = "ext"
    dataset = Echo(root=data_dir,
                   split=split,
                   external_data_dir=external_dir,
                   mean=mean, std=std,
                   length=None, max_length=None, period=1)

    dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=num_workers, pin_memory=False,
                collate_fn=_video_collate_fn)


    # video saving (optional) 
    if save_video and not all(os.path.isfile(os.path.join(output, "videos", f))
                              for f in dataset.fnames):
        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        seg_model.eval(); sub_bs = 32
        with torch.no_grad():
            
            for (x, (filenames, *_), length) in tqdm.tqdm(dataloader, desc="Save videos"):
                print("[DEBUG] Normalized input stats:",
                "min:", x.min().item(), "max:", x.max().item(), "mean:", x.mean().item())
                print(x.shape)
                # kademe kademe çıkarım (OOM korumalı)
                y = np.concatenate([seg_model(x[i:i+sub_bs].to(device))["out"]
                                        .cpu().numpy()
                                        for i in range(0, x.shape[0], sub_bs)])
                print("[DEBUG] Model output y shape:", y.shape)
                print("[DEBUG] Output stats:",
                    "min:", y.min().item(), "max:", y.max().item(), "mean:", y.mean().item())
                start = 0; x = x.numpy()
                for fname, offset in zip(filenames, length):
                    vid   = x[start:start+offset]                        # (F,C,H,W)
                    logit = y[start:start+offset, 0]                 # (F,H,W)
                    start += offset
                    # un-normalize
                    vid = vid * std.reshape(1,3,1,1) + mean.reshape(1,3,1,1)
                    F,C,H,W = vid.shape; vid = vid.astype(np.uint8)

                    pair = np.concatenate((vid, vid), 3)                # side-by-side
                    pair[:, 0, :, W:] = np.maximum(                      # blue mask overlay
                        255 * (logit > 0), pair[:, 0, :, W:])

                    pair = pair.transpose(1,0,2,3)                      # (C,F,H,2W)
                    savevideo(os.path.join(output, "videos", fname), pair, 50)
                    print("[DEBUG] logit shape:", logit.shape, "offset:", offset, "start:", start, "y.shape:", y.shape)

    # -------------------- ED/ES + Simpson measurements ------------------------
    os.makedirs(os.path.join(output, "size"), exist_ok=True)
    seg_model.eval(); sub_bs = 32; results = []
    with torch.no_grad():
        for (x, (fnames, large_i, small_i, *_), length) in tqdm.tqdm(dataloader,
                                                                     desc="Measure"):
            y = np.concatenate([seg_model(x[i:i+sub_bs].to(device))["out"]
                                .cpu().numpy()
                                for i in range(0, x.shape[0], sub_bs)])
            start = 0
            for fname, offset in zip(fnames, length):
                logit = y[start:start+offset, 0]
                size  = (logit > 0).sum((1,2))

                trim_min, trim_max = np.percentile(size, [5,95])
                trim_range = trim_max - trim_min
                systole  = set(scipy.signal.find_peaks(-size, distance=20,
                                          prominence=0.5*trim_range)[0])
                diastole = set(scipy.signal.find_peaks( size, distance=20,
                                          prominence=0.5*trim_range)[0])

                results.append((fname, logit, size, systole, diastole,
                                large_i, small_i))
                start += offset

    savemajoraxis_with_simpson(results, output, split,
                               num_discs=20, type="ed-es")
    input_dir = os.path.join(output, "simpsons_ext.csv")
    import pandas as pd 
    # Read combined file
    df_output = pd.read_csv(input_dir)

    # Add length column
    df_output["Length"] = np.sqrt((df_output["End_X"] - df_output["Start_X"]) ** 2 + (df_output["End_Y"] - df_output["Start_Y"]) ** 2)

    df_output.to_csv(input_dir, index=False)
    
    # EF Prediction 
    print("\n Running EF prediction using LSTM and RNN")
    dataset_reg = Dataset_lstm(root=input_dir,split="ext")

    dataset_pred = torch.utils.data.DataLoader(dataset_reg, batch_size=1, shuffle=False)

    with torch.no_grad():
        for (filename, X, _) in tqdm.tqdm(dataset_pred, desc="Predict EF"):
            X = X.to(device)

            ef_lstm = model_lstm(X).item()
            ef_rnn  = model_rnn(X).item()

            print(f"[RESULT] {filename[0]}")
            print(f" LSTM Predicted EF: {ef_lstm:.2f}")
            print(f" RNN  Predicted EF: {ef_rnn:.2f}")

def _video_collate_fn(batch):
    vids, tars = zip(*batch)
    lengths = [v.shape[1] for v in vids]
    vids = torch.as_tensor(np.swapaxes(np.concatenate(vids, 1), 0, 1))
    tars = tuple(zip(*tars))   # <-- zip -> tuple
    return vids, tars, lengths

if __name__ == "__main__":
    run()  

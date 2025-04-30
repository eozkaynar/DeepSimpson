from EchoSegmentation import Echo
from utils import compute_ef_histogram_weights
from nn_models import LSTM, RNN
from RNNDataset import Dataset

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from scipy.stats import pearsonr
import torch
import torchvision
import tqdm
import os
import math
import time


@click.command("segmentation")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="/home/eda/Desktop/EF-Estimation-Paper-Imlementation/WorkingFolder/output/segmentation/deeplabv3_resnet50_pretrained")
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_name", type=click.Choice(["LSTM", "RNN"]), default="RNN")
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=True)
@click.option("--num_epochs", type=int, default=85)
@click.option("--lr", type=float, default=1e-3)
@click.option("--weight_decay", type=float, default=1e-5)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=64)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)

@click.option("--input_size", type=int, default=21)
@click.option("--num_layers", type=int, default=3)

def run(
    data_dir=None,
    output=None,
    model_name="LSTM",
    weights=None,

    run_test=False,
    num_epochs=50,
    lr=1e-3,
    weight_decay=1e-5,
    lr_step_period=None,
    num_train_patients=None,
    num_workers=4,
    batch_size=32,
    device=None,
    seed=0,

    input_size=21,
    num_layers=3,
):
  
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "prediction", "_{}".format(model_name))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    if model_name == "LSTM":
        model = LSTM(input_size=input_size)
    elif model_name == "RNN":
        model = RNN(input_size=input_size)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer
    if model_name == "LSTM":
        optimizer= torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif model_name == "RNN":
        optimizer= torch.optim.Adam(model.parameters(), lr=lr)

    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)

    
    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"]    = Dataset(root=data_dir, split="train")
    dataset["val"]      = Dataset(root=data_dir, split="val")
    dataset["test"]     = Dataset(root=data_dir, split="test")

    # Compute EF-based bin weights from training data
    bin_edges, weights_per_bin = compute_ef_histogram_weights(
    dataset["train"], bins=100, value_range=(0, 100)
)
 
    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        train_losses = []
        val_losses   = []  
        

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds          = dataset[phase]
                dataloader  = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                
                loss, yhat, y, filename = run_epoch(model, dataloader, phase, optimizer, device, train_losses, val_losses,output,bin_edges=bin_edges,
    weights_per_bin=weights_per_bin) 

                f.write("{},{},{},{},{},{}\n".format(epoch,
                                                    phase,
                                                    loss,
                                                    sklearn.metrics.r2_score(y, yhat),
                                                    time.time() - start_time,
                                                    batch_size))
                f.flush()
            scheduler.step(loss)

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:

            for split in ["val", "test"]:
                dataset     = Dataset(root=data_dir, split=split)
                dataloader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda")) 
                                                                                                                 
                loss, yhat, y, filename = run_epoch(model,dataloader,split,None, device,train_losses=[], val_losses=[], output=output)

                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    g.write("filename,true_value,prediction\n")
                    for (file, pred, target) in zip(filename, yhat, y):
                            g.write("{},{:.4f},{:.4f}\n".format(file,float(target),float(pred)))

                    g.write("{} R2:   {:.3f} \n".format(split, sklearn.metrics.r2_score(y, yhat)))
                    g.write("{} MAE:  {:.4f} \n".format(split, sklearn.metrics.mean_absolute_error(y, yhat)))
                    g.write("{} RMSE: {:.4f} \n".format(split, math.sqrt(sklearn.metrics.mean_squared_error(y, yhat))))
                    y = np.array(y)
                    yhat = np.array(yhat)

                    corr, _ = pearsonr(y.ravel(), yhat.ravel())
                    g.write("{} Corr: {:.3f} \n".format(split, corr))
                    f.flush()
            
    np.save(os.path.join(output, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(output, "val_losses.npy"), np.array(val_losses))
    print(f"Train and validation losses saved to {output}")

def run_epoch(model, dataloader, split, optimizer, device, train_losses, val_losses,output,bin_edges=None,weights_per_bin=None):

    total_loss = 0.
    n          = 0
    s1         = 0     # sum of ground truth EF
    s2         = 0     # Sum of ground truth EF squared

    yhat       = []
    y          = []
    filenames  = []   

    train_flag = (split ==  'train')

    if split == 'train':
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train_flag):
        with tqdm.tqdm(total=len(dataloader)) as pbar:

            for (filename, X, ejection) in dataloader:
    
                X = X.to(device)
                ejection = ejection.to(device)


                s1 += ejection.sum()
                s2 += (ejection ** 2).sum()
                
                output  = model(X)
                #  # === Weighted loss computation ===
                # if split == 'train'and weights_per_bin is not None and bin_edges is not None:
                #     # Inverse EF normalization for binning
                #     ef_values = (ejection).detach().cpu().numpy()  # shape: (batch,)

                #     # Get bin indices (length = bins - 1)
                #     bin_indices = np.digitize(ef_values, bins=bin_edges[1:-1])
                #     bin_indices = np.clip(bin_indices, 0, len(weights_per_bin) - 1)

                #     # Get weight for each sample
                #     sample_weights = weights_per_bin[bin_indices]
                #     sample_weights = torch.tensor(sample_weights, dtype=torch.float32).to(device)

                #     # Compute per-sample loss, apply weights
                #     loss_per_sample = torch.nn.functional.mse_loss(output.view(-1), ejection, reduction='none')
                #     loss = (loss_per_sample * sample_weights).mean()
                # else:
                #     # Standard MSE loss
                loss = torch.nn.functional.mse_loss(output.view(-1), ejection)

                
    
                
            
                # Graidient for training
                if train_flag:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Accumulate losses and compute baselines
                total_loss += loss.item() * X.size(0)
                n          += X.size(0)
                avg_loss    = total_loss / n

                # Save predictions
                y.extend(ejection.cpu().numpy())
                yhat.extend(output.detach().cpu().numpy())
                filenames.extend(filename)

    
                # Show info on process bar
                pbar.set_postfix_str("{:.4f} ({:.4f})".format(avg_loss, loss.item()))
                pbar.update()

            


            if (split == "train"):
                train_losses.append(avg_loss)
            elif split == "val":
                val_losses.append(avg_loss)
            else:
                pass

    return (avg_loss, yhat, y, filenames)


if __name__ == "__main__":
    run()  

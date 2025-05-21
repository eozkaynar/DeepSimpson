import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from deepsimpson.datasets.EchoSegmentation import Echo
from deepsimpson.models.sequence_models import LSTM, RNN
# from deepsimpson.datasets.RNNDataset import Dataset
from deepsimpson.datasets.LSTMDataset import Dataset_lstm

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
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="/home/eda/Desktop/DeepSimpson/deepsimpson/output/features")
@click.option("--output_path", type=click.Path(file_okay=False), default=None)
@click.option("--model_name", type=click.Choice(["LSTM", "RNN"]), default="LSTM")
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=True)
@click.option("--num_epochs", type=int, default=100)
@click.option("--lr", type=float, default=1e-3)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=32)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)

@click.option("--input_size", type=int, default=21)
@click.option("--num_layers", type=int, default=3)

def run(
    data_dir=None,
    output_path=None,
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
    if output_path is None:
        output_path = os.path.join("deepsimpson/output", "prediction", "{}".format(model_name))
    os.makedirs(output_path, exist_ok=True)

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
        optimizer= torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    
    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"]    = Dataset_lstm(root=data_dir, split="train")
    dataset["val"]      = Dataset_lstm(root=data_dir, split="val")
    dataset["test"]     = Dataset_lstm(root=data_dir, split="test")
 
    # Run training and testing loops
    with open(os.path.join(output_path, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output_path, "checkpoint.pt"))
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
                
                loss, yhat, y, filename = run_epoch(model, dataloader, phase, optimizer, device, train_losses, val_losses,output_path) 
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
            torch.save(save, os.path.join(output_path, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output_path, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output_path, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:

            for test_split in ["val", "test"]:
                dataset     = Dataset_lstm(root=data_dir, split=test_split)
                dataloader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda")) 
                                                                                                                 
                loss, yhat, y, filename = run_epoch(model,dataloader,test_split,None, device,train_losses=[], val_losses=[], output_path=output_path)

                with open(os.path.join(output_path, "{}_predictions.csv".format(test_split)), "w") as g:
                    g.write("filename,true_value,prediction\n")
                    for (file, pred, target) in zip(filename, yhat, y):
                            g.write("{},{:.4f},{:.4f}\n".format(file,float(target),float(pred)))

                    g.write("{} R2:   {:.3f} \n".format(test_split, sklearn.metrics.r2_score(y, yhat)))
                    g.write("{} MAE:  {:.4f} \n".format(test_split, sklearn.metrics.mean_absolute_error(y, yhat)))
                    g.write("{} RMSE: {:.4f} \n".format(test_split, math.sqrt(sklearn.metrics.mean_squared_error(y, yhat))))
                    y = np.array(y)
                    yhat = np.array(yhat)

                    corr, _ = pearsonr(y.ravel(), yhat.ravel())
                    g.write("{} Corr: {:.3f} \n".format(test_split, corr))
                    f.flush()
            
    np.save(os.path.join(output_path, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(output_path, "val_losses.npy"), np.array(val_losses))
    print(f"Train and validation losses saved to {output_path}")

def run_epoch(model, dataloader, split, optimizer, device, train_losses, val_losses,output_path):

    total_loss = 0.
    avg_loss   = 0
    n          = 0
    s1 =0
    s2 =0

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
  
              #     # Standard MSE loss
                loss = torch.nn.functional.mse_loss(output.view(-1), ejection)
                # weights = torch.ones_like(ejection, device=ejection.device)
                # weights[ejection >= 70.0] = 4        # eşik ve katsayı ayarlanabilir
                # weights[ejection < 30] = 2
                # weights[(ejection >= 30) & (ejection < 50)] = 1

                # 2) Element-wise MSE (reduction='none')
                per_sample_loss = torch.nn.functional.mse_loss(
                                    output.view(-1), ejection, reduction='none')

                # 3) Ağırlıkla çarpıp ortalama al
                # loss = (weights * per_sample_loss).mean()
                

            
            
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

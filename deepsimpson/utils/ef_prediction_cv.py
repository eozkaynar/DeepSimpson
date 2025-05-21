import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from deepsimpson.datasets.EchoSegmentation import Echo
from deepsimpson.models.sequence_models import LSTM, RNN
# from deepsimpson.datasets.RNNDataset import Dataset
from deepsimpson.datasets.LSTMDataset import Dataset_lstm

from torch.utils.data import ConcatDataset, Subset
from sklearn.model_selection import KFold
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
@click.option("--num_epochs", type=int, default=150)
@click.option("--lr", type=float, default=1e-3)
@click.option("--weight_decay", type=float, default=1e-5)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=64)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=42)
@click.option("--n_folds", type=int, default=10)
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
    n_folds=5,
    num_layers=3,
):
  
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output_path is None:
        output_path = os.path.join("deepsimpson/output", "prediction", "ED_ES_CV_{}".format(model_name))
    os.makedirs(output_path, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if lr_step_period is None:
        lr_step_period = math.inf
    

    
    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"]    = Dataset_lstm(root=data_dir, split="train")
    dataset["val"]      = Dataset_lstm(root=data_dir, split="val")
    dataset["test"]     = Dataset_lstm(root=data_dir, split="test")

    dataset["all"]      = ConcatDataset([dataset["train"], dataset["val"]])
 
    # Run training and testing loops
    with open(os.path.join(output_path, "log.csv"), "a") as f:
        epoch_resume = 0
        train_losses = []
        val_losses   = []  
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_stats = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(range(len(dataset["all"]))), 1):

            print(f"Fold {fold}")

            train_loader  = torch.utils.data.DataLoader(
                        Subset(dataset["all"] , tr_idx), batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)

            val_loader    = torch.utils.data.DataLoader(
                        Subset(dataset["all"] , val_idx), batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=False)

            model = build_model(model_name, input_size, device)
                # Set up optimizer
            if model_name == "LSTM":
                optimizer= torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            elif model_name == "RNN":
                optimizer= torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
            
            bestLoss  = float("inf")
            for epoch in range(epoch_resume, num_epochs):
                print("Epoch #{}".format(epoch), flush=True)
                
                
                train_loss, _, _, _ = run_epoch(model, train_loader, "train", optimizer, device, train_losses, val_losses,output_path)
                val_loss, yhat, y, _ = run_epoch(model, val_loader, "val", optimizer, device, train_losses, val_losses,output_path)  
                scheduler.step(val_loss)

                f.write(f"{fold},{epoch},{train_loss:.4f},{val_loss:.4f},{sklearn.metrics.r2_score(y, yhat):.4f}\n")
                scheduler.step(val_loss)

                if val_loss < bestLoss:
                    bestLoss = val_loss
                    torch.save(model.state_dict(), os.path.join(output_path, f"best_model_fold_{fold}.pt"))

            fold_stats.append({
                "fold": fold,
                "val_loss": bestLoss
            })

        pd.DataFrame(fold_stats).to_csv(os.path.join(output_path, "fold_metrics.csv"), index=False)

        best_fold = min(fold_stats, key=lambda x: x["val_loss"])['fold']
        print(f"Best fold: {best_fold}")


        # Plot fold performance
        fold_df = pd.DataFrame(fold_stats)
        plt.figure()
        plt.bar(fold_df['fold'], fold_df['val_loss'], color='skyblue')
        plt.xlabel('Fold')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss per Fold')
        plt.savefig(os.path.join(output_path, 'fold_val_loss_barplot.png'))
        plt.close()

        if run_test:
            model = build_model(model_name, input_size, device)
            model.load_state_dict(torch.load(os.path.join(output_path, f"best_model_fold_{best_fold}.pt")))
            model.eval()


            for test_split in ["test"]:
                dataset     = Dataset_lstm(root=data_dir, split=test_split)
                dataloader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda")) 
                                                                                                                 
                loss, yhat, y, filename = run_epoch(model,dataloader,test_split,None, device,train_losses=[], val_losses=[], output_path=output_path)
                # Scatter plot
                plt.figure()
                plt.scatter(y, yhat, alpha=0.6, edgecolors='k')
                plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
                plt.xlabel("True EF")
                plt.ylabel("Predicted EF")
                plt.title("True vs Predicted EF (Scatter Plot)")
                plt.savefig(os.path.join(output_path, "scatter_true_vs_pred.png"))
                plt.close()

                # Bland-Altman plot
                y, yhat = np.array(y), np.array(yhat)
                mean = (y + yhat) / 2
                diff = yhat - y
                md = np.mean(diff)
                sd = np.std(diff)
                plt.figure()
                plt.scatter(mean, diff, alpha=0.6, edgecolors='k')
                plt.axhline(md, color='gray', linestyle='--')
                plt.axhline(md + 1.96*sd, color='red', linestyle='--')
                plt.axhline(md - 1.96*sd, color='red', linestyle='--')
                plt.xlabel("Mean of True and Predicted EF")
                plt.ylabel("Difference (Pred - True)")
                plt.title("Bland-Altman Plot")
                plt.savefig(os.path.join(output_path, "bland_altman_plot.png"))
                plt.close()

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

def build_model(model_name, input_size, device):

    if model_name == "LSTM":
        model = LSTM(input_size=input_size)
    elif model_name == "RNN":
        model = RNN(input_size=input_size)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    

    return model.to(device)
if __name__ == "__main__":
    run()  

from EchoSegmentation import Echo
from utils import get_mean_and_std, latexify, savevideo,savemask, savesize, savemajoraxis

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm
import os
import math
import time
import cv2


@click.command("segmentation")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="/home/eda/Desktop/dynamic/EchoNet-Dynamic")
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.segmentation.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.segmentation.__dict__[name]))),
    default="deeplabv3_resnet50")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=True)
@click.option("--feature_extraction/", default=False)
@click.option("--num_epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-5)
@click.option("--weight_decay", type=float, default=0)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=16)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,

    model_name="deeplabv3_resnet50",
    pretrained=False,
    weights=None,

    run_test=False,
    feature_extraction=False,
    num_epochs=50,
    lr=1e-5,
    weight_decay=1e-5,
    lr_step_period=None,
    num_train_patients=None,
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
):
  
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation", "{}_{}".format(model_name, "pretrained" if pretrained else "random"))
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

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer
    optimizer= torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)

    # Compute mean and std
    mean, std = get_mean_and_std(Echo(root=data_dir, split="train"))
    kwargs = {
              "mean": mean,
              "std": std
              }
    
    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = Echo(root=data_dir, split="train", **kwargs)
    dataset["val"] = Echo(root=data_dir, split="val", **kwargs)
 
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
        results = []

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                
                loss, ed_inter, ed_union, es_inter, es_union = run_epoch(model, dataloader, phase, optimizer, device, train_losses, val_losses, results,output) 

                overall_dice = 2 *(ed_inter.sum() + es_inter.sum()) / (ed_union.sum() + ed_inter.sum() + es_union.sum() + es_inter.sum())
                ed_dice      = 2 * ed_inter.sum() / (ed_union.sum() + ed_inter.sum())
                es_dice      = 2 * es_inter.sum() / (es_union.sum() + es_inter.sum())

                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                    phase,
                                                                    loss,
                                                                    overall_dice,
                                                                    ed_dice,
                                                                    es_dice,
                                                                    time.time() - start_time,
                                                                    ed_inter.size,
                                                                    sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                    sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
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

        results.clear()
        if run_test:

            for split in ["val", "test"]:
                dataset = Echo(root=data_dir, split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda")) 
                                                                                                                 
                loss, ed_inter, ed_union, es_inter, es_union = run_epoch(model,dataloader,split,None, device,train_losses=[], val_losses=[], results=results, output=output)

                
                overall_dice = 2 *(ed_inter + es_inter) / (ed_union + ed_inter + es_union + es_inter)
                ed_dice      = 2 * ed_inter / (ed_union + ed_inter)
                es_dice      = 2 * es_inter / (es_union + es_inter)
                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, ed_dice, es_dice):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))

                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *bootstrap(np.concatenate((ed_inter, es_inter)), np.concatenate((ed_union, es_union)), dice_similarity_coefficient)))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *bootstrap(ed_inter, ed_union, dice_similarity_coefficient)))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *bootstrap(es_inter, es_union, dice_similarity_coefficient)))
                f.flush()
            
    # Saving videos with segmentations
    dataset = Echo(root=data_dir, split="test",
                                    mean=mean, std=std,  # Normalization
                                    length=None, max_length=None, period=1  # Take all frames
                                    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=num_workers, shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)

    # Save videos with segmentation


    # if feature_extraction:  # Ensure feature_extraction is defined
    #     os.makedirs(os.path.join(output, "videos"), exist_ok=True)
    #     os.makedirs(os.path.join(output, "size"), exist_ok=True)

    #     model.eval()

    #     with torch.no_grad():
    #         # Run segmentation model once for all videbbbbos
    #         results = []
    #         for (x, (filenames, large_index, small_index, _, _, _, _, _, _, _,_,_,_,_), length) in tqdm.tqdm(dataloader):
    #             y = np.concatenate([
    #                 model(x[i:(i + batch_size), :, :, :].to(device))["out"].detach().cpu().numpy()
    #                 for i in range(0, x.shape[0], batch_size)
    #             ])

    #             x = x.cpu().numpy()  # Ensure x is on CPU before converting to NumPy

    #             start = 0
    #             for (i, (filename, offset)) in enumerate(zip(filenames, length)):
    #                 logit = y[start:(start + offset), 0, :, :] 

    #                 # Compute segmentation size per frame
    #                 size = (logit > 0).sum((1, 2))

    #                 # Identify systole and diastole frames using peak detection
    #                 trim_min = sorted(size)[round(len(size) ** 0.05)]
    #                 trim_max = sorted(size)[round(len(size) ** 0.95)]
    #                 trim_range = trim_max - trim_min
    #                 systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])
    #                 diastole = set(scipy.signal.find_peaks(size, distance=20, prominence=(0.50 * trim_range))[0])

    #                 # Store results for both CSVs
    #                 results.append((filename,logit, size, systole, diastole, large_index[i], small_index[i]))

    #                 start += offset  # Move to next video
            
            # savemask(results,output,split)
            # savesize(results,output,split)
            # savemajoraxis(results,output,split)



    np.save(os.path.join(output, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(output, "val_losses.npy"), np.array(val_losses))
    print(f"Train and validation losses saved to {output}")

def run_epoch(model, dataloader, split, optimizer, device, train_losses, val_losses,results,output):

    total_loss = 0.
    n          = 0

    pos     = 0
    neg     = 0
    pos_pix = 0
    neg_pix = 0

    ed_inter = 0
    ed_union = 0
    es_inter = 0
    es_union = 0

    ed_inter_list = []
    ed_union_list = []
    es_inter_list = []
    es_union_list = []  

    train_flag = (split ==  'train')

    if split == 'train':
        model.train()
    else:
        model.eval()

    # Çıktıları kaydetmek için dizin oluştur
    output_dir = "output/segmentation_results"
    os.makedirs(output_dir, exist_ok=True)
    results.clear()
    with torch.set_grad_enabled(train_flag):
        with tqdm.tqdm(total=len(dataloader)) as pbar:

            for (_, (filename, ed_frame_idx, es_frame_idx, ed_frame_img, es_frame_img, ed_trace_mask, es_trace_mask, ef_value, edv_value, esv_value,ed_mitral_x,ed_mitral_y, es_mitral_x, es_mitral_y)) in dataloader:
    
                # Count number of pixels in and out of segmentation
                pos += (ed_trace_mask == 1).sum().item()
                pos += (es_trace_mask == 1).sum().item()
                neg += (ed_trace_mask == 0).sum().item()
                neg += (es_trace_mask == 0).sum().item()

                # Count number of pixels in and out of segmentation
                pos_pix += (ed_trace_mask == 1).sum(0).to("cpu").detach().numpy()
                pos_pix += (es_trace_mask == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (ed_trace_mask == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (es_trace_mask == 0).sum(0).to("cpu").detach().numpy()


               # Prediction for diastolic frame 
                ed_frame_img = ed_frame_img.to(device)
                ed_trace_mask = ed_trace_mask.to(device)

                y_ed = model(ed_frame_img)["out"] 

                loss_ed = torch.nn.functional.binary_cross_entropy_with_logits(y_ed[:,0,:,:], ed_trace_mask, reduction="sum")

                ed_inter += np.logical_and(y_ed[:,0,:,:].detach().cpu().numpy() > 0.,
                                              ed_trace_mask[:,:,:].detach().cpu().numpy() > 0.  ).sum() 
                ed_union += np.logical_or(y_ed[:,0,:,:].detach().cpu().numpy() > 0.,
                                              ed_trace_mask[:,:,:].detach().cpu().numpy() > 0.  ).sum() 
                ed_inter_list.extend(np.logical_and(y_ed[:,0,:,:].detach().cpu().numpy() > 0.,
                                              ed_trace_mask[:,:,:].detach().cpu().numpy() > 0.  ).sum((1,2)))
                ed_union_list.extend(np.logical_or(y_ed[:,0,:,:].detach().cpu().numpy() > 0.,
                                              ed_trace_mask[:,:,:].detach().cpu().numpy() > 0.  ).sum((1,2)))
                
                # Prediction for systolic frame 
                es_frame_img = es_frame_img.to(device)
                es_trace_mask = es_trace_mask.to(device)

                y_es = model(es_frame_img)["out"] 

                loss_es = torch.nn.functional.binary_cross_entropy_with_logits(y_es[:,0,:,:], es_trace_mask, reduction="sum")

                es_inter += np.logical_and(y_es[:,0,:,:].detach().cpu().numpy() > 0.,
                                              es_trace_mask[:,:,:].detach().cpu().numpy() > 0.  ).sum() 
                
                es_union += np.logical_or(y_es[:,0,:,:].detach().cpu().numpy() > 0.,
                                              es_trace_mask[:,:,:].detach().cpu().numpy() > 0.  ).sum() 
                
                es_inter_list.extend(np.logical_and(y_es[:,0,:,:].detach().cpu().numpy() > 0.,
                                              es_trace_mask[:,:,:].detach().cpu().numpy() > 0.  ).sum((1,2)))
                
                es_union_list.extend(np.logical_or(y_es[:,0,:,:].detach().cpu().numpy() > 0.,
                                              es_trace_mask[:,:,:].detach().cpu().numpy() > 0.  ).sum((1,2)))
                
                
                # **Segmentasyon Sonucu ve Major Axis Bilgisini Kaydet**
                for i, filename in enumerate(filename):
                    # ED (End-Diastole) Maskesi
                    logit_ed = y_ed[i, 0, :, :].detach().cpu().numpy()  # ED için logit maskesi
                    size_ed = (logit_ed > 0).sum()  # ED maskesindeki toplam piksel sayısı
                    ed_idx = int(ed_frame_idx[i])  # ED için frame indexi

                    # ES (End-Systole) Maskesi
                    logit_es = y_es[i, 0, :, :].detach().cpu().numpy()  # ES için logit maskesi
                    size_es = (logit_es > 0).sum()  # ES maskesindeki toplam piksel sayısı
                    es_idx = int(es_frame_idx[i])  # ES için frame indexi

                    # Sonuçları kaydet
                    results.append((filename, "ED", logit_ed, size_ed, ed_idx,"ES", logit_es, size_es, es_idx))
                
            

                loss = (loss_ed + loss_es) / 2
                # Graidient for training
                if train_flag:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Accumulate losses and compute baselines
                total_loss += loss.item()
                n          += ed_trace_mask.size(0)
                p           = pos / (pos + neg)
                p_pix       = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                # Show info on process bar
                pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(total_loss / n / 112 / 112, loss.item() / ed_trace_mask.size(0) / 112 / 112, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 2 * ed_inter / (ed_union + ed_inter), 2 * es_inter / (es_union + es_inter)))
                pbar.update()

            


            if (split == "train"):
                train_losses.append(total_loss/ n / 112 / 112)
            elif split == "val":
                val_losses.append(total_loss/ n / 112 / 112)
            else:
                pass

    ed_inter_list = np.array(ed_inter_list)
    ed_union_list = np.array(ed_union_list)
    es_inter_list = np.array(es_inter_list)
    es_union_list = np.array(es_union_list)

    savesize(results,output,split)
    savemajoraxis(results,output,split)
    savemask(results,output,split)

    return (total_loss / n / 112 / 112,
            ed_inter_list,
            ed_union_list,
            es_inter_list,
            es_union_list,
            )

def bootstrap(intersection, union, metric, num_samples=1000, confidence=0.95):
    """Bootstrap method for computing confidence intervals of a metric."""
    metrics = []
    n = len(intersection)
    for _ in range(num_samples):
        idx = np.random.choice(n, n, replace=True)
        metrics.append(metric(intersection[idx], union[idx]))
    metrics = np.array(metrics)
    lower = np.percentile(metrics, (1 - confidence) / 2 * 100)
    upper = np.percentile(metrics, (1 + confidence) / 2 * 100)
    return np.mean(metrics), lower, upper

def dice_similarity_coefficient(intersection, union):
    """
    Computes the Dice Similarity Coefficient (DSC).
    
    Args:
        intersection (np.ndarray or float): Intersection of two segmentations.
        union (np.ndarray or float): Union of two segmentations.
    
    Returns:
        float: Dice coefficient score.
    """
    intersection = np.array(intersection, dtype=np.float64)
    union = np.array(union, dtype=np.float64)
    
    # Union + Intersection sıfır olmamalı (bölme hatasını önlemek için)
    if np.sum(union) + np.sum(intersection) == 0:
        return 1.0  # Eğer tamamen boşsa, segmentasyon tam örtüşüyor demektir
    
    return (2 * np.sum(intersection)) / (np.sum(union) + np.sum(intersection))

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

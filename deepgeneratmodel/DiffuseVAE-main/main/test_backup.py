import os

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm


from datasets.latent import UncondLatentDataset
from models.vae import VAE
from models.diffusion import DDPM, DDPMv2, DDPMWrapper, SuperResModel, UNetModel
from util import configure_device, get_dataset, save_as_images


@click.group()
def cli():
    pass


def compare_samples(gen, refined, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].imshow(gen.permute(1, 2, 0))
    ax[0].set_title("VAE Reconstruction")
    ax[0].axis("off")

    ax[1].imshow(refined.permute(1, 2, 0))
    ax[1].set_title("Refined Image")
    ax[1].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def plot_interpolations(interpolations, save_path=None, figsize=(10, 5)):
    N = len(interpolations)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)

    for i, inter in enumerate(interpolations):
        ax[i].imshow(inter.permute(1, 2, 0))
        ax[i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def compare_interpolations(
    interpolations_1, interpolations_2, save_path=None, figsize=(10, 2)
):
    assert len(interpolations_1) == len(interpolations_2)
    N = len(interpolations_1)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=2, ncols=N, figsize=figsize)

    for i, (inter_1, inter_2) in enumerate(zip(interpolations_1, interpolations_2)):
        ax[0, i].imshow(inter_1.permute(1, 2, 0))
        ax[0, i].axis("off")

        ax[1, i].imshow(inter_2.permute(1, 2, 0))
        ax[1, i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


# TODO: Upgrade the commands in this script to use hydra config
# and support Multi-GPU inference
@cli.command()
@click.argument("chkpt-path")
@click.argument("root")
@click.option("--device", default="gpu:0")
@click.option("--dataset", default="cifar10")
@click.option("--image-size", default=34)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default="/home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/final2")
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
def reconstruct(
    chkpt_path,
    root,
    device="gpu:0",
    dataset="cifar10",
    image_size=32,
    num_samples=-1,
    save_path=os.getcwd(),
    write_mode="image",
):
    dev = configure_device(device)
    if num_samples == 0:
        raise ValueError(f"`--num-samples` can take value=-1 or > 0")

    # Dataset
    dataset = get_dataset(dataset, root, image_size, norm=False, flip=False)

    print("leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    # Loader
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    # Load checkpoint and extract hyperparameters
    checkpoint = torch.load("/home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/final/result2/checkpoints/ddpmv2--epoch=999-loss=0.0328.ckpt", map_location="cuda:0")
    #hyperparams = checkpoint["hyper_parameters"]
    chkpt_path = "/home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/final/result2/checkpoints/ddpmv2--epoch=999-loss=0.0328.ckpt"
    print("leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    
    
    ddpm = DDPMWrapper.load_from_checkpoint(chkpt_path, input_res=image_size).to("cuda:0")
    print("leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    ddpm.eval()
    print("leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")

    sample_list = []
    img_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = ddpm.forward_recons(batch)

        if count + recons.size(0) >= num_samples and num_samples != -1:
            img_list.append(batch[:num_samples, :, :, :].cpu())
            sample_list.append(recons[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        img_list.append(batch.cpu())
        count += recons.size(0)
    print("leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    cat_img = torch.cat(img_list, dim=0)
    cat_sample = torch.cat(sample_list, dim=0)
    print("leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)
    print("leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    if write_mode == "image":
        save_as_images(
            cat_sample,
            file_name=os.path.join(save_path, "ddpm"),
            denorm=False,
        )
        save_as_images(
            cat_img,
            file_name=os.path.join(save_path, "orig"),
            denorm=False,
        )
    else:
        np.save(os.path.join(save_path, "images.npy"), cat_img.numpy())
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())
    print(save_path)
    print("leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")

@cli.command()
@click.argument("z-dim", type=int)
@click.argument("chkpt-path")
@click.option("--seed", default=0, type=int)
@click.option("--device", default="gpu:0")
@click.option("--image-size", default=34)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default="/home/dsail/leejonghyun/deepgeneratmodel/DiffuseVAE-main/final2")
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
def sample(
    z_dim,
    chkpt_path,
    seed=0,
    device="gpu:0",
    image_size=34,
    num_samples=1,
    save_path=os.getcwd(),
    write_mode="image",
):
    seed_everything(seed)
    dev = configure_device(device)
    if num_samples <= 0:
        raise ValueError(f"`--num-samples` can take values > 0")

    dataset = UncondLatentDataset((num_samples, z_dim, 1, 1))

    # Loader
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    vae.eval()

    sample_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = vae.forward(batch)

        if count + recons.size(0) >= num_samples and num_samples != -1:
            sample_list.append(recons[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        count += recons.size(0)

    cat_sample = torch.cat(sample_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    if write_mode == "image":
        save_as_images(
            cat_sample,
            file_name=os.path.join(save_path, "ddpm"),
            denorm=False,
        )
    else:
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())


if __name__ == "__main__":
    cli()

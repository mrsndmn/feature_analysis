import argparse
import sys
import os
from math import ceil

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import AutoImageProcessor

from tqdm import tqdm

from model import R
from scheduler import CycleScheduler
import distributed as dist

##########################################################################################################
##########################################################################################################
##########################################################################################################
def log_metrics(mse_list, cos_list, lr_list, epoch_list, i_list, mse, cos, lr, epoch, i):
    mse_list.append(mse)
    cos_list.append(cos)
    lr_list.append(lr)
    epoch_list.append(epoch)
    i_list.append(i)


def save_metrics(metric_dict, values_save_dir, plots_save_dir, save_name_preffix, block_size=6):
    ncols = 1
    nrows = len(metric_dict)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * block_size * 4, nrows * block_size))

    for i, (metric_name, metric_list) in enumerate(metric_dict.items()):
        name = save_name_preffix + f'_{metric_name}'

        values_dir = os.path.join(values_save_dir, 'values')
        os.makedirs(values_dir, mode=0o777, exist_ok=True)
        np.save(os.path.join(values_dir, name + '.npy'), np.asarray(metric_list))

        plot_dir = os.path.join(plots_save_dir, 'plots')
        os.makedirs(plot_dir, mode=0o777, exist_ok=True)
        axes[i].plot(metric_list, label=metric_name)

        axes[i].legend()
    plt.savefig(os.path.join(plot_dir, name + '.png'))


##########################################################################################################
##########################################################################################################
##########################################################################################################
def draw_in_out_cos(images, 
                    reconstructed_images,
                    in_out_cos_list,
                    examples=5, block_size=6, save_dir=None, save_name=None):
    examples = min(examples, len(images))
    ncols = 4
    nrows = examples
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * block_size, nrows * block_size))
    
    for i in range(examples):
        ###############################################################################
        image = images[i]
        axes[i, 0].imshow(image)

        reconstructed_image = reconstructed_images[i].cpu().permute(1, 2, 0)
        axes[i, 1].imshow((reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min()))

        ###############################################################################
        in_out_cos = in_out_cos_list[i].cpu()
        axes[i, 2].imshow(in_out_cos)

        ###############################################################################
        flatten_in_out_cos = in_out_cos.view(-1)
        axes[i, 3].plot(flatten_in_out_cos, alpha=0.5, label='in_out_cos')
        axes[i, 3].legend()

        if i == 0:
            axes[i, 0].set_title('image')
            axes[i, 1].set_title(f'reconstructed_image')
            axes[i, 2].set_title(f'cosine similarity between\ninput and reconstruction')
            axes[i, 3].set_title(f'cosine similarity between\ninput and reconstruction\n(flatten version)')

        axes[i, 0].set_axis_off()
        axes[i, 1].set_axis_off()
        axes[i, 2].set_axis_off()

    if save_dir and save_name:
        plt.savefig(os.path.join(save_dir, save_name))
        

##########################################################################################################
##########################################################################################################
##########################################################################################################        
def draw_input_norm(inputs, outputs, examples=5, block_size=6, save_dir=None, save_name=None):
    ncols = 1
    examples = min(examples, len(inputs))
    nrows = examples
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * block_size * 4, nrows * block_size))
    
    for i in range(examples):
        input_norm = inputs[i].cpu().norm(dim=0).view(-1)
        output_norm = outputs[i].cpu().norm(dim=0).view(-1)

        axes[i].plot(input_norm, alpha=0.5, label='input')
        axes[i].plot(output_norm, alpha=0.5, label='output')
        axes[i].legend()
        
        if i == 0:
            axes[i].set_title(f'norms of input and reconstructed features')

    if save_dir and save_name:
        plt.savefig(os.path.join(save_dir, save_name))


##########################################################################################################
##########################################################################################################
##########################################################################################################
def val(model, eval_dataset, batch_size, device):
    model.eval()

    images = []
    interpolated_images = []
    sample = []
    for j, (im, inter_im, fe) in enumerate(eval_dataset):
        if j > batch_size: break
        images.append(im)
        interpolated_images.append(inter_im.to(device)[None])
        sample.append(fe.to(device)[None])
    interpolated_images = torch.cat(interpolated_images, dim=0)
    sample = torch.cat(sample, dim=0)
    sample /= sample.norm(dim=1, keepdim=True)

    with torch.no_grad():
        recovered_image, input, check_dict = model(sample, interpolated_images)
        loss = model.loss_function(interpolated_images, recovered_image)
    
    images = interpolated_images.permute(0, 2, 3, 1).cpu()
    images = (images - images.min()) / (images.max() - images.min())
    return images, interpolated_images, recovered_image, input, check_dict, loss


def train(epoch, loader, model, optimizer, scheduler, device, eval_dataset, args, train_tuple, val_tuple):
    if dist.is_primary():
        loader = tqdm(loader)

    train_mse_list, train_cos_list, train_lr_list, train_epoch_list, train_i_list = train_tuple
    val_mse_list, val_cos_list, val_lr_list, val_epoch_list, val_i_list = val_tuple
    for i, (im, fe) in enumerate(loader):
        model.zero_grad()
        fe = fe.to(device)
        fe /= fe.norm(dim=1, keepdim=True)
        im = im.to(device)

        recovered_image, _, _ = model(fe, im)
        loss = model.loss_function(im, recovered_image)
        loss['loss'].backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        #########################################################
        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]
            loader.set_description(
                (
                    f"epoch: {epoch + 1}; "
                    f"mse: {loss['mse'].item():.5f}; "
                    f"cos: {loss['cos'].item():.5f}; "
                    f"lr: {lr:.5f}"
                )
            )
                        
            if i % 100 == 0:
                log_metrics(train_mse_list, train_cos_list, train_lr_list, train_epoch_list, train_i_list, 
                            loss['mse'].item(), loss['cos'].item(), lr, epoch, i)

                train_values_save_dir=os.path.join(args.sample_path, f'num-hidden-layers-{model.config.num_hidden_layers}', 'metrics')
                train_plots_save_dir=os.path.join(args.sample_path, f'num-hidden-layers-{model.config.num_hidden_layers}', 'metrics')
                os.makedirs(train_values_save_dir, mode=0o777, exist_ok=True)
                os.makedirs(train_plots_save_dir, mode=0o777, exist_ok=True)
                save_metrics(
                    {
                        'mse': train_mse_list,
                        'cos': train_cos_list,
                        'lr': train_lr_list,
                        'epoch': train_epoch_list,
                        'i': train_i_list
                    }, 
                    values_save_dir=train_values_save_dir, 
                    plots_save_dir=train_plots_save_dir, 
                    save_name_preffix=f'train')
                

            if i % 100 == 0:
                model.eval()

                val_images, interpolated_images, val_out, val_input, val_check_dict, val_loss = val(model, eval_dataset, fe.shape[0], device)
                log_metrics(val_mse_list, val_cos_list, val_lr_list, val_epoch_list, val_i_list, 
                            val_loss['mse'].item(), val_loss['cos'].item(), lr, epoch, i)
                
                val_values_save_dir=os.path.join(args.sample_path, f'num-hidden-layers-{model.config.num_hidden_layers}', 'metrics')
                val_plots_save_dir=os.path.join(args.sample_path, f'num-hidden-layers-{model.config.num_hidden_layers}', 'metrics')
                os.makedirs(val_values_save_dir, mode=0o777, exist_ok=True)
                os.makedirs(val_plots_save_dir, mode=0o777, exist_ok=True)
                save_metrics(
                    {
                        'mse': val_mse_list, 
                        'cos': val_cos_list, 
                        'lr': val_lr_list, 
                        'epoch': val_epoch_list, 
                        'i': val_i_list
                    }, 
                    values_save_dir=val_values_save_dir,
                    plots_save_dir=val_plots_save_dir,
                    save_name_preffix=f'val')
                
                sample_save_dir = os.path.join(args.sample_path, f'num-hidden-layers-{model.config.num_hidden_layers}')
                os.makedirs(sample_save_dir, mode=0o777, exist_ok=True)
                
                in_out_cos_save_name = f'{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_in-out-cos.png'
                draw_in_out_cos(val_images, val_check_dict['transposed_input'], val_check_dict['in_out_cos'],
                                save_dir=sample_save_dir, save_name=in_out_cos_save_name)
                
                draw_input_norm_name = f'{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_input-reconstruction-norm.png'
                draw_input_norm(interpolated_images, val_out, save_dir=sample_save_dir, save_name=draw_input_norm_name)
            
                model.train()


class FeaturesDataset(Dataset):
    def __init__(self, features_json_path, mode='train', side_size=None, image_processor=None):
        with open(features_json_path, 'r') as map_file:
            self.map = list(json.load(map_file).items())
        self.mode = mode

        self.side_size = side_size
        self.image_processor = image_processor

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        image_path, feature_path = self.map[idx]
        feature = torch.load(feature_path, map_location='cpu').to(torch.float32)
        
        image = Image.open(image_path).convert('RGB')
        
        processed_image = self.image_processor(image)['pixel_values'][0]
        processed_image = torch.tensor(processed_image)
        if processed_image.shape[-1] != self.side_size:
            interpolated_image = F.interpolate(processed_image[None], size=(self.side_size, self.side_size), mode='bilinear', align_corners=False)[0]
        else:
            interpolated_image = processed_image
        
        if self.mode == 'train':
            return interpolated_image, feature.permute(2, 0, 1) # tensor[3 x 224 x 224], tensor[768 x 14 x 14]
        else:
            return np.asarray(image), interpolated_image, feature.permute(2, 0, 1) # array(427, 640, 3), tensor[3 x 224 x 224], tensor[768 x 14 x 14]


def calculat_num_parameters(model):
    params = 0
    for n, p in model.named_parameters():
        print(f'{n}: {np.prod(p.shape)}')
        params += np.prod(p.shape)
    print(f'NUM PARAMETERS: {params}')


def main(args):
    device = args.device

    args.distributed = dist.get_world_size() > 1
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(current_dir, '..', 'feature_extractor_weights')
    args.sample_path = os.path.join(current_dir, 'samples')
    args.save_path = os.path.join(current_dir, 'checkpoint')
    vision_model_name = args.vision_model_name

    image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
    side_size = int(vision_model_name.split('-')[-1])
    features_json_path = os.path.join(current_dir, f'../generated_datasets/{"-".join(vision_model_name.split("/"))}/map_train.json')
    val_features_json_path = os.path.join(current_dir, f'../generated_datasets/{"-".join(vision_model_name.split("/"))}/map_val.json')

    dataset = FeaturesDataset(features_json_path, mode='train', image_processor=image_processor, side_size=side_size)
    eval_dataset = FeaturesDataset(val_features_json_path, mode='val', image_processor=image_processor, side_size=side_size)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(dataset, batch_size=32 // args.n_gpu, sampler=sampler, num_workers=16)
    
    model = R(vision_model_name, weights_dir).to(device)
    calculat_num_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )


    train_tuple = [], [], [], [], []
    val_tuple = [], [], [], [], []
    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, eval_dataset, args, train_tuple, val_tuple)

        if i % 1 == 0:
            save_path = os.path.join(args.save_path, f'num-hidden-layers-{model.config.num_hidden_layers}')
            os.makedirs(save_path, mode=0o777, exist_ok=True)
            if dist.is_primary():
                torch.save(model.state_dict(), os.path.join(save_path, f'vqvae-{str(i + 1).zfill(3)}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    # training parameters
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str, default='cycle')
    parser.add_argument("--device", type=str, default='cuda')
    
    # model parameters
    parser.add_argument("--vision_model_name", type=str)

    args = parser.parse_args()

    print(args)
    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))

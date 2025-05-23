import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from transformers import SiglipVisionConfig, AutoImageProcessor, SiglipVisionModel

valid_model_name_list = [
    'google/siglip-base-patch16-224', 
    'google/siglip-base-patch16-256', 
    'google/siglip-base-patch16-384', 
    'google/siglip-base-patch16-512',

    'google/siglip2-base-patch16-224',
    'google/siglip2-base-patch16-256',
    'google/siglip2-base-patch16-384', 
    'google/siglip2-base-patch16-512'
]

class SigLipVisionTower(nn.Module):
    def __init__(self, vision_model_name, weights_dir):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_model_name
        self.weights_dir = weights_dir
        self.processor_path = 'facebook/dinov2-base'
        self.select_layer = -2
        self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name, cache_dir=self.weights_dir)

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        self.image_processor = AutoImageProcessor.from_pretrained(self.processor_path, 
                                                                  cache_dir=self.weights_dir, 
                                                                  crop_size={"height": self.cfg_only.image_size, 
                                                                             "width": self.cfg_only.image_size},
                                                                  image_mean=[0.5, 0.5, 0.5], 
                                                                  image_std=[0.5, 0.5, 0.5], 
                                                                  size={"shortest_edge": self.cfg_only.image_size})
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, 
                                                              cache_dir=self.weights_dir, 
                                                              device_map=device_map)

        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.eval()

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[self.select_layer].to(image.dtype)
                image_features.append(image_feature)
        return image_features

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
    

@torch.no_grad()
def calc_feature(inputs, num_patches_per_side):
    features = vision_tower([inputs[0]])

    features = features[0].reshape(num_patches_per_side, num_patches_per_side, 768)
    features = features.permute(2, 0, 1)[None]
    features = features / features.norm(dim=1, keepdim=True)
    return inputs, features


@torch.no_grad()
def interpolate(im, Fe, model, device):
    processed_image = torch.tensor(im).to(device)
    processed_Fe = Fe.to(device)
    reconstructed_image, processed_Fe, check_dict = model.forward(processed_Fe, processed_image)
    return processed_image, reconstructed_image, processed_Fe, check_dict


def from_1_to_255(image, image_mean, image_std):
    image_unnormed = np.asarray(image) * image_std[:, None, None] + image_mean[:, None, None]
    image_01 = image_unnormed.clip(0, 1)
    image_255 = image_01 * 255
    image_255_uint = np.array(image_255, dtype=np.uint8)
    return image_255_uint


@torch.no_grad()
def clip_image_similarity(image1, image2, clip_processor, clip_model) -> float:
    image1, image2 = image1.convert('L').convert('RGB'), image2.convert('L').convert('RGB')
    inputs = clip_processor(images=[image1, image2], return_tensors="pt", padding=True)
    outputs = clip_model.get_image_features(**inputs)
    embeddings = torch.nn.functional.normalize(outputs, p=2, dim=1)
    similarity = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    return similarity.item()


def calculate_score(clip_vision_model_name, cache_dir, vision_tower, interpolated_list, reconstruction_list, vision_model_name_for_path):
    clip_model = AutoModel.from_pretrained(clip_vision_model_name, cache_dir=cache_dir)
    clip_processor = AutoProcessor.from_pretrained(clip_vision_model_name, cache_dir=cache_dir)
    clip_vision_model_name_for_file = '-'.join(clip_vision_model_name.split('/'))

    ###########################################################

    image_mean = np.array(vision_tower.image_processor.image_mean)
    image_std = np.array(vision_tower.image_processor.image_std)
    clip_sim_list = []
    for i in tqdm(range(len(interpolated_list))):
        real = from_1_to_255(interpolated_list[i][0].cpu(), image_mean, image_std)
        rec = from_1_to_255(reconstruction_list[i][0].cpu(), image_mean, image_std)
        real = Image.fromarray(real.transpose(1, 2, 0))
        rec = Image.fromarray(rec.transpose(1, 2, 0))

        clip_sim_value = clip_image_similarity(real, rec, clip_processor, clip_model)
        clip_sim_list.append(clip_sim_value)


    #########################################################################################
    #########################################################################################
    import os
    with open(os.path.join(save_path, f'metrics_clip-{B}__{clip_vision_model_name_for_file}.json'), 'w') as f: #__{vision_model_name_for_path}
        json.dump({'clip_sim': clip_sim_list}, f)

    ncols, nrows, scale = 1, len(images_pathes_list), 5
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * scale, nrows * scale))

    for j in range(len(clip_sim_list)):
        real = from_1_to_255(interpolated_list[j][0].cpu(), image_mean, image_std)
        rec = from_1_to_255(reconstruction_list[j][0].cpu(), image_mean, image_std)

        axes[j].imshow(rec.transpose(1, 2, 0))
        axes[j].set_axis_off()

    plt.savefig(f'{os.path.join(save_path, f"first_10.png")}') #__{clip_vision_model_name_for_file}__{vision_model_name_for_path}


if __name__ == "__main__":
    import os
    import json
    import argparse
    import matplotlib.pyplot as plt

    from PIL import Image
    from tqdm import tqdm
    from model import R

    from transformers import AutoModel, AutoProcessor
    from torch.nn.functional import cosine_similarity

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--vision_model_name', help='')
    parser.add_argument('--max_count', type=int, help='')
    parser.add_argument('--reconstructor_weights_path', help='')
    args = parser.parse_args()

    #################################### pathes and names ####################################
    device = 'cuda:0'
    vision_model_name = args.vision_model_name
    vision_model_name_for_path = '-'.join(vision_model_name.split('/'))
    feature_extractor_weights_dir = os.path.join(current_dir, '../../feature_extractor_weights')
    reconstructor_weights_path = args.reconstructor_weights_path
    json_path = os.path.join(current_dir, f'../../generated_datasets/{"-".join(vision_model_name.split("/"))}/map_val.json')
    save_path = os.path.join(current_dir, 'results', vision_model_name_for_path)
    os.makedirs(save_path, 0o777, exist_ok=True)

    #################################### collecting images ####################################
    with open(json_path, 'r') as json_file:
        json_dict = json.load(json_file)

    B = args.max_count
    images_pathes_list = []
    for i, (im_path, feature_path) in enumerate(json_dict.items()):
        if i == B: break
        images_pathes_list.append(im_path)
    print('--------------> path reading done. <--------------')

    images_list = [Image.open(p).convert('RGB') for p in tqdm(images_pathes_list)]
    print('images_list done.')

    #################################### loading models ####################################
    vision_tower = SigLipVisionTower(vision_model_name, feature_extractor_weights_dir)
    vision_tower.load_model(device_map=device)
    print('--------------> vision model loading done. <--------------')

    model = R(vision_model_name, feature_extractor_weights_dir).to(device)
    model.load_state_dict(torch.load(reconstructor_weights_path, weights_only=True), strict=True)
    model.eval()
    print('--------------> reconstructor loading done. <--------------')
    
    #################################### compute reconsturctions ####################################
    N = 1
    reconstruction_list = []
    interpolated_list = []
    for im in tqdm(images_list):
        inputs = vision_tower.image_processor(im, return_tensors="pt")['pixel_values'].to(device)
        recs = torch.clone(inputs)
        for _ in range(N):
            _, features = calc_feature(recs, vision_tower.num_patches_per_side)
            _, recs, _, _ = interpolate(recs, features, model, device)

        inputs, recs = inputs.cpu(), recs.cpu()
        inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
        recs = F.interpolate(recs, size=(224, 224), mode='bilinear', align_corners=False)

        reconstruction_list.append(recs)
        interpolated_list.append(inputs)
    print('--------------> reconstructions computing done. <--------------')

    ###########################################################
    calculate_score('openai/clip-vit-large-patch14', 
                    feature_extractor_weights_dir, vision_tower, 
                    interpolated_list, reconstruction_list, 
                    vision_model_name_for_path)
    calculate_score('google/siglip2-large-patch16-256', 
                    feature_extractor_weights_dir, vision_tower, 
                    interpolated_list, reconstruction_list, 
                    vision_model_name_for_path)
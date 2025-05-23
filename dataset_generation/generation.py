import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipImageProcessor, SiglipVisionModel

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
        self.select_layer = -2
        self.cfg_only = SiglipVisionConfig(self.vision_tower_name, cache_dir=self.weights_dir)

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name, cache_dir=self.weights_dir)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, cache_dir=self.weights_dir, device_map=device_map)

        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.eval()

        self.is_loaded = True

    def forward(self, images):
        image_forward_out = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_feature = image_forward_out.hidden_states[self.select_layer].to(images.dtype)
        return image_feature

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


if __name__ == "__main__":
    import os
    import json
    import argparse
    from tqdm import tqdm
    from PIL import Image

    current_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--vision_model_name', help='')
    parser.add_argument('--coco_images_path', help='')
    parser.add_argument('--split', help='')
    parser.add_argument('--max_count', type=int, help='')
    parser.add_argument('--image_types', type=str, nargs='+', default=['png', 'jpg', 'jpeg'], help='')
    args = parser.parse_args()

    if args.vision_model_name not in valid_model_name_list:
        raise Exception(f'vision_model_name should be in {valid_model_name_list}')
    if not os.path.isdir(args.coco_images_path):
        raise Exception(f'coco_images_path should be a dir with images')
    if args.split not in ['train', 'val']:
        raise Exception(f'split should be in ["train", "val"]')

    #################################### pathes and names ####################################
    device = 'cuda:0'
    vision_model_name = args.vision_model_name
    vision_model_name_for_path = '-'.join(vision_model_name.split('/'))
    weights_dir = os.path.join(current_dir, '..', 'feature_extractor_weights')
    datasets_dir = os.path.join(current_dir, '..', 'generated_datasets')
    os.makedirs(weights_dir, mode=0o777, exist_ok=True)
    os.makedirs(datasets_dir, mode=0o777, exist_ok=True)

    batch_size = 2
    mode = args.split
    max_images = args.max_count
    images_dir = args.coco_images_path
    features_dir = f'{datasets_dir}/{vision_model_name_for_path}/tensors_{mode}'
    features_json = f'{datasets_dir}/{vision_model_name_for_path}/map_{mode}.json'
    image_types = [args.image_types] if type(args.image_types) == str else args.image_types
    image_names = [
        n for n in os.listdir(images_dir) 
        if n.split('.')[-1].lower() in image_types
    ][:max_images]


    os.makedirs(features_dir, mode=0o777, exist_ok=True)
    print('----------> A directory for the dataset has been created. <----------')


    #################################### dataset generation ####################################
    vision_tower = SigLipVisionTower(vision_model_name, weights_dir)
    vision_tower.load_model(device_map=device)
    print('----------> The model has been downloaded. <----------')

    image_feature_map = {}
    with torch.inference_mode(), torch.no_grad():
        for i in tqdm(range(0, len(image_names), batch_size)):
            batch_image_names = image_names[i:i+batch_size]
            batch_processed_images = []
            batch_image_paths = []
            batch_feature_paths = []

            for image_name in batch_image_names:
                feature_name = image_name.split('.')[0]
                feature_path = os.path.join(features_dir, f'{feature_name}.pt')
                image_path = os.path.join(images_dir, image_name)

                try:
                    example = Image.open(image_path).convert('RGB')
                    processed_image = vision_tower.image_processor(example, return_tensors='pt')['pixel_values'][0]
                    batch_processed_images.append(processed_image)
                    batch_image_paths.append(image_path)
                    batch_feature_paths.append(feature_path)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue

            if not batch_processed_images:
                continue

            images_batch = torch.stack(batch_processed_images).to(device)

            batch_features: torch.Tensor = vision_tower.forward(images_batch)
            batch_features = batch_features.to(torch.bfloat16)
            assert batch_features.dtype == torch.bfloat16
            assert batch_features.shape[0] == len(batch_image_paths)

            for idx in range(batch_features.shape[0]):
                image_path = batch_image_paths[idx]
                feature_path = batch_feature_paths[idx]
                image_feature_map[image_path] = feature_path
                
                features = batch_features[idx]

                features_reshaped = features.unflatten(0, [vision_tower.num_patches_per_side, vision_tower.num_patches_per_side])
                features_reshaped = features_reshaped.clone()
                torch.save(features_reshaped, feature_path)

        with open(features_json, 'w') as config:
            json.dump(image_feature_map, config)
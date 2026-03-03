import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from utils.align_data import pre_process_images
from scripts.run import DeepProtect
from IPython.display import display
import matplotlib.pyplot as plt
from scripts.latent_editor_wrapper import LatentEditorWrapper

image_dir_name = 'image'

## If set to true download desired image from given url. If set to False, assumes you have uploaded personal image to
## 'image_original' dir
use_image_online = True
img_path = ""
image_name = img_path.split('/')[-1].split('.')[0]
modelASCII = 'KTBIKGJNKQJE'
use_multi_id_training = False
global_config.device = 'cuda'
paths_config.e4e = 'pretrained_models/e4e_ffhq_encode.pt'
paths_config.embedding_base_dir = 'embeddings'
paths_config.input_data_id = 'barcelona'
paths_config.input_data_path = f'{image_dir_name}_processed'
paths_config.stylegan2_ada_ffhq = 'pretrained_models/ffhq.pkl'
paths_config.checkpoints_dir = 'checkpoints'
paths_config.style_clip_pretrained_mappers = 'pretrained_models'
hyperparameters.use_locality_regularization = False

os.makedirs(f'./{image_dir_name}_original', exist_ok=True)
os.makedirs(f'./{image_dir_name}_processed', exist_ok=True)
print(os.getcwd())
aligned_image = Image.open(img_path)
aligned_image.resize((256,256))
model_id = DeepProtect(use_wandb=False, use_multi_id_training=use_multi_id_training)

def display_alongside_source_image(images): 
    res = np.concatenate([np.array(image) for image in images], axis=1) 
    return Image.fromarray(res) 

def load_generators(model_id, image_name,modelASCII):
  with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
    old_G = pickle.load(f)['G_ema'].cuda()
  with open(f'{paths_config.checkpoints_dir}/model_{modelASCII}_{image_name}.pt', 'rb') as f_new: 
    new_G = torch.load(f_new).cuda()

  return old_G, new_G

def plot_syn_images(syn_images,image_name,aligned_image): 
  for i, img in enumerate(syn_images): 
      img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
      plt.axis('off') 
      resized_image = Image.fromarray(img,mode='RGB').resize((256,256))
      resized_image.save(f"result/{image_name}_{i}.jpg")

      del img 
      del resized_image 
      torch.cuda.empty_cache()
  aligned_image.resize((256,256)).save(f"result/{image_name}.jpg")

generator_type = paths_config.multi_id_model_type if use_multi_id_training else image_name
old_G, new_G = load_generators(model_id, generator_type,modelASCII)

w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
w_o = torch.load(f'{embedding_dir}/0.pt')


old_image = old_G.synthesis(w_o, noise_mode='const', force_fp32 = True)
new_image = new_G.synthesis(w_o, noise_mode='const', force_fp32 = True)

print(image_name)
plot_syn_images([old_image, new_image],image_name,aligned_image)
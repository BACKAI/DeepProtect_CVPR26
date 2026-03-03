import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w,gen_img
import h5py
import numpy as np
import wandb
import torch.nn.functional as F
from PIL import Image
from Gen_Adv_clip import adv_wm

class DeepProtect(BaseCoach):
    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
    
    
    def sort_img(self, qf, gf):
        query = qf.view(-1,1)
        score = torch.mm(gf,query)
        score = score.squeeze(1).cpu()
        score = score.detach().numpy()
        # predict index
        index = np.argsort(score)  #from small to large
        index = index[::-1]
        return index, score[index]

    def train(self):
        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.results_keyword}', exist_ok=True)

        use_ball_holder = True
        FRmodel = torch.load("./pretrained_model/arcface_model/arcface_checkpoint.tar")
        FRmodel = FRmodel.eval()
        FRmodel = FRmodel.to(global_config.device)

        for p in FRmodel.parameters():
            p.requires_grad = False

        file_path = './VGGface2_hq_Gallery_e4e.h5'
        with h5py.File(file_path, 'r') as f:
            gallery_feature = torch.FloatTensor(f['gallery_e4e'][:])
            gallery_name = [name.decode('utf-8') for name in f['gallery_name'][:]]
            gallery_feature = gallery_feature.cuda()
            lcnorm = torch.norm(gallery_feature, p=2, dim=2, keepdim=True)
            gallery_feature_norm = gallery_feature.div(lcnorm.expand_as(gallery_feature))
        
        for fname, image,img_path in tqdm(self.data_loader):
            n, c, h, w = image.size()
            lc = torch.FloatTensor(n,18,512).zero_().cuda()
            fused_latentcode = torch.FloatTensor(n,18,512).zero_().cuda()
            image_name = fname[0]
            id_number = img_path[0].replace("\\", "/").split("/")[-2]
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_plus = self.get_e4e_inversion(image)
            w_plus = w_plus.to(global_config.device)
            torch.save(w_plus, f'{embedding_dir}/0.pt')

            lc += w_plus
            lcnorm = torch.norm(lc, p=2, dim=2, keepdim=True)
            lc = lc.div(lcnorm.expand_as(lc))
            for j in range(2,7):
                current_lc = lc[:, j, :]
                current_gallary = gallery_feature_norm[:, j, :]

                index, score = self.sort_img(current_lc, current_gallary)
                top_indices = index[:30]
                mask_with_indices = [i for i in top_indices if gallery_name[i].split('\\')[-2] != id_number]
                valid_index = mask_with_indices[0]
                selected_style_code = gallery_feature[:, j, :][valid_index]
                fused_latentcode[:,j,:] = selected_style_code*score[0] + w_plus[0][j]*(1-score[0])
            
            if self.use_wandb:
                log_images_from_w([fused_latentcode], self.G, [image_name+'_fused_G'])
            
            loss_lpips = 10
            blended_img = None
            for i in tqdm(range(hyperparameters.max_optimization_steps)):
                generated_images = self.forward(fused_latentcode)
                if i==0:
                    blended_img = generated_images.clone().detach()
                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break
                else: 
                    loss, l2_loss_val, loss_lpips, id_loss_val = self.calc_loss_obf(FRmodel, generated_images, real_images_batch, image_name, self.G, use_ball_holder, fused_latentcode, i,blended_img)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([fused_latentcode], self.G, [image_name+'_intermediate'])
                global_config.training_step += 1
                log_images_counter += 1

            if self.use_wandb:
                log_images_from_w([fused_latentcode], self.G, [image_name+'_optimized_tunedG'])

            blended_img = gen_img(fused_latentcode, self.G)
            adv_wm(blended_img, hyperparameters.txt_prompt, hyperparameters.candidate_num, hyperparameters.epsilon, alpha=1, test_dir='./results', arcface_path="./pretrained_model/arcface_checkpoint.tar", farl_path="./pretrained_model/FaRL-Base-Patch16-LAIONFace20M-ep64.pth", gallery_file='./VGGface2_hq_Gallery_e4e_id32_clip.h5')
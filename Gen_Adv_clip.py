# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
import os
import torch.nn.functional as F
import clip
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import h5py
from orderaware_lda import OALDA

######################################################################
# Options
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--attack', default='I-FGSM', type=str, choices=['FGSM','I-FGSM','MI-FGSM'])
parser.add_argument('--epsilon', default=4, type=int)
parser.add_argument('--save_fea', default=True, action='store_true')
parser.add_argument('--test_dir', default='./samples/', type=str, help='./test_data')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--s_path', default='./samples/sample.jpg', type=str)
parser.add_argument('--text', default='eyes', type=str)
parser.add_argument('--candidate_num', default=30, type=int, help='number of candidates')


opt = parser.parse_args()
torch.cuda.set_device(int(opt.gpu_ids))

def generate_prompts(attribute):
    generic_templates = [
        "a close-up photo of a person's {}.",
        "a portrait showing the {} clearly.",
        "a detailed photo focusing on the {}.",
        "a blurry image where only the {} stands out.",
        "a highly detailed painting of the {}.",
        "an artistic rendering highlighting the {}.",
        "a photo where the most noticeable feature is the {}.",
        "a 3D sculpture emphasizing the {}.",
        "a cartoon character with a unique {}.",
        "a side profile showing the {} in detail.",
    ]
    
    return [template.format(attribute) for template in generic_templates]

def find_similar_images(gallery_features, text_features, descending=True):
    attr_component = text_features

    gallery_attr_components = gallery_features
    similarities = (gallery_attr_components @ attr_component.T).squeeze()
    top_k_indices = similarities.argsort(descending=descending)

    return top_k_indices, similarities[top_k_indices]



def extract_feature_img(model, image):
    img = torch.nn.functional.interpolate(image, size=(112, 112), mode='bilinear', align_corners=False)
    img -= torch.cuda.FloatTensor([[[0.5]], [[0.5]], [[0.5]]])
    img /= torch.cuda.FloatTensor([[[0.5]], [[0.5]], [[0.5]]])
    image_features = model(img)
    image_features = F.normalize(image_features, p=2, dim=1)
    return image_features

def criterion(f1s, f2s):
    loss = -torch.mean(torch.sum(f1s * f2s, dim=1))
    return loss
    
def MI_FGSM(model, source_data, target_feat, epsilon=10.0, alpha=1.0, momentum=0.0):
    max_iter = int( min(epsilon+40, 12.5*epsilon) )
    x_adv = Variable(source_data.cuda(non_blocking=True))
    lower_bound = source_data.cuda() - epsilon / 255.0
    lower_bound[lower_bound < 0.0] = 0.0
    upper_bound = source_data.cuda() + epsilon / 255.0
    upper_bound[upper_bound > 1.0] = 1.0

    x_adv.requires_grad = True
    grad = None
    for cnt in range(max_iter):
    
        id_feat = extract_feature_img(model, x_adv)
        loss = criterion(id_feat, target_feat)
        loss.backward()
        x_grad = x_adv.grad.data
        norm = torch.mean(torch.abs(x_grad).view((x_grad.shape[0], -1)), dim=1).view((-1, 1, 1, 1))
        norm[norm < 1e-12] = 1e-12
        x_grad /= norm
        grad = x_grad if grad is None else momentum * grad + x_grad
        x_adv = x_adv.data + alpha / 255.0 * torch.sign(grad)

        x_adv = torch.max(x_adv, lower_bound)
        x_adv = torch.min(x_adv, upper_bound)
        
        x_adv.requires_grad = True

    return x_adv

#################################################

def adv_wm(
    s_img, 
    text, 
    candidate_num, 
    epsilon=4, 
    alpha=1, 
    test_dir='./results', 
    arcface_path="./pretrained_model/arcface_checkpoint.tar",
    farl_path="./pretrained_model/FaRL-Base-Patch16-LAIONFace20M-ep64.pth",
    gallery_file='./VGGface2_hq_Gallery_e4e_id32_clip.h5'
):
    

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        transforms.ToTensor(),
    ])
    toImage = transforms.ToPILImage()

    # Load ArcFace
    model = torch.load(arcface_path).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Load gallery
    with h5py.File(gallery_file, 'r') as f:
        gallery_feature_clip = torch.FloatTensor(f['gallery_clip'][:]).to(device)
        gallery_name = [name.decode('utf-8') for name in f['gallery_name'][:]]
        gallery_feature_id = torch.FloatTensor(f['gallery_f'][:]).to(device)

    # Load FaRL
    with torch.no_grad():
        FaRL, preprocess = clip.load("ViT-B/16", device="cpu")
        FaRL = FaRL.to(device)
        farl_state = torch.load(farl_path)
        FaRL.load_state_dict(farl_state["state_dict"], strict=False)

        img = preprocess(s_img).unsqueeze(0).to(device)
        image_features = FaRL.encode_image(img)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = FaRL.encode_text(clip.tokenize(text).to(device))
        text_features = text_features.mean(dim=0, keepdim=True)

        index, score = find_similar_images(gallery_feature_clip, text_features)

        selected_indices = []
        for i in index:
            selected_indices.append(i.item())
            if len(selected_indices) == candidate_num:
                break
        for i in reversed(index):
            selected_indices.append(i.item())
            if len(selected_indices) == candidate_num * 2:
                break


        selected_features = gallery_feature_id[selected_indices]
        group_1 = selected_features[:candidate_num].detach().cpu().numpy()
        group_2 = selected_features[candidate_num:].detach().cpu().numpy()

        X_clip = np.vstack([group_1, group_2])   
        y = np.array([0] * candidate_num + [1] * candidate_num)  

        lda = LinearDiscriminantAnalysis(n_components=1, solver='svd')
        lda.fit(X_clip, y)
        w_clip = lda.coef_.T
        projections_clip = (X_clip @ w_clip).flatten()
        lda_features_clip = lda.transform(X_clip) 
        order_clip = np.argsort(projections_clip)[::-1]
        ######################################################################################

        selected_features = gallery_feature_id[selected_indices]
        group_1 = selected_features[:candidate_num].detach().cpu().numpy()
        group_2 = selected_features[candidate_num:].detach().cpu().numpy()

        X_id = np.vstack([group_1, group_2])  
        y = np.array([0] * candidate_num + [1] * candidate_num) 

        lda = OALDA(order=order_clip, n_components=1, shrinkage='auto')
        lda.fit(X_id, y)
        w_id = lda.coef_.T

        lda_direction = lda.coef_.flatten() 
        lda_direction = lda_direction / np.linalg.norm(lda_direction)  
        lda_direction=torch.from_numpy(lda_direction).to('cuda')
    
        mu_A, mu_B = map(lambda x: torch.from_numpy(x).to(device), lda.means_)

    source_data = s_img.convert("RGB")
    source_data = data_transforms(source_data).unsqueeze(0).to(device)
    source = torch.nn.functional.interpolate(source_data, size=(112, 112), mode='bilinear', align_corners=False)
    source = (source - 0.5) / 0.5
   
    source_feat = model(source)
    source_feat = source_feat / source_feat.norm(dim=-1, keepdim=True)

    # Adversarial direction
    projs = torch.stack([mu_A, mu_B]) @ lda_direction.T
    proj_source = source_feat @ lda_direction.T
    dists = torch.abs(projs - proj_source)
    target_sign = (projs[torch.argmax(dists)] - proj_source).sign()
    target_feat = target_sign * lda_direction.unsqueeze(0)
    sign = target_sign.item()
    sign_symbol = '+' if sign > 0 else '-'
    
    x_adv = MI_FGSM(model, source_data, target_feat, epsilon=4, alpha=1)
    
    # Save image if needed
    x_adv_img = x_adv.squeeze(0)
    perturbed_name = f"adv_{epsilon}{sign_symbol}{text}.jpg"
    os.makedirs(test_dir, exist_ok=True)
    image = toImage(x_adv_img.cpu())
    image.save(os.path.join(test_dir, perturbed_name))
    print(f"Saved to {os.path.join(test_dir, perturbed_name)}")

    return x_adv


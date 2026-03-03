# : DeepProtect: Proactive Face-Swapping Defense Using Identity Blending and Attribute Distortion

## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (Not mandatory bur recommended)
- Python 3

### Installation
- Dependencies:  
	1. lpips
	2. wandb
	3. pytorch
	4. torchvision
	5. matplotlib
	6. dlib
- All dependencies can be installed using *pip install* and the package name

## Pretrained Models
Please download the pretrained models from the following links.
This includes the StyleGAN generator and pre-trained models.
| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) | StyleGAN2-ada model trained on FFHQ with 1024x1024 output resolution.
|[Dlib alignment](https://drive.google.com/file/d/1HKmjg6iXsWr4aFPuU0gBXPGR83wqMzq7/view?usp=sharing) | Dlib alignment used for images preproccessing.
|[FFHQ e4e encoder](https://drive.google.com/file/d/1ALC5CLA89Ouw40TwvxcwebhzWXM5YSCm/view?usp=sharing) | Pretrained e4e encoder. Used for StyleCLIP editing.
|[ArcFace](https://drive.google.com/drive/folders/1jV6_0FIMPC53FZ2HzZNJZGMe55bbu17R) | Pretrained ArcFace. Used for optimization and watermark generation.
| [FaRL (CLIP model)](https://github.com/FacePerceiver/FaRL) | Used for text-image embedding. Download weights and place them in `pretrained_model/` as described in the FaRL repository.

Note: The StyleGAN model is used directly from the official [stylegan2-ada-pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch).

By default, it is assumed that all pretrained models are downloaded and stored in the `pretrained_model` directory. 
However, you can specify your own paths by modifying the relevant values in `configs/path_configs.py`. 


## Inversion
### Preparing your Data
To invert a real image, you need to first align and crop it to the correct size. To do this, follow these steps:
Execute `utils/align_data.py` and update the "images_path" variable to point to the raw images directory.


### Running DeepProtect
The primary training script is `scripts/run.py`. It takes aligned and cropped images from the paths specified in the "Input info" subsection of `configs/paths_config.py`.
The results, including inversion latent codes and optimized generators, are saved to the directories listed under "Dirs for output files" in `configs/paths_config.py`.
The hyperparameters for the inversion task are defined in `configs/hyperparameters.py`, initialized with the default values used in the paper.

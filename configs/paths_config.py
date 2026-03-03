## Pretrained models paths
e4e = '../pretrained_models/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = '../pretrained_models/ffhq.pkl'
style_clip_pretrained_mappers = ''
ir_se50 = './pretrained_models/model_ir_se50.pth'
dlib = './pretrained_models/align.dat'

## Dirs for output files
checkpoints_dir = '../checkpoints'
embedding_base_dir = '../embeddings'
styleclip_output_dir = '../StyleCLIP_results'
experiments_output_dir = '../output'

## Input info
### Input dir, where the images reside
input_data_path = '../samples/n000008'

### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = ''

## Keywords
results_keyword = 'Optimization'
e4e_results_keyword = 'e4e'
## Architechture
lpips_type = 'alex'
first_inv_type = 'w+'
optim_type = 'adam'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = False
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 30

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1
pt_id_lambda = 0.1

## Steps
LPIPS_value_threshold = 0.06
max_optimization_steps = 450

## Optimization
learning_rate = 3e-4
first_inv_lr = 5e-3
train_batch_size = 1
use_last_w_pivots = False

epsilon = 4
txt_prompt = 'nose'
candidate_num = 30
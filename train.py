import sys, os
import torch
import torch.nn as nn
from monai.networks.nets import densenet
sys.path.append("/home/ubuntu/sushio/Kaggle_Pracs")
from models.effnet_model import EffnetModel
from models.model_saver import ModelSaver
from utils.optimizers import Optimizer

def _get_model_opt(opt, checkpoint=None):
    if checkpoint is not None:
        pass
    else:
        model_opt = opt
    return model_opt
    
def build_model(model_opt, fields, opts):
    # logger.info('Building model...')
    if opts.gpu and opts.gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif opts.gpu and not opts.gpu_id:
        device = torch.device("cuda")
    elif not opts.gpu:
        device = torch.device("cpu")

    if model_opt.model == "densenet121":
        model = densenet.densenet121(spatial_dims=3, in_channels=3,
                                 out_channels=opts.OUT_DIM)
        model.class_layers.out = nn.Seqeuntail(nn.Linear(in_features=1024, out_features=opts.OUT_DIM),
                                                nn.Softmax(dim=1))
        model.to(opts.DEVICE)
        
        return model
    if model_opt.model == "effnet-v2":
        model = EffnetModel()
        model.to(opts.DEVICE)
        return model

    # logger.info(model)

def build_model_saver(model_opt, opt, model, fields, optim):
    # _check_save_model_path
    save_model_path = os.path.abspath(opt.save_model)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.keep_checkpoint)
    return model_saver
    
def main(opts, fields):
    model_opt = _get_model_opt(opts)
    model = build_model(model_opt, fields, opts)
    optim = Optimizer.from_opt(model, opts)
    model_saver = build_model_saver(model_opt, opts, model, fields, optim)
    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=model_saver
    )

if __name__ == '__main__':

    opts = dict(model="densenet121", 
                  epochs=EPOCHS, 
                  split=N_SPLITS, 
                  batch=BATCH_SIZE, lr=LR, gpu=0
                  img_size=IMG_RESIZE, stack_size=STACK_RESIZE,
                  data_size=DF_SIZE)
    main()
    
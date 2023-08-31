### Ref: DreamDiff https://github.com/bbaaii/DreamDiffusion/tree/main/code/sc_mbm
import math, sys
import torch
import wandb
import models.utils as ut
from torch import inf
import numpy as np
import time


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def train_one_epoch(model, data_loader, optimizer, device, epoch, 
                    loss_scaler, logger=None, tensorboard_writer=None,
                    config=None, start_time=None, model_without_ddp=None, 
                    img_feature_extractor=None, preprocess=None, add_cor_loss=False):
    model.train(True)
    optimizer.zero_grad()

    if tensorboard_writer is not None:
        print('log_dir: {}'.format(tensorboard_writer.log_dir))

    total_loss = []
    total_cor_loss = []
    total_cor = []
    accum_iter = config.accum_iter
    start_time = time.time()
    for data_iter_step, (data_dcit) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        # print(data_iter_step)
        # print(len(data_loader))
        
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples = data_dcit['eeg']
        
        img_features = None
        valid_idx = None
        if img_feature_extractor is not None:
            images = data_dcit['image']
            valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)
            img_feature_extractor.eval()
            with torch.no_grad():
                img_features = img_feature_extractor(preprocess(images[valid_idx]).to(device))['layer2']
        samples = samples.to(device)
        # img_features = img_features.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _ = model(samples, img_features, valid_idx=valid_idx, mask_ratio=config.mask_ratio)

            # cal the cor
            pred = pred.detach()
            samples = samples.detach()
            pred = model_without_ddp.unpatchify(pred)
            cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))[0,1] for p, s in zip(pred, samples)]))

            if add_cor_loss:
                cor_loss = 1 - cor
            else:
                cor_loss = torch.tensor(0)
            final_loss = loss + cor_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(final_loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        optimizer.zero_grad()

        total_loss.append(loss_value)
        total_cor.append(cor.item())
        total_cor_loss.append(cor_loss.item())

        # wandb.log({"loss": loss_value, "correlation": cor})
        if device == torch.device('cuda:0'):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f'[{epoch}] [{data_iter_step}/{len(data_loader)}]', 
                'loss:', np.mean(total_loss), 
                'cor_loss:', np.mean(total_cor_loss), 
                'lr:', lr, 
                'cor', np.mean(total_cor), 
                'time', (time.time() - start_time)
            )
            tensorboard_writer.add_scalar('loss', np.mean(total_loss), data_iter_step)
            tensorboard_writer.add_scalar('cor_loss', np.mean(total_cor_loss), data_iter_step)
            tensorboard_writer.add_scalar('lr', lr, data_iter_step)
            tensorboard_writer.add_scalar('cor', np.mean(total_cor), data_iter_step)
            start_time = time.time()

    if logger is not None:
        lr = optimizer.param_groups[0]["lr"]
        logger.log('loss', np.mean(total_loss), step=epoch)
        logger.log('cor_loss', np.mean(total_cor_loss), step=epoch)
        logger.log('lr', lr, step=epoch)
        logger.log('cor', np.mean(total_cor), step=epoch)
        if start_time is not None:
            logger.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 0:        
        print(f'[Epoch {epoch}] loss: {np.mean(total_loss)} cor_loss: {np.mean(total_cor_loss)}')

    return np.mean(total_cor)
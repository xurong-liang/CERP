"""
    Some handy functions for pytorch model training ...
"""
import torch
import os


def assign_tensorboard_path(opt: dict) -> dict:
    """
    Assign the path to tensorboard log and alias name for tensorboard log

    :param opt: the opt dict
    :return: updated opt dict
    """
    opt["tensorboard"] = os.path.join(opt["prepath"], "runs")
    if not os.path.exists(opt["tensorboard"]):
        os.makedirs(opt["tensorboard"], exist_ok=True)

    # alias for tensorboard
    opt['alias'] = '{}_BaseDim{}_lr_{}_optim_{}_thresholdType{}_thres_init{}_l2_penalty{}'.format(
        opt['model'].upper(),
        opt['latent_dim'],
        opt['fm_lr'],
        opt['fm_optimizer'],
        opt['threshold_type'].upper(),
        opt['threshold_init'],
        opt['l2_penalty']
    )

    if 'retrain_emb_sparsity' in opt:
        # add retrain method to tensorboard alias
        if opt['re_init']:
            opt['alias'] += '_reinitTrue'
        else:
            opt['alias'] += '_reinitFalse'
        opt['alias'] += '_retrain_emb_sparsity{}'.format(opt['retrain_emb_sparsity'])

    opt["tensorboard"] = os.path.join(opt["tensorboard"], opt['alias'])
    return opt


def assign_candidate_p(opt: dict) -> dict:
    """
    Use the value of sparsity_rates, bucket size K and latent_dim to get
    the list of minimum pruning parameter targets and the max param possible.

    :param opt: the opt dict
    :return: the updated dict
    """
    candidate_p = []
    total_rows = opt["bucket_size"] * 2
    total_param = total_rows * opt["latent_dim"]

    # sparsity rates already in ascending order, so the candidate_p is in descending order
    for sparsity_rate in opt['sparsity_rates']:
        candidate_p.append(int((1 - sparsity_rate) * total_param))
    opt["candidate_p"] = candidate_p
    opt["max_param"] = total_param
    return opt


def get_grad_norm(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.data.view(-1, 1))
    if len(grads) == 0:
        grads.append(torch.FloatTensor([0]))
    grad_norm = torch.norm(torch.cat(grads))
    if grad_norm.is_cuda:
        grad_norm = grad_norm.cpu()
    return grad_norm.item()


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    # ensure all storage are on gpu
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))
    model.load_state_dict(state_dict)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['lr'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=params['lr'],
                                     betas=params['betas'],
                                     weight_decay=params['l2_regularization'],
                                     amsgrad=params['amsgrad'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['lr'],
                                        alpha=params['alpha'],
                                        momentum=params['momentum'],
                                        weight_decay=params['l2_regularization'])
    else:
        raise ValueError("No appropriate optimizer selected")
    return optimizer

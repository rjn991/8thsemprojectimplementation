import pickle
import numpy as np
import torch
import time

def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)
    
    images = torch.sub(images,0.5)
    images = torch.mul(images,2.0)
    
    return images

def calc_pairwise_distance(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [N,D]
        Y: [M,D]
    Returns:
        dist: [N,M] matrix of euclidean distances
    """
    rx=X.pow(2).sum(dim=1).reshape((-1,1))
    ry=Y.pow(2).sum(dim=1).reshape((-1,1))
    dist=rx-2.0*X.matmul(Y.t())+ry.t()
    return torch.sqrt(dist)

def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B=X.shape[0]
    
    rx=X.pow(2).sum(dim=2).reshape((B,-1,1))
    ry=Y.pow(2).sum(dim=2).reshape((B,-1,1))
    
    dist=rx-2.0*X.matmul(Y.transpose(1,2))+ry.transpose(1,2)
    
    return torch.sqrt(dist)

def sincos_encoding_2d(positions,d_emb):
    """
    Args:
        positions: [N,2]
    Returns:
        positions high-dimensional representation: [N,d_emb]
    """

    N=positions.shape[0]
    
    d=d_emb//2
    
    idxs = [np.power(1000,2*(idx//2)/d) for idx in range(d)]
    idxs = torch.FloatTensor(idxs).to(device=positions.device)
    
    idxs = idxs.repeat(N,2)  #N, d_emb
    
    pos = torch.cat([ positions[:,0].reshape(-1,1).repeat(1,d),positions[:,1].reshape(-1,1).repeat(1,d) ],dim=1)

    embeddings=pos/idxs
    
    embeddings[:,0::2]=torch.sin(embeddings[:,0::2])  # dim 2i
    embeddings[:,1::2]=torch.cos(embeddings[:,1::2])  # dim 2i+1
    
    return embeddings

# START: Original code by Zijian and Xinran
def ncc_val(I,J):
  I_mean = torch.mean(I)
  J_mean = torch.mean(J)
  I_std = torch.std(I)
  J_std = torch.std(J)
  ncc = torch.mean((I-I_mean)*(J-J_mean)/(I_std*J_std))
  return ncc

def calc_ncc(X, Y, index):
    ncc_top_list = []

    for i in range(index):
        ncc_list = []

        for j in range(index):
            ncc_list.append(ncc_val(X[0][i], Y[0][j]))
        ncc_top_list.append(ncc_list)
        
    ncc_top_list = np.array(ncc_top_list, dtype=np.float)
    ncc_top_list = torch.from_numpy(ncc_top_list).float()

    return ncc_top_list[None]

def calc_sad(X, Y):
   sad_all = []
   for i in range(X.shape[1]):
       sad_individual = []
       x = X[0, i, :]
       for j in range(Y.shape[1]):
           y=Y[0, j, :]
           l1_norm = torch.norm(x-y, p=1)
           sad_individual.append(l1_norm)
       sad_all.append(sad_individual)
   sad_all = torch.unsqueeze(torch.FloatTensor(sad_all), dim=0)
   return sad_all
# END: Original code by Zijian and Xinran

def print_log(file_path,*args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args,file=f)

def show_config(cfg):
    print_log(cfg.log_path, '=====================Config=====================')
    for k,v in cfg.__dict__.items():
        print_log(cfg.log_path, k,': ',v)
    print_log(cfg.log_path, '======================End=======================')
    
def show_epoch_info(phase, log_path, info):
    print_log(log_path, '')
    if phase=='Test':
        print_log(log_path, '====> %s at epoch #%d'%(phase, info['epoch']))
    else:
        print_log(log_path, '%s at epoch #%d'%(phase, info['epoch']))
        
    print_log(log_path, 'Group Activity Accuracy: %.2f%%, Individual Actions Accuracy: %.2f%%, Loss: %.5f, Using %.1f seconds'%(
                info['activities_acc'], info['actions_acc'], info['loss'], info['time']))
        
def log_final_exp_result(log_path, data_path, exp_result):
    no_display_cfg=['num_workers', 'use_gpu', 'use_multi_gpu', 'device_list',
                   'batch_size_test', 'test_interval_epoch', 'train_random_seed',
                   'result_path', 'log_path', 'device']
    
    with open(log_path, 'a') as f:
        print('', file=f)
        print('', file=f)
        print('', file=f)
        print('=====================Config=====================', file=f)
        
        for k,v in exp_result['cfg'].__dict__.items():
            if k not in no_display_cfg:
                print( k,': ',v, file=f)
            
        print('=====================Result======================', file=f)
        
        print('Best result:', file=f)
        print(exp_result['best_result'], file=f)
            
        print('Cost total %.4f hours.'%(exp_result['total_time']), file=f)
        
        print('======================End=======================', file=f)
    
    
    data_dict=pickle.load(open(data_path, 'rb'))
    data_dict[exp_result['cfg'].exp_name]=exp_result
    pickle.dump(data_dict, open(data_path, 'wb'))
        
    
class AverageMeter(object):
    """
    Computes the average value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time=time.time()
        self.init_time=time.time()
        self.total_time = 0
        
    def timeit(self):
        old_time=self.last_time
        self.last_time=time.time()
        return self.last_time-old_time

    def totaltime(self):
        self.total_time = time.time() - self.init_time
        return self.total_time
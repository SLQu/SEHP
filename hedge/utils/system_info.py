import subprocess


def get_gpu_information(gpu = 0):
    # run nvidia-smi
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True, shell=True)
    device = 'cuda:{}'.format(gpu) if gpu != -1 else 'cpu'
    
    
    return result,device
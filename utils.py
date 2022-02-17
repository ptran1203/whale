import pickle
import math
from torch.optim.lr_scheduler import LambdaLR

def pickle_save(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps , num_cycles=0.5, last_epoch=-1
):
    """
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py#L104
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

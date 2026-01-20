import torch
from trainer import Trainer, TrainerArgs

def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_model(config, output_path, model, train_samples, eval_samples, pretrained_path=''):
    clear_cuda()
    trainer = Trainer(
        TrainerArgs(continue_path=pretrained_path), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()

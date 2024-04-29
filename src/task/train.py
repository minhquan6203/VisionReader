import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import os
import numpy as np
from tqdm import tqdm
import gc
from data_utils.load_data import Load_Data
from builder.builder import build_model
from eval_metric.evaluate import ScoreCalculator
from utils.utils import countTrainableParameters, countParameters
class Training:
    def __init__(self, config):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.save_path = os.path.join(config['train']['output_dir'],config['model']['type_model'])
        self.best_metric= config['train']['metric_for_best_model']
        self.learning_rate = config['train']['learning_rate']
        self.weight_decay=config['train']['weight_decay']
        self.dataloader = Load_Data(config)
        if config['train']['precision']=='float32':
            self.cast_dtype=torch.float32
        elif config['train']['precision']=='bfloat16':
            self.cast_dtype=torch.bfloat16
        else:
            self.cast_dtype=torch.float16
        self.cuda_device=config['train']['cuda_device']
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.base_model=build_model(config).to(self.device)
        params=countParameters(self.base_model)
        trainable_param=countTrainableParameters(self.base_model)
        print('total params: ', params)
        print('trainable params: ', trainable_param)
        print(f'% trainable params / total params: {100*(trainable_param/params):.2f}')
        self.compute_score = ScoreCalculator()
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        lambda1 = lambda epoch: 0.95 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
    
        train,valid = self.dataloader.load_train_dev()
        
        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']
        else:
            best_score = 0.
            
        threshold=0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            valid_em = 0.
            valid_wups=0.
            valid_f1 =0.
            valid_cider=0.
            valid_bert=0.
            train_loss = 0.
            valid_loss = 0.
            with tqdm(desc='Epoch %d - Training stage' % (epoch+1), unit='it', total=len(train)) as pbar:
                for it, item in enumerate(train):
                    with torch.autocast(device_type='cuda', dtype=self.cast_dtype, enabled=True):
                        logits, loss = self.base_model(item['question'],item['image_id'],item['answer'])
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    train_loss += loss.item()
                    pbar.set_postfix(loss=train_loss / (it + 1))
                    pbar.update()
                self.scheduler.step()
                train_loss /=len(train)
            
            with torch.no_grad():
                for it, item in enumerate(tqdm(valid)):
                    with torch.autocast(device_type='cuda', dtype=self.cast_dtype, enabled=True):
                        pred_answers = self.base_model(item['question'],item['image_id'])    
                        valid_wups+=self.compute_score.wup(item['answer'],pred_answers)
                        valid_em+=self.compute_score.em(item['answer'],pred_answers)
                        valid_f1+=self.compute_score.f1_token(item['answer'],pred_answers)
                        valid_cider+=self.compute_score.cider_score(item['answer'],pred_answers)
                valid_loss /=len(valid)
                valid_wups /= len(valid)
                valid_em /= len(valid)
                valid_f1 /= len(valid)
                valid_bert/=len(valid)
                valid_cider/=len(valid)
            
            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f}")
            print(f"valid loss: {valid_loss:.4f} valid wups: {valid_wups:.4f} valid em: {valid_em:.4f} valid f1: {valid_f1:.4f} valid cider: {valid_cider:.4f} valid bert: {valid_bert:.4f}")

            with open(os.path.join(self.save_path,'log.txt'), 'a') as file:
                file.write(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}\n")
                file.write(f"train loss: {train_loss:.4f}\n")
                file.write(f"valid loss: {valid_loss:.4f} valid wups: {valid_wups:.4f} valid em: {valid_em:.4f} valid f1: {valid_f1:.4f} valid cider: {valid_cider:.4f} valid bert: {valid_bert:.4f}\n")

            if self.best_metric =='em':
                score=valid_em
            if self.best_metric=='f1':
                score=valid_f1
            if self.best_metric=='wups':
                score=valid_wups
            if self.best_metric=='cider':
                score=valid_cider
            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'score': score}, os.path.join(self.save_path, 'last_model.pth'))
            
            # save the best model
            if epoch > 0 and score <= best_score:
              threshold += 1
            else:
              threshold = 0

            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    # 'optimizer_state_dict': self.optimizer.state_dict(),
                    'score':score}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with {self.best_metric} of {score:.4f}")
            
            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break

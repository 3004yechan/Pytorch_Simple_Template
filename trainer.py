import os
import sys
import torch
import numpy as np
from tqdm import tqdm

class Trainer():
    def __init__(self, train_loader, valid_loader, model, loss_fn, optimizer, scheduler, device, patience, epochs, result_path, fold_logger, **kargs):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.epochs = epochs
        self.logger = fold_logger
        self.best_model_path = os.path.join(result_path, 'best_model.pt')
        self.last_model_path = os.path.join(result_path, 'last_model.pt')

        self.start_epoch = kargs['start_epoch'] if 'start_epoch' in kargs else 0
    
    def train(self):
        best = np.inf
        for epoch in range(self.start_epoch + 1, self.epochs+1):
            print(f'lr: {self.scheduler.get_last_lr()}')
            loss_train, score_train = self.train_step()
            loss_val, score_val = self.valid_step()
            self.scheduler.step()

            self.logger.info(f'Epoch {str(epoch).zfill(5)}: t_loss:{loss_train:.3f} t_score:{score_train:.3f} v_loss:{loss_val:.3f} v_score:{score_val:.3f}')

            if loss_val < best:
                best = loss_val
                torch.save({
                    'model':self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'score_val': score_val.item(),
                    'loss_val': loss_val.item(), 
                }, self.best_model_path)
                bad_counter = 0

            else:
                bad_counter += 1

            if bad_counter == self.patience:
                break

            torch.save({
                'model':self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': epoch,
                'score_val': score_val.item(),
                'loss_val': loss_val.item(), 
            }, self.last_model_path)

    def train_step(self):
        self.model.train()

        total_loss = 0
        correct = 0
        for batch in tqdm(self.train_loader, file=sys.stdout): #tqdm output will not be written to logger file(will only written to stdout)
            x, y = batch['image'].to(self.device), batch['label'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            # BCEWithLogitsLoss를 위해 라벨 텐서의 차원을 맞춤 (B, 1)
            loss = self.loss_fn(output, y.unsqueeze(1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.shape[0]
            # 정확도 계산 로직 수정: logit > 0 이면 1, 아니면 0으로 예측
            preds = (output.squeeze() > 0).float()
            correct += (preds == y).sum().item()
        
        return total_loss/len(self.train_loader.dataset), correct/len(self.train_loader.dataset)
    
    def valid_step(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            for batch in self.valid_loader:
                x, y = batch['image'].to(self.device), batch['label'].to(self.device)

                output = self.model(x)
                # BCEWithLogitsLoss를 위해 라벨 텐서의 차원을 맞춤 (B, 1)
                loss = self.loss_fn(output, y.unsqueeze(1))

                total_loss += loss.item() * x.shape[0]
                # 정확도 계산 로직 수정: logit > 0 이면 1, 아니면 0으로 예측
                preds = (output.squeeze() > 0).float()
                correct += (preds == y).sum().item()
                
        return total_loss/len(self.valid_loader.dataset), correct/len(self.valid_loader.dataset)
    
    def test(self, test_loader):
        # self.model.load_state_dict(torch.load(self.best_model_path)) # main.py에서 trainer.train()을 호출하면 이미 최적 모델이 로드되어 있음
        self.model.eval()
        with torch.no_grad():
            predictions = []
            true_labels = []
            for batch in test_loader:
                x = batch['image'].to(self.device)
                y = batch['label'].numpy()
                
                output = self.model(x)
                # Softmax 대신 원시 logit 값을 그대로 반환
                output = output.detach().cpu().numpy()
                
                predictions.append(output)
                true_labels.append(y)

        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        return predictions, true_labels
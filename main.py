import os
import sys
import logging
import pandas as pd
from functools import partial
# from sklearn.model_selection import train_test_split # 더 이상 필요 없음

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

# 프로젝트 모듈 임포트
from models import Temp
from data import CelebADataset  # CelebADataset을 직접 임포트
from trainer import Trainer
from config import get_args
from lr_scheduler import get_sch
from utils import seed_everything, handle_unhandled_exception, save_to_json

if __name__ == "__main__":
    # 1. 초기 설정 및 환경 구성
    args = get_args()
    seed_everything(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 이진 분류를 위해 출력 뉴런 수를 1로 설정
    args.num_classes = 1

    # 2. 결과 저장 경로 설정
    if args.continue_train > 0:
        result_path = args.continue_from_folder
    else:
        result_path = os.path.join(args.result_path, args.model +'_celeba_'+str(len(os.listdir(args.result_path))))
        os.makedirs(result_path)
    
    # 3. 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))
    logger.info(args)
    save_to_json(vars(args), os.path.join(result_path, 'config.json'))
    sys.excepthook = partial(handle_unhandled_exception, logger=logger)

    # 4. CelebA 데이터셋 생성
    # --path 인자로 CelebA 데이터셋의 루트 디렉토리를 지정해야 함 (예: 'dataset/celebA')
    train_dataset = CelebADataset(root_dir=args.path, mode='train', image_size=args.image_size)
    valid_dataset = CelebADataset(root_dir=args.path, mode='valid', image_size=args.image_size)
    test_dataset = CelebADataset(root_dir=args.path, mode='test', image_size=args.image_size)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # 5. 모델, 손실함수, 옵티마이저, 스케줄러 설정
    model = Temp(args).to(device)
    # 이진 분류에 적합한 BCEWithLogitsLoss로 변경
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_sch(args.scheduler, optimizer, warmup_epochs=args.warmup_epochs, epochs=args.epochs)

    # 6. 이어서 학습하는 경우 체크포인트 로드
    if args.continue_train_from is not None:
        state = torch.load(args.continue_train_from)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch = state['epoch']
    else:
        epoch = 0

    # 7. 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: {'image': torch.stack([i['image'] for i in x]), 'label': torch.stack([i['label'] for i in x])}
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: {'image': torch.stack([i['image'] for i in x]), 'label': torch.stack([i['label'] for i in x])}
    )
    
    # 8. 트레이너 생성 및 학습 시작
    trainer = Trainer(
        train_loader, valid_loader, model, loss_fn, optimizer, scheduler, 
        device, args.patience, args.epochs, result_path, logger, start_epoch=epoch
    )
    trainer.train()

    # 9. 테스트 데이터에 대한 예측 수행
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: {'image': torch.stack([i['image'] for i in x]), 'label': torch.stack([i['label'] for i in x])}
    )

    # 10. 최종 예측 결과 저장 (테스트 성능 평가)
    test_predictions, test_labels = trainer.test(test_loader)
    
    # 예시: 예측 결과를 파일로 저장
    # Logits > 0 이면 클래스 1로 예측
    predictions_df = pd.DataFrame({
        'predictions': (test_predictions.squeeze() > 0).astype(int),
        'labels': test_labels
    })
    predictions_df.to_csv(os.path.join(result_path, 'test_predictions.csv'), index=False)
    
    logger.info("Training and testing finished.") 
import os
import argparse
import time
import yaml
from loguru import logger
from easydict import EasyDict
import json

import torch

from src.util import *
from src.model import *
from src.quantizer import *

def main(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(config.base.seed)
    else:
        device = torch.device('cpu')

    logger.info(f'device: {device}')
    
    if config.base.mode == 'train_org':
        model = FourViewClassifier4MNIST().to(device)
        train_loader, test_loader = Dataloader(config).load_data()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        train(model, train_loader, optimizer, criterion, device, config)    
        test(model, test_loader, criterion, device, config, is_quant = False)     
        
        # save model parameters
        save_path = os.path.join(config.base.root_path, 'save', 'ckpt')
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, 'model_params.pth'))
        logger.info(f'Model parameters saved to {os.path.join(save_path, "model_params.pth")}')
    
    elif config.base.mode == 'train_quant':
        quantizer = FeatQuantizer(config.train.qbit)
        
        model = QFourViewClassifier4MNIST(quantizer).to(device)
        train_loader, test_loader = Dataloader(config).load_data()
        
        # execute fake input forward to init the quantizer
        logger.info('Init the quantizer: fake_input forward ...')
        fake_input, _ = next(iter(train_loader))
        model(fake_input.to(device))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        train(model, train_loader, optimizer, criterion, device, config)    
        test(model, test_loader, criterion, device, config, is_quant = False)     
        
        # save model parameters
        save_path = os.path.join(config.base.root_path, 'save', 'ckpt')
        os.makedirs(save_path, exist_ok=True)
        save_path = f"{save_path}/model_params_{config.train.qbit}bit.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model parameters saved!")  
            
    elif config.base.mode == 'eval_org':
        model = FourViewClassifier4MNIST().to(device)
        
        ckpt = f'{config.base.root_path}/{config.eval.ckpt}'
        model.load_state_dict(torch.load(ckpt, map_location=device))
        logger.info(f'Model weights loaded from {ckpt}')
        
        test_loader = Dataloader(config).load_data()[1]
        criterion = torch.nn.CrossEntropyLoss()
        
        test(model, test_loader, criterion, device, config, is_quant = True)        
        
    elif config.base.mode == 'eval_quant':
        # evaluation on quantization-aware trained model can be found at test(...) under the 'mode == train_quant'
        raise NotImplementedError('find eval_quant results in train_quant mode')
    else:
        raise ValueError(f'Invalid mode: {config.base.mode}')

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    logger.info(f'args: {args}')

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)

    logger.info(f'config:\n{json.dumps(config, ensure_ascii=False, indent=4)}')
   
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.base.cuda_visible_devices)

    main(config)

    end_time = time.time()
    logger.info(f'Total time: {end_time - start_time:.2f} seconds')
import torch
import torch.nn as nn

from loguru import logger

def train(model, train_loader, optimizer, criterion, device, config):
    
    logger.info('-- start training --')
    model.train()
    for epoch in range(config.train.epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % config.train.log_interval == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                            f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
        
def test(model, test_loader, criterion, device, config, is_quant = False):
    logger.info('-- start testing --')
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if not is_quant:
                qbit = -1
                output = model(data)
                
            else:
                feats = model.get_feats(data)  # sz=(bsz, 32*4)
                qbit = config.eval.qbit
                max_val, _ = torch.max(feats, dim=1, keepdim=True) 
                min_val, _ = torch.min(feats, dim=1, keepdim=True)
                scales = (max_val - min_val) / (2**qbit - 1)
                feats = torch.round(torch.clamp(feats / scales, -2**(qbit-1), 2**(qbit-1) - 1)) * scales
                output = model.classify(feats)
        
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)  # 累加损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info(f'====> Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%) @ {qbit}bit quantization')
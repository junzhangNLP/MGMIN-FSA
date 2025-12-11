import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch import optim
from tools.Utils import str2bools, str2floats, str2listoffints, get_labels_from_datas, get_loss_label_from_labels

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", default='fasmr', type=str, 
                       choices=['CMU', 'fasmr', 'Ch-sims'])
    parser.add_argument("--dataset", default='mosei_SDK', type=str)
    parser.add_argument("--normalize", default='0-0-0', type=str2bools)
    parser.add_argument("--text", default='glove', type=str)
    parser.add_argument("--audio", default='covarep', type=str)
    parser.add_argument("--video", default='facet42', type=str)
    parser.add_argument("--data_path", default='', type=str)
    parser.add_argument("--persistent_workers", action='store_true')
    parser.add_argument("--pin_memory", action='store_true')
    parser.add_argument("--d_t", default=768, type=int)
    parser.add_argument("--d_a", default=74, type=int) 
    parser.add_argument("--d_v", default=35, type=int)

    # parser.add_argument("--d_t", default=768, type=int)
    # parser.add_argument("--d_a", default=74, type=int) 
    # parser.add_argument("--d_v", default=47, type=int)
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--task", default='regression', type=str, 
                       choices=['classification', 'regression'])
    parser.add_argument("--num_class", default=1, type=int)
    
    parser.add_argument("--epochs", default=100, type=int )
    parser.add_argument("--lr", default=0.002, type=float )
    parser.add_argument("--text_lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--nmda", default=0.9, type=float)
    parser.add_argument("--beta", default=0.3, type=float)
    parser.add_argument("--early_stop", default=8, type=int)
    
    parser.add_argument("--hidden_dim", default='256-64-64-128', type=str)
    parser.add_argument("--post_dim", default='64-16-16-128', type=str)
    parser.add_argument("--prob", default='0.1-0.1-0.1-0.1', type=str)
    
    parser.add_argument("--tav_loss", default='0.8-0.8-0.4', type=str)
    
    parser.add_argument("--seed", default=1111, type=int)
    parser.add_argument("--gpu", default='0', type=str)
    
    return parser.parse_args()


def get_config_preset(args):
    configs = {
        'CMU': {
            'data_loader': 'DataLoaderUniversal',
            'model_class': '112',
            'model_type': 'english_roberta',
            'emd_dim': {'T': args.d_t, 'A': args.d_a, 'V': args.d_v},
            'model_params': {
                'hidden_dim': {'T': 256, 'A': 64, 'V': 64, 'M': 128},
                'prob': {'T': 0.1, 'A': 0.1, 'V': 0.1, 'M': 0.1},
                'post_prob': {'T': 0.1, 'A': 0.1, 'V': 0.1, 'M': 0.1},
                'post_dim': {'T': 64, 'A': 16, 'V': 16, 'M': 128}
            }
        },
        'fasmr': {
            'data_loader': 'MMDataLoader',
            'model_class': '113',
            'model_type': 'chinese_bert',
            'data_path': './dataset/fasmr/feature_New.pkl',
            'emd_dim': {'T': 768, 'A': 768, 'V': 465},
            'model_params': {
                'hidden_dim': {'T': 128, 'A': 128, 'V': 64, 'M': 128},
                'prob': {'T': 0.1, 'A': 0.1, 'V': 0.1, 'M': 0.1},
                'post_prob': {'T': 0.1, 'A': 0.1, 'V': 0.1, 'M': 0.1},
                'post_dim': {'T': 64, 'A': 16, 'V': 16, 'M': 128}
            },
            'special': {
                'tokenizer_path': ""
            }
        },
        'Ch-sims': {
            'data_loader': 'CHDataLoader',
            'model_class': '114',
            'model_type': 'english_roberta',
            'data_path': './dataset/CH-SIMS/unaligned_39.pkl',
            'emd_dim': {'T': 768, 'A': 33, 'V': 709},
            'model_params': {
                'hidden_dim': {'T': 256, 'A': 32, 'V': 64, 'M': 32},
                'prob': {'T': 0.1, 'A': 0.1, 'V': 0.1, 'M': 0.1},
                'post_prob': {'T': 0.1, 'A': 0.1, 'V': 0.1, 'M': 0.1},
                'post_dim': {'T': 128, 'A': 16, 'V': 16, 'M': 32}
            }
        }
    }
    
    config = configs[args.config].copy()
    
    if args.hidden_dim:
        dims = list(map(int, args.hidden_dim.split('-')))
        config['model_params']['hidden_dim'] = {'T': dims[0], 'A': dims[1], 'V': dims[2], 'M': dims[3]}
    
    if args.post_dim:
        dims = list(map(int, args.post_dim.split('-')))
        config['model_params']['post_dim'] = {'T': dims[0], 'A': dims[1], 'V': dims[2], 'M': dims[3]}
    
    if args.prob:
        probs = list(map(float, args.prob.split('-')))
        config['model_params']['prob'] = {'T': probs[0], 'A': probs[1], 'V': probs[2], 'M': probs[3]}
        config['model_params']['post_prob'] = {'T': probs[0], 'A': probs[1], 'V': probs[2], 'M': probs[3]}
    
    return config

def load_data(args, config):
    if config['data_loader'] == 'DataLoaderUniversal':
        from tools.DataLoaderUniversal import get_data_loader
        train_loader, valid_loader, test_loader = get_data_loader(args)
        return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    
    elif config['data_loader'] == 'MMDataLoader':
        from tools.data_loader import MMDataLoader
        dataloader = MMDataLoader(
            config['data_path'],
            config['emd_dim'],
            num_workers=args.num_workers
        )
        return dataloader
    
    elif config['data_loader'] == 'CHDataLoader':
        from tools.CH_dataloader import MMDataLoader
        dataloader = MMDataLoader(config['data_path'])
        return dataloader
    
    else:
        raise ValueError(f"未知的数据加载器: {config['data_loader']}")


def load_model(args, config, device):
    from model.model import MGMIN_FSA
    
    if 'model_params' not in config:
        raise ValueError("Lack model_params")
    
    if 'emd_dim' not in config:
        config['emd_dim'] = {
            "T": getattr(args, 'd_t', 768),
            "A": getattr(args, 'd_a', 74),
            "V": getattr(args, 'd_v', 47)
        }
    
    model_config = {
        "model_type": config.get("model_type", "chinese_bert"), 
        "model_path": config.get("special", {}).get("tokenizer_path", None), 
        "use_token_type": config.get("use_token_type", True),  
        "use_kan_attention": config.get("use_kan_attention", True), 
        "use_avg_padding": config.get("use_avg_padding", False),  
        "use_enhanced_same": config.get("use_enhanced_same", True)  
    }
    
    print(f"Loading model with config:")
    print(f"  model_type: {model_config['model_type']}")
    print(f"  model_path: {model_config['model_path']}")
    
    model = MGMIN_FSA(
        feature_dims=config['emd_dim'],
        hidden_dims=config['model_params']['hidden_dim'],
        dropouts=config['model_params']['prob'],
        post_dropouts=config['model_params']['post_prob'],
        post_dim=config['model_params']['post_dim'],
        train_mode=args.task,
        num_classes=args.num_class,
        model_config=model_config  
    ).to(device)
    
    return model


def do_test(model, dataloader, device, train_mode, criterion, args, config):

    model.eval()
    y_pred = []
    y_true = []
    eval_loss = 0.0
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Testing") as td:
            for batch_data in td:
                if config['data_loader'] == 'DataLoaderUniversal':
                    t_data, a_data, v_data, input_ids, att_mask = (
                        batch_data[0].to(device).float(),
                        batch_data[1].to(device).float(),
                        batch_data[2].to(device).float(),
                        batch_data[3].to(device),
                        batch_data[4].to(device)
                    )
                    
                    labels = get_labels_from_datas(batch_data, args)
                    targets = get_loss_label_from_labels(labels, args).to(device)
                    
                    outputs = model(input_ids, att_mask, a_data, v_data)
                    
                else:
                    input_ids = batch_data['input_ids'].to(device)
                    att_mask = batch_data['att_mask'].to(device)
                    token_type_ids = batch_data['token_type_ids'].to(device) if 'token_type_ids' in batch_data else None
                    vision_cut = batch_data['vision'].to(device)
                    audio_cut = batch_data['audio'].to(device)
                    labels = batch_data['labels']
                    
                    for k in labels.keys():
                        if train_mode == 'classification':
                            labels[k] = labels[k].to(device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(device).view(-1, 1)
                    
                    outputs = model(input_ids, att_mask, audio_cut, vision_cut, 
                                   fs=False, token_type_ids=token_type_ids)
                    targets = labels['M']
                
                m_loss = criterion(outputs["M"], targets)
                eval_loss += m_loss.item()
                
                y_pred.append(outputs['M'].detach().cpu())
                y_true.append(targets.detach().cpu())
    
    eval_loss = eval_loss / len(dataloader)
    
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    

    from tools.function import eval_fgmsa_regression
    train_results = eval_fgmsa_regression(pred, true)
        
    print(f"FG-MSA: {train_results}")
        
    return {'loss': eval_loss, 'results': train_results}


def train_epoch(model, dataloader, optimizer, criterion, info_nce_loss_fn, 
                device, train_mode, args, config):
    model.train()
    train_loss = 0.0
    y_pred = []
    y_true = []
    
    mem = {"T": None, "A": None, "V": None}
    npm = {"T": {"neg": None, "pos": None}, 
           "A": {"neg": None, "pos": None}, 
           "V": {"neg": None, "pos": None}}
    
    with tqdm(dataloader, desc="Training") as pbar:
        for batch_data in pbar:
            if config['data_loader'] == 'DataLoaderUniversal':
                t_data, a_data, v_data, input_ids, att_mask = (
                    batch_data[0].to(device).float(),
                    batch_data[1].to(device).float(),
                    batch_data[2].to(device).float(),
                    batch_data[3].to(device),
                    batch_data[4].to(device)
                )
                
                labels = get_labels_from_datas(batch_data, args)
                targets = get_loss_label_from_labels(labels, args).to(device)
                
                outputs = model(input_ids, att_mask, a_data, v_data)
                
            else:
                input_ids = batch_data['input_ids'].to(device)
                att_mask = batch_data['att_mask'].to(device)
                token_type_ids = batch_data['token_type_ids'].to(device) if 'token_type_ids' in batch_data else None
                vision_cut = batch_data['vision'].to(device)
                audio_cut = batch_data['audio'].to(device)
                labels = batch_data['labels']
                
                for k in labels.keys():
                    if train_mode == 'classification':
                        labels[k] = labels[k].to(device).view(-1).long()
                    else:
                        labels[k] = labels[k].to(device).view(-1, 1)
                
                outputs = model(input_ids, att_mask, audio_cut, vision_cut, 
                               fs=False, token_type_ids=token_type_ids)
                targets = labels['M']
            
            optimizer.zero_grad()
            
            if mem["T"] is not None:
                mem["T"] = (mem["T"] + outputs["T"].detach()) / 2
                mem["A"] = (mem["A"] + outputs["A"].detach()) / 2
                mem["V"] = (mem["V"] + outputs["V"].detach()) / 2
            else:
                mem["T"] = outputs["T"].detach()
                mem["A"] = outputs["A"].detach()
                mem["V"] = outputs["V"].detach()
            
            mem_loss = (info_nce_loss_fn(outputs["A"], mem["A"], torch.stack((mem["T"], mem["V"]), dim=1)) +
                       info_nce_loss_fn(outputs["V"], mem["V"], torch.stack((mem["A"], mem["T"]), dim=1))+
                       info_nce_loss_fn(outputs["T"], mem["T"], torch.stack((mem["A"], mem["V"]), dim=1)))
            
            for i in range(len(targets)):
                for modality in ["T", "A", "V"]:
                    if targets[i] > 0:
                        if npm[modality]["pos"] is not None:
                            npm[modality]["pos"] = (npm[modality]["pos"] + outputs[modality].detach()) / 2
                        else:
                            npm[modality]["pos"] = outputs[modality].detach()
                    else:
                        if npm[modality]["neg"] is not None:
                            npm[modality]["neg"] = (npm[modality]["neg"] + outputs[modality].detach()) / 2
                        else:
                            npm[modality]["neg"] = outputs[modality].detach()
            
            mnei_loss = 0
            for i in range(len(targets)):
                for modality in ["T", "A", "V"]:
                    if targets[i] > 0:
                        if npm[modality]["neg"] is not None:
                            neg_samples = torch.stack([npm[modality]["neg"]] * 2, dim=1)
                            mnei_loss += info_nce_loss_fn(outputs[modality], npm[modality]["pos"], neg_samples)
                    else:
                        if npm[modality]["pos"] is not None:
                            pos_samples = torch.stack([npm[modality]["pos"]] * 2, dim=1)
                            mnei_loss += info_nce_loss_fn(outputs[modality], npm[modality]["neg"], pos_samples)
            
            if config['data_loader'] == 'DataLoaderUniversal':
                m_loss = criterion(outputs["M"], targets.unsqueeze(1))
            else:
                tav_weights = list(map(float, args.tav_loss.split('-'))) if args.tav_loss else [0.8, 0.8, 0.4]
                m_loss = (criterion(outputs["M"], targets) + 
                         tav_weights[0] * criterion(outputs.get("pre_t", outputs["T"]), labels.get('T', targets)) +
                         tav_weights[1] * criterion(outputs.get("pre_a", outputs["A"]), labels.get('A', targets)) +
                         tav_weights[2] * criterion(outputs.get("pre_v", outputs["V"]), labels.get('V', targets)))
            
            loss = (args.nmda * m_loss + 
                   (1 - args.nmda) * (args.beta * mem_loss + (1 - args.beta) * mnei_loss))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            y_pred.append(outputs["M"].detach().cpu())
            y_true.append(targets.detach().cpu())
            
            pbar.set_postfix({'loss': loss.item()})
    
    train_loss = train_loss / len(dataloader)
    return train_loss, y_pred, y_true, mem, npm

def main():
    args = parse_args()
    
    from tools.function import setup_seed, assign_gpu
    setup_seed(args.seed)
    
    device = assign_gpu([int(args.gpu)])
    
    config = get_config_preset(args)
    
    print(f"Loading data with config: {args.config}")
    dataloaders = load_data(args, config)
    
    print(f"Loading model: {config['model_class']}")
    model = load_model(args, config, device)
    
    model_params_other = [p for n, p in model.named_parameters() if 'text_enc' not in n]
    
    optimizer = optim.AdamW([
        {"params": model.text_enc.parameters(), "lr": args.text_lr, "weight_decay": 1e-4},
        {"params": model_params_other, "lr": args.lr, "weight_decay": args.weight_decay}
    ])
    
    criterion = nn.MSELoss()
    from tools.function import InfoNCELoss
    info_nce_loss_fn = InfoNCELoss(temperature=0.1)

    best_valid = float('inf')
    best_epoch = 0
    early_stop_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        

        train_loss, y_pred, y_true, mem, npm = train_epoch(
            model, dataloaders['train'], optimizer, criterion, 
            info_nce_loss_fn, device, args.task, args, config
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        

        val_results = do_test(model, dataloaders['valid'], device, 
                            args.task, criterion, args, config)
        

        cur_valid = val_results['loss']
        if cur_valid < best_valid:
            best_valid = cur_valid
            best_epoch = epoch
            early_stop_counter = 0

            torch.save(model.state_dict(), f'best_model_{args.config}.pth')
            print(f"Best model saved at epoch {epoch}")
        else:
            early_stop_counter += 1
        

        if early_stop_counter >= args.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    

    print(f"\nLoading best model from epoch {best_epoch} for testing...")
    model.load_state_dict(torch.load(f'best_model_{args.config}.pth'))
    
    test_results = do_test(model, dataloaders['test'], device, 
                          args.task, criterion, args, config)
    
    print(f"\nTraining completed!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_valid:.4f}")
    print(f"Final test results: {test_results}")

if __name__ == "__main__":
    main()
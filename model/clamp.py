from typing import Any, Dict, List, Mapping
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from data.augmentation import GraphAttrMaskingAugmentation, GraphPerturbationAugmentation, TokenRandomMaskingAugmentation
import random
from peft import get_peft_model, LoraConfig


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optimizer.optimizer, 
            # params=[
            #     {'params':self.graph_encoder.parameters(), 'lr':self.hparams.optimizer.optimizer.lr*10},
            #     {'params':self.text_encoder.parameters(), 'lr':self.hparams.optimizer.optimizer.lr},
            #     {'params':self.loss.parameters(), 'lr':self.hparams.optimizer.optimizer.lr},
            # ], 
            params=self.parameters(),
            _convert_="partial"
        )
        if not self.hparams.optimizer.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}

    # def configure_optimizers(self) -> OptimizerLRScheduler:
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)        
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    #     return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}
 

class GNNSupervised(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.graph_encoder = hydra.utils.instantiate(self.hparams.graph_encoder)

    def forward(self, batch):
        return self.graph_encoder(batch)
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.mse_loss(out, batch.y)
        self.log_dict(
            {"train_loss", loss},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.l1_loss(out, batch.y)
        self.log_dict(
            {"val_loss", loss},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss
    
    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.l1_loss(out, batch.y)
        self.log_dict(
            {"test_loss", loss},
        )
        return loss

    
class CLaMP(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.graph_encoder = hydra.utils.instantiate(self.hparams.graph_encoder)
        self.text_encoder = hydra.utils.instantiate(self.hparams.text_encoder)
        
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.text_encoder.llm)

        self.graph_augmentation = hydra.utils.instantiate(self.hparams.graph_augmentation)
        self.text_augmentation = hydra.utils.instantiate(self.hparams.text_augmentation, mask_token=self.tokenizer.mask_token)

        self.w_graph = torch.nn.Linear(self.hparams.graph_encoder.hidden_dim, self.hparams.clamp_dim)
        self.w_text = torch.nn.Linear(self.hparams.text_encoder.hidden_dim, self.hparams.clamp_dim)

        self.temperature = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, **inputs):
        '''
        graph_logits: (batch_size, clamp_dim)
        text_logits: (batch_size, clamp_dim)
        '''
        graph_logits = self.w_graph(self.graph_encoder(**inputs))
        text_logits = self.w_text(self.text_encoder(**inputs).last_hidden_state[:,0])
        
        all_graph_logits = F.normalize(self.all_gather(graph_logits, sync_grads=True).view(-1, graph_logits.shape[-1]))
        all_text_logits = F.normalize(self.all_gather(text_logits, sync_grads=True).view(-1, text_logits.shape[-1]))
        return all_graph_logits, all_text_logits
    
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        graphs, texts = batch
        graphs = self.graph_augmentation(graphs)
        texts = self.text_augmentation(texts)
        return graphs, texts
    
    def clamp_loss(self, graph_logits, text_logits, mode):
        gt_logits = torch.matmul(graph_logits, text_logits.transpose(0,1)) * torch.exp(self.temp)  # (batch_size, batch_size)
        labels = torch.arange(gt_logits.shape[0], device=gt_logits.device)  # (batch_size)
        gt_loss_graph = self.cross_entropy_loss(gt_logits, labels)  # (batch_size) graph가 text를 보고 잘 구별할 수 있는지
        gt_loss_text = self.cross_entropy_loss(gt_logits.transpose(0,1), labels)  # (batch_size) text가 graph를 보고 잘 구별할 수 있는지

        if mode != 'train':
            self_mask = torch.eye(gt_logits.shape[0], device=gt_logits.device, dtype=torch.bool)
            comb_sim = torch.cat([gt_logits[self_mask][:,None], gt_logits.masked_fill(self_mask, -torch.inf)], dim=-1)
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
            self.log(mode + "_acc_top10", (sim_argsort < 10).float().mean())
            self.log(mode + "_acc_top50", (sim_argsort < 50).float().mean())
        return (gt_loss_graph + gt_loss_text) / 2

    def training_step(self, batch, batch_idx):
        graph_logits, text_logits = self(**batch)
        loss = self.clamp_loss(graph_logits, text_logits, 'train')
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        graph_logits, text_logits = self(**batch)
        loss = self.clamp_loss(graph_logits, text_logits, 'val')
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        graph_logits, text_logits = self(**batch)
        loss = self.clamp_loss(graph_logits, text_logits, 'test')
        self.log('test_loss', loss, prog_bar=True)
        return loss
       

class DeCLaMP(pl.LightningModule):
    def __init__(\
            self,
            cfg,
            graph_encoder: torch.nn.Module,
            text_encoder: torch.nn.Module,
            graph_encoder_dim: int,
            text_encoder_dim: int,
            clamp_dim: int,
            lr: float = 1e-4
        ):        
        super().__init__()
        self.save_hyperparameters(ignore=['graph_encoder', 'text_encoder'])
        self.cfg = cfg
        
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model_link)
        self.graph_augmentation_1 = GraphAttrMaskingAugmentation(cfg)
        self.graph_augmentation_2 = GraphPerturbationAugmentation(cfg)
        self.text_augmentation = TokenRandomMaskingAugmentation(cfg, self.tokenizer.mask_token)

        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.w_graph = torch.nn.Linear(graph_encoder_dim, clamp_dim)
        self.w_text = torch.nn.Linear(text_encoder_dim, clamp_dim)
        self.temperature = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, **inputs):
        '''
        graph_logits: (batch_size, clamp_dim)
        text_logits: (batch_size, clamp_dim)
        '''
        graph_logits = self.w_graph(self.graph_encoder(**inputs))
        text_logits = self.w_text(self.text_encoder(**inputs).last_hidden_state[:,0])
        
        all_graph_logits = F.normalize(self.all_gather(graph_logits, sync_grads=True).view(-1, graph_logits.shape[-1]))
        all_text_logits = F.normalize(self.all_gather(text_logits, sync_grads=True).view(-1, text_logits.shape[-1]))
        return all_graph_logits, all_text_logits
    
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        graphs, texts_1, texts_2 = batch
        graphs_1 = self.graph_augmentation_1(graphs)
        graphs_2 = self.graph_augmentation_2(graphs)
        texts_1 = self.text_augmentation(texts_1)
        texts_2 = self.text_augmentation(texts_2)
        return graphs_1, graphs_2, texts_1, texts_2       
    
    def clamp_loss(self, graph_logits, text_logits, mode):
        gt_logits = torch.matmul(graph_logits, text_logits.transpose(0,1)) * torch.exp(self.temp)  # (batch_size, batch_size)
        labels = torch.arange(gt_logits.shape[0], device=gt_logits.device)  # (batch_size)
        gt_loss_graph = self.cross_entropy_loss(gt_logits, labels)  # (batch_size) graph가 text를 보고 잘 구별할 수 있는지
        gt_loss_text = self.cross_entropy_loss(gt_logits.transpose(0,1), labels)  # (batch_size) text가 graph를 보고 잘 구별할 수 있는지

        if mode != 'train':
            self_mask = torch.eye(gt_logits.shape[0], device=gt_logits.device, dtype=torch.bool)
            comb_sim = torch.cat([gt_logits[self_mask][:,None], gt_logits.masked_fill(self_mask, -torch.inf)], dim=-1)
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
            self.log(mode + "_acc_top10", (sim_argsort < 10).float().mean())
            self.log(mode + "_acc_top50", (sim_argsort < 50).float().mean())
        return (gt_loss_graph + gt_loss_text) / 2

    def training_step(self, batch, batch_idx):
        graph_logits, text_logits = self(**batch)
        loss = self.clamp_loss(graph_logits, text_logits, 'train')
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        graph_logits, text_logits = self(**batch)
        loss = self.clamp_loss(graph_logits, text_logits, 'val')
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        graph_logits, text_logits = self(**batch)
        loss = self.clamp_loss(graph_logits, text_logits, 'test')
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}


class GNNSSL(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Instantiate the encoders and tokenizers
        self.graph_encoder = hydra.utils.instantiate(self.hparams.graph_encoder, _recursive_=False)
                
        # Instantiate the augmentation modules
        self.graph_aug_1 = hydra.utils.instantiate(self.hparams.graph_augmentation1, _recursive_=False)
        self.graph_aug_2 = hydra.utils.instantiate(self.hparams.graph_augmentation2, _recursive_=False)

        # Instantiate the loss module
        self.loss = JSDInfoMaxLoss(
            image_dim=self.hparams.graph_encoder.out_dim,
            text_dim=self.hparams.graph_encoder.out_dim,
            type="dot",
            prior_weight=0.1,
            image_prior=True,
            text_prior=True,
            visual_self_supervised=False,
            textual_self_supervised=False,
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        graphs, texts = batch
        # Augment the graphs and texts only if augmentation is enabled
        aug_graphs_1 = self.graph_aug_1(graphs) if self.hparams.augmentation else None
        aug_graphs_2 = self.graph_aug_2(graphs) if self.hparams.augmentation else None
        return aug_graphs_1, aug_graphs_2
    
    def forward(self, inputs):
        '''
        graph_logits: (batch_size, clamp_dim)
        text_logits: (batch_size, clamp_dim)
        '''
        aug_graphs_1, aug_graphs_2 = inputs
        graph_feat_1 = self.graph_encoder(aug_graphs_1)
        graph_feat_2 = self.graph_encoder(aug_graphs_2)
        return graph_feat_1, graph_feat_2
        
    def clamp_loss(self, graph_feat_1, graph_feat_2, mode):
        # prepare negative samples by randomly rolling batch
        # neg_graph_feat = aug_graph_feat if self.hparams.augmentation else graph_feat
        # neg_text_feat = aug_text_feat if self.hparams.augmentation else text_feat
        # rn = random.randint(1, self.hparams.datamodule.batch_size-1)
        # neg_graph_feat = torch.roll(neg_graph_feat, rn, dims=0)
        # neg_text_feat = torch.roll(neg_text_feat, rn, dims=0)
        
        # Compute Jensen-Shannon Divergence loss
        loss_dict, gt_logits = self.loss(
            image_features=graph_feat_1,
            text_features=graph_feat_2,
            neg_image_features=None,
            neg_text_features=None,
            aug_image_features=None,
            aug_text_features=None,
        )

        self.log_dict(loss_dict, prog_bar=False, on_step=True, on_epoch=True, batch_size=graph_feat_1.shape[0], sync_dist=True)
        if mode != 'train':
            self_mask = torch.eye(gt_logits.shape[0], device=gt_logits.device, dtype=torch.bool)
            comb_sim = torch.cat([gt_logits[self_mask][:,None], gt_logits.masked_fill(self_mask, -torch.inf)], dim=-1)
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), prog_bar=True, batch_size=graph_feat_1.shape[0], sync_dist=True)
            self.log(mode + "_acc_top3", (sim_argsort < 3).float().mean(), prog_bar=True, batch_size=graph_feat_1.shape[0], sync_dist=True)
            self.log(mode + "_acc_top10", (sim_argsort < 10).float().mean(), prog_bar=True, batch_size=graph_feat_1.shape[0], sync_dist=True)
            # self.log(mode + "_acc_top50", (sim_argsort < 50).float().mean())
        self.log(mode + "_loss", loss_dict['total_loss'], prog_bar=True, batch_size=graph_feat_1.shape[0], sync_dist=True)
        return loss_dict

    def training_step(self, batch, batch_idx):
        graph_feat_1, graph_feat_2 = self(batch)
        loss_dict = self.clamp_loss(graph_feat_1, graph_feat_2, 'train')
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        graph_feat_1, graph_feat_2 = self(batch)
        loss_dict = self.clamp_loss(graph_feat_1, graph_feat_2, 'val')
        return loss_dict['total_loss']
    
    def test_step(self, batch, batch_idx):
        graph_feat_1, graph_feat_2 = self(batch)
        loss_dict = self.clamp_loss(graph_feat_1, graph_feat_2, 'test')
        return loss_dict['total_loss']
    
    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optimizer.optimizer, 
            params=self.parameters(), 
            _convert_="partial"
        )
        if not self.hparams.optimizer.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}

class CLaMPLite(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Instantiate the encoders and tokenizers
        self.graph_encoder = hydra.utils.instantiate(self.hparams.graph_encoder, _recursive_=False)
        self.crystal_system_readout = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.graph_encoder.out_dim, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 7),
            torch.nn.Softmax(dim=-1)
        )
        self.text_encoder = BertForMaskedLM.from_pretrained(self.hparams.datamodule.tokenizer_model)
        self.text_encoder.config.output_hidden_states = True
        
        # # Lora Config
        # lora_alpha = 128
        # lora_dropout = 0.05
        # lora_r = 256
        
        # config = LoraConfig(
        #     r=lora_r,
        #     lora_alpha=lora_alpha,
        #     lora_dropout=lora_dropout,
        #     bias="none",
        #     task_type="SEQ_CLS",
        #     # target_modules=[
        #     #     "q_proj",
        #     # ]
        # )
        # self.text_encoder = get_peft_model(self.text_encoder, config)
        self.text_out_dim = BertConfig.from_pretrained(self.hparams.datamodule.tokenizer_model).hidden_size
        
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.datamodule.tokenizer_model)
        
        # Instantiate the augmentation modules
        self.graph_aug_1 = hydra.utils.instantiate(self.hparams.graph_augmentation1, _recursive_=False)
        self.graph_aug_2 = hydra.utils.instantiate(self.hparams.graph_augmentation2, _recursive_=False)
        self.text_aug = hydra.utils.instantiate(
            self.hparams.text_augmentation, _recursive_=False)

        # Instantiate the loss module
        self.loss = JSDInfoMaxLoss(
            image_dim=self.hparams.graph_encoder.out_dim,
            text_dim=self.text_out_dim,
            type="dot",
            prior_weight=0.1,
            image_prior=True,
            text_prior=True,
            visual_self_supervised=True,
            textual_self_supervised=True,
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        graphs, texts = batch
        # Augment the graphs and texts only if augmentation is enabled
        graphs = self.graph_aug_1(graphs) if self.hparams.augmentation else graphs
        aug_graphs = self.graph_aug_2(graphs) if self.hparams.augmentation else None
        texts = self.text_aug(texts) if self.hparams.augmentation else texts
        aug_texts = self.text_aug(texts) if self.hparams.augmentation else None
        return graphs, aug_graphs, texts, aug_texts
    
    def forward(self, inputs):
        '''
        graph_logits: (batch_size, clamp_dim)
        text_logits: (batch_size, clamp_dim)
        '''
        # Prepare batch
        graphs, aug_graphs, texts, aug_texts = inputs
        # encode graph
        graph_feat = self.graph_encoder(graphs)
        aug_graph_feat = self.graph_encoder(aug_graphs) if self.hparams.augmentation else None
        crys_loss_graph = F.cross_entropy(self.crystal_system_readout(graph_feat),  graphs.y)
        crys_loss_aug_graph = F.cross_entropy(self.crystal_system_readout(aug_graph_feat),  aug_graphs.y) if self.hparams.augmentation else None
        crys_loss = (crys_loss_graph + crys_loss_aug_graph) / 2 if self.hparams.augmentation else crys_loss_graph
        # encode text
        output = self.text_encoder(
            input_ids=texts['input_ids'], 
            attention_mask=texts['attention_mask'],
            token_type_ids=texts['token_type_ids'],
            labels=texts['labels']
            )
        loss_texts = output.loss
        text_feat = output.hidden_states[-1][:,0]
        output = self.text_encoder(
            input_ids=aug_texts['input_ids'], 
            attention_mask=aug_texts['attention_mask'],
            token_type_ids=aug_texts['token_type_ids'],
            labels=aug_texts['labels']
            ) if self.hparams.augmentation else None
        loss_aug_texts = output.loss if self.hparams.augmentation else None
        aug_text_feat = output.hidden_states[-1][:,0] if self.hparams.augmentation else None
        mlm_loss = (loss_texts + loss_aug_texts) / 2 if self.hparams.augmentation else loss_texts
        return graph_feat, text_feat, aug_graph_feat, aug_text_feat, mlm_loss, crys_loss
        
    def clamp_loss(self, graph_feat, text_feat, aug_graph_feat, aug_text_feat, mlm_loss, crys_loss, mode):
        # Compute Jensen-Shannon Divergence loss
        loss_dict, gt_logits = self.loss(
            image_features=graph_feat,
            text_features=text_feat,
            neg_image_features=None,
            neg_text_features=None,
            aug_image_features=aug_graph_feat,
            aug_text_features=aug_text_feat,
        )

        loss_dict['mlm_loss'] = mlm_loss
        loss_dict['crys_loss'] = crys_loss
        loss_dict['total_loss'] = loss_dict['total_loss'] + mlm_loss / 10 + crys_loss
        
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True, batch_size=graph_feat.shape[0], sync_dist=True)
        if mode != 'train':
            self_mask = torch.eye(gt_logits.shape[0], device=gt_logits.device, dtype=torch.bool)
            comb_sim = torch.cat([gt_logits[self_mask][:,None], gt_logits.masked_fill(self_mask, -torch.inf)], dim=-1)
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), prog_bar=True, batch_size=graph_feat.shape[0], sync_dist=True)
            self.log(mode + "_acc_top3", (sim_argsort < 3).float().mean(), prog_bar=True, batch_size=graph_feat.shape[0], sync_dist=True)
            # self.log(mode + "_acc_top10", (sim_argsort < 10).float().mean(), prog_bar=True, batch_size=graph_feat.shape[0], sync_dist=True)
            # self.log(mode + "_acc_top50", (sim_argsort < 50).float().mean())
        self.log(mode + "_loss", loss_dict['total_loss'], prog_bar=True, batch_size=graph_feat.shape[0], sync_dist=True)
        return loss_dict

    def training_step(self, batch, batch_idx):
        graph_feat, text_feat, aug_graph_feat, aug_text_feat, mlm_loss, crys_loss = self(batch)
        loss_dict = self.clamp_loss(graph_feat, text_feat, aug_graph_feat, aug_text_feat, mlm_loss, crys_loss,'train')
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        graph_feat, text_feat, aug_graph_feat, aug_text_feat, mlm_loss, crys_loss = self(batch)
        loss_dict = self.clamp_loss(graph_feat, text_feat, aug_graph_feat, aug_text_feat, mlm_loss, crys_loss,'val')
        return loss_dict['total_loss']
    
    def test_step(self, batch, batch_idx):
        graph_feat, text_feat, aug_graph_feat, aug_text_feat, mlm_loss, crys_loss = self(batch)
        loss_dict = self.clamp_loss(graph_feat, text_feat, aug_graph_feat, aug_text_feat, mlm_loss, crys_loss,'test')
        return loss_dict['total_loss']
    

class JSDInfoMaxLoss(nn.Module):
    '''
    Modified from
    https://github.com/4m4n5/CLIP-Lite
    '''
    def __init__(
        self,
        image_dim=512,
        text_dim=768,
        type="dot",
        prior_weight=0.1,
        image_prior=True,
        text_prior=False,
        visual_self_supervised=False,
        textual_self_supervised=False,
    ):
        super().__init__()

        # Settings to be saved for forward
        self.prior_weight = prior_weight
        self.image_prior = image_prior
        self.text_prior = text_prior

        if type == "concat":
            self.global_d = GlobalDiscriminator(sz=image_dim + text_dim)
            if visual_self_supervised:
                self.visual_d = GlobalDiscriminator(sz=image_dim + image_dim)
            if textual_self_supervised:
                self.textual_d = GlobalDiscriminator(sz=text_dim + text_dim)

        elif type == "dot":
            self.global_d = GlobalDiscriminatorDot(
                image_sz=image_dim,
                text_sz=text_dim,
            )
            if visual_self_supervised:
                self.visual_d = GlobalDiscriminatorDot(
                    image_sz=image_dim, text_sz=image_dim
                )
            if textual_self_supervised:
                self.textual_d = GlobalDiscriminatorDot(
                    image_sz=text_dim, text_sz=text_dim
                )

        elif type == "condot":
            self.global_d = GlobalDiscriminator(sz=image_dim + text_dim)
            if visual_self_supervised:
                self.visual_d = GlobalDiscriminatorDot(
                    image_sz=image_dim, text_sz=image_dim
                )
            if textual_self_supervised:
                self.textual_d = GlobalDiscriminatorDot(
                    image_sz=text_dim, text_sz=text_dim
                )

        elif type == "dotcon":
            self.global_d = GlobalDiscriminatorDot(
                image_sz=image_dim,
                text_sz=text_dim,
            )
            if visual_self_supervised:
                self.visual_d = GlobalDiscriminator(sz=image_dim + image_dim)
            if textual_self_supervised:
                self.textual_d = GlobalDiscriminator(sz=text_dim + text_dim)

        if self.image_prior:
            self.prior_d = PriorDiscriminator(sz=image_dim)
        if self.text_prior:
            self.text_prior_d = PriorDiscriminator(sz=text_dim)

    def forward(
        self,
        image_features,
        text_features,
        neg_image_features=None,
        neg_text_features=None,
        aug_image_features=None,
        aug_text_features=None,
    ):
        batch_size = image_features.shape[0]
        # Prior losses
        PRIOR = torch.tensor(0.0).cuda()
        # Image prior loss
        if self.image_prior:
            image_prior = torch.rand_like(image_features)
            term_a = torch.log(self.prior_d(image_prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(image_features)).mean()
            IMAGE_PRIOR = -(term_a + term_b)
            PRIOR = PRIOR + IMAGE_PRIOR
        # Text prior loss
        if self.text_prior:
            text_prior = torch.rand_like(text_features)
            term_a = torch.log(self.text_prior_d(text_prior)).mean()
            term_b = torch.log(1.0 - self.text_prior_d(text_features)).mean()
            TEXT_PRIOR = -(term_a + term_b)
            PRIOR = PRIOR + TEXT_PRIOR

        # Cross modal MI maximization loss
        # Normal mode
        if neg_text_features is None:
            # Positive pairs
            similarity, image_out, text_out = self.global_d(
                features1=image_features,
                features2=text_features,
            )
            Ej = -F.softplus(-similarity).mean()  # limited to positive value

            # Negative pairs
            # text_features_prime = torch.cat(
            #     (text_features[1:], text_features[0].unsqueeze(0)), dim=0
            # )  # Shift one
            i = torch.randint(1, batch_size, (1,)).item()
            text_features_prime = torch.roll(text_features.clone(), i, dims=0)
            similarity, _, _ = self.global_d(
                features1=image_features,
                features2=text_features_prime,
            )
            Em = F.softplus(similarity).mean()  # limited to negative value

        # Cluster mode
        elif neg_text_features is not None:
            # Positive pairs
            image_features_all = torch.cat(
                (image_features, neg_image_features), dim=0)
            text_features_all = torch.cat(
                (text_features, neg_text_features), dim=0)
            Ej = -F.softplus(
                -self.global_d(
                    features1=image_features_all,
                    features2=text_features_all,
                )
            ).mean()

            # Shuffle text_features so have half batch does not have hard negatives
            # text_features = torch.cat(
            #     (text_features[1:], text_features[0].unsqueeze(0)), dim=0
            # )  # Shift one

            i = torch.randint(1, batch_size, (1,)).item()
            text_features = torch.roll(text_features.clone(), i, dims=0)
            # Negative pairs
            text_features_prime_all = torch.cat(
                (neg_text_features, text_features), dim=0
            )
            Em = F.softplus(
                self.global_d(
                    features1=image_features_all,
                    features2=text_features_prime_all,
                )
            ).mean()

        CROSS_MODAL_LOSS = Em - Ej  # > 0 similarity of negative pairs are larger than positive pairs

        # Visual self supervised loss
        VISUAL_LOSS = torch.tensor(0.0).cuda()
        if aug_image_features is not None:
            # Positive pairs

            similarity, _, _ = self.visual_d(
                features1=image_features,
                features2=aug_image_features,
            )
            Ej = -F.softplus(-similarity).mean()
            # Negative pairs
            # aug_image_features_prime = torch.cat(
            #     (aug_image_features[1:], aug_image_features[0].unsqueeze(0)), dim=0
            # )
            i = torch.randint(1, batch_size, (1,)).item()
            aug_image_features_prime = torch.roll(aug_image_features.clone(), i, dims=0)
            similarity, _, _ = self.visual_d(
                features1=image_features,
                features2=aug_image_features_prime,
            )
            Em = F.softplus(similarity).mean()

            VISUAL_LOSS = Em - Ej

        # Textal self supervised loss
        TEXTUAL_LOSS = torch.tensor(0.0).cuda()
        if aug_text_features is not None:
            # Positive pairs
            similarity, _, _ = self.textual_d(
                features1=text_features,
                features2=aug_text_features,
            )
            Ej = -F.softplus(-similarity).mean()
            # Negative pairs
            # aug_text_features_prime = torch.cat(
            #     (aug_text_features[1:], aug_text_features[0].unsqueeze(0)), dim=0
            # )
            i = torch.randint(1, batch_size, (1,)).item()
            aug_text_features_prime = torch.roll(aug_text_features.clone(), i, dims=0)
            similarity, _, _ = self.textual_d(
                features1=text_features,
                features2=aug_text_features_prime,
            )
            Em = F.softplus(similarity).mean()

            TEXTUAL_LOSS = Em - Ej

        JSD_LOSS = CROSS_MODAL_LOSS + VISUAL_LOSS + TEXTUAL_LOSS
        TOTAL_LOSS = ((1.0 - self.prior_weight) * JSD_LOSS) + (
            self.prior_weight * PRIOR
        )

        it_logits = torch.matmul(image_out, text_out.transpose(0,1))

        loss_dict = {
            "total_loss": TOTAL_LOSS,
            "cross_modal_loss": CROSS_MODAL_LOSS,
            "visual_loss": VISUAL_LOSS,
            "textual_loss": TEXTUAL_LOSS,
        }

        return loss_dict, it_logits
    

class PriorDiscriminator(nn.Module):
    def __init__(self, sz):
        '''
        Distinguish between prior and real data. 
        features are trained to distinguishable from prior data during training.
        '''
        super(PriorDiscriminator, self).__init__()
        self.l0 = nn.Linear(sz, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))
    

class GlobalDiscriminator(nn.Module):
    '''
    Discriminator only depends on a neural network with 3 linear layers.
    '''
    def __init__(self, sz):
        super(GlobalDiscriminator, self).__init__()
        self.l0 = nn.Linear(sz, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, features1, features2):
        x = torch.cat((features1, features2), dim=1)
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))

        return self.l2(h)


class GlobalDiscriminatorDot(nn.Module):
    '''
    Discriminator depends on dot product of two features.
    Before dot product, we normalize the features to unit length.
    '''
    def __init__(self, image_sz, text_sz, units=512, bln=True):
        super(GlobalDiscriminatorDot, self).__init__()
        self.img_block = MILinearBlock(image_sz, units=units, bln=bln)
        self.text_block = MILinearBlock(text_sz, units=units, bln=bln)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        features1=None,
        features2=None,
    ):

        # Computer cross modal loss
        feat1 = self.img_block(features1)
        feat2 = self.text_block(features2)

        feat1, feat2 = map(lambda t: F.normalize(
            t, p=2, dim=-1), (feat1, feat2))

        # ## Method 1
        # # Dot product and sum
        # o = torch.sum(feat1 * feat2, dim=1) * self.temperature.exp()

        # ## Method 2
        # o = self.cos(feat1, feat2) * self.temperature.exp()

        # Method 3
        o = einsum("n d, n d -> n", feat1, feat2) * self.temperature.exp()

        return o, feat1, feat2
    

class MILinearBlock(nn.Module):
    '''
    To stabilize training, we combine the nonlinear projection with a linear shortcut projection.
    '''
    def __init__(self, feature_sz, units=512, bln=True):
        super(MILinearBlock, self).__init__()
        # Pre-dot product encoder for "Encode and Dot" arch for 1D feature maps
        self.feature_nonlinear = nn.Sequential(
            nn.Linear(feature_sz, units, bias=False),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Linear(units, units),
        )
        self.feature_shortcut = nn.Linear(feature_sz, units)
        self.feature_block_ln = nn.LayerNorm(units)

        # initialize the initial projection to a sort of noisy copy
        eye_mask = torch.zeros((units, feature_sz), dtype=torch.bool)
        for i in range(min(feature_sz, units)):
            eye_mask[i, i] = 1

        self.feature_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.feature_shortcut.weight.data.masked_fill_(
            eye_mask, 1.0)
        self.bln = bln

    def forward(self, feat):
        f = self.feature_nonlinear(feat) + self.feature_shortcut(feat)
        if self.bln:
            f = self.feature_block_ln(f)

        return f


class GraphSupervisedLearning(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Instantiate the encoders and tokenizers
        if self.hparams.fine_tune_from is not None:
            if self.hparams.base_model_type == 'clamp':
                model = CLaMPLite.load_from_checkpoint(self.hparams.fine_tune_from, map_location={'cuda:0': 'cpu'})
            elif self.hparams.base_model_type == 'ssl':
                model = GNNSSL.load_from_checkpoint(self.hparams.fine_tune_from, map_location={'cuda:0': 'cpu'})
            self.graph_encoder = model.graph_encoder
        else:
            self.graph_encoder = hydra.utils.instantiate(self.hparams.graph_encoder, _recursive_=False)
        self.projection_head = nn.Linear(self.hparams.graph_encoder.out_dim, self.hparams.num_classes)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        graphs = batch
        return graphs, graphs.y
    
    def forward(self, inputs):
        '''
        graph_logits: (batch_size, graph_encoder_dim)
        '''
        return self.projection_head(self.graph_encoder(inputs))
        
    def supervised_loss(self, y_pred, label, mode):
        if self.task == 'classification':
            loss = F.cross_entropy(y_pred, label)
        elif self.task == 'regression':
            loss = F.mse_loss(y_pred.squeeze(), label.squeeze())
        self.log(mode + "_loss", loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=y_pred.shape[0], sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        graph, y = batch
        y_pred = self(graph)
        loss = self.supervised_loss(y_pred, y, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        graph, y = batch
        y_pred = self(graph)
        loss = self.supervised_loss(y_pred, y, 'val')
        if self.task == 'classification':
            predicted_labels = torch.argmax(y_pred, dim=1)
            correct = (predicted_labels == y).sum().item()
            loss_dict = {'val_loss': loss, 'correct_count': correct, 'total_count': len(y)}
        elif self.task == 'regression':
            loss_dict = {'val_loss': loss, 'total_count': len(y)}
        self.validation_step_outputs.append(loss_dict)
        return loss_dict
        
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True)
        if self.task == 'classification':
            total_correct = sum(x['correct_count'] for x in outputs)
            total_count = sum(x['total_count'] for x in outputs)
            accuracy = total_correct / total_count
            self.log('val_acc', accuracy, prog_bar=True)
        self.validation_step_outputs = []
    
    def test_step(self, batch, batch_idx):
        graph, y = batch
        y_pred = self(graph)
        loss = self.supervised_loss(y_pred, y, 'test')
        if self.task == 'classification':
            predicted_labels = torch.argmax(y_pred, dim=1)
            correct = (predicted_labels == y).sum().item()
            loss_dict = {'test_loss': loss, 'correct_count': correct, 'total_count': len(y)}
        elif self.task == 'regression':
            loss_dict = {'test_loss': loss, 'total_count': len(y)}
        self.test_step_outputs.append(loss_dict)
        return loss_dict
        
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss)
        if self.task == 'classification':
            total_correct = sum(x['correct_count'] for x in outputs)
            total_count = sum(x['total_count'] for x in outputs)
            accuracy = total_correct / total_count
            self.log('test_acc', accuracy)
        self.test_step_outputs = []
    
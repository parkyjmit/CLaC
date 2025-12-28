from typing import Any, Dict, List, Mapping
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, AutoTokenizer
from data.augmentation import GraphAttrMaskingAugmentation, GraphPerturbationAugmentation, TokenRandomMaskingAugmentation
from model.text_encoder import TextEncoder, CLaCTokenizer
import random
# from peft import get_peft_model, LoraConfig


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

    
class CLaC(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.graph_encoder = hydra.utils.instantiate(self.hparams.graph_encoder)
        self.text_encoder = hydra.utils.instantiate(self.hparams.text_encoder)
        
        self.tokenizer = hydra.utils.instantiate(self.hparams.tokenizer)

        self.graph_augmentation = hydra.utils.instantiate(self.hparams.graph_augmentation)
        self.text_augmentation = hydra.utils.instantiate(self.hparams.text_augmentation, mask_token=self.tokenizer.mask_token)

        self.w_graph = torch.nn.Linear(self.hparams.graph_encoder.hidden_dim, self.hparams.clac_dim)
        self.w_text = torch.nn.Linear(self.hparams.text_encoder.hidden_dim, self.hparams.clac_dim)

        self.temperature = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, **inputs):
        '''
        graph_logits: (batch_size, clac_dim)
        text_logits: (batch_size, clac_dim)
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
    
    def clac_loss(self, graph_logits, text_logits, mode):
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
        loss = self.clac_loss(graph_logits, text_logits, 'train')
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        graph_logits, text_logits = self(**batch)
        loss = self.clac_loss(graph_logits, text_logits, 'val')
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        graph_logits, text_logits = self(**batch)
        loss = self.clac_loss(graph_logits, text_logits, 'test')
        self.log('test_loss', loss, prog_bar=True)
        return loss
       

# class DeCLaC(pl.LightningModule):
#     def __init__(\
#             self,
#             cfg,
#             graph_encoder: torch.nn.Module,
#             text_encoder: torch.nn.Module,
#             graph_encoder_dim: int,
#             text_encoder_dim: int,
#             clac_dim: int,
#             lr: float = 1e-4
#         ):        
#         super().__init__()
#         self.save_hyperparameters(ignore=['graph_encoder', 'text_encoder'])
#         self.cfg = cfg
        
#         self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model_link)
#         self.graph_augmentation_1 = GraphAttrMaskingAugmentation(cfg)
#         self.graph_augmentation_2 = GraphPerturbationAugmentation(cfg)
#         self.text_augmentation = TokenRandomMaskingAugmentation(cfg, self.tokenizer.mask_token)

#         self.graph_encoder = graph_encoder
#         self.text_encoder = text_encoder
#         self.w_graph = torch.nn.Linear(graph_encoder_dim, clac_dim)
#         self.w_text = torch.nn.Linear(text_encoder_dim, clac_dim)
#         self.temperature = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

#     def forward(self, **inputs):
#         '''
#         graph_logits: (batch_size, clac_dim)
#         text_logits: (batch_size, clac_dim)
#         '''
#         graph_logits = self.w_graph(self.graph_encoder(**inputs))
#         text_logits = self.w_text(self.text_encoder(**inputs).last_hidden_state[:,0])
        
#         all_graph_logits = F.normalize(self.all_gather(graph_logits, sync_grads=True).view(-1, graph_logits.shape[-1]))
#         all_text_logits = F.normalize(self.all_gather(text_logits, sync_grads=True).view(-1, text_logits.shape[-1]))
#         return all_graph_logits, all_text_logits
    
#     def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
#         graphs, texts_1, texts_2 = batch
#         graphs_1 = self.graph_augmentation_1(graphs)
#         graphs_2 = self.graph_augmentation_2(graphs)
#         texts_1 = self.text_augmentation(texts_1)
#         texts_2 = self.text_augmentation(texts_2)
#         return graphs_1, graphs_2, texts_1, texts_2       
    
#     def clac_loss(self, graph_logits, text_logits, mode):
#         gt_logits = torch.matmul(graph_logits, text_logits.transpose(0,1)) * torch.exp(self.temp)  # (batch_size, batch_size)
#         labels = torch.arange(gt_logits.shape[0], device=gt_logits.device)  # (batch_size)
#         gt_loss_graph = self.cross_entropy_loss(gt_logits, labels)  # (batch_size) graph가 text를 보고 잘 구별할 수 있는지
#         gt_loss_text = self.cross_entropy_loss(gt_logits.transpose(0,1), labels)  # (batch_size) text가 graph를 보고 잘 구별할 수 있는지

#         if mode != 'train':
#             self_mask = torch.eye(gt_logits.shape[0], device=gt_logits.device, dtype=torch.bool)
#             comb_sim = torch.cat([gt_logits[self_mask][:,None], gt_logits.masked_fill(self_mask, -torch.inf)], dim=-1)
#             sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
#             self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
#             self.log(mode + "_acc_top10", (sim_argsort < 10).float().mean())
#             self.log(mode + "_acc_top50", (sim_argsort < 50).float().mean())
#         return (gt_loss_graph + gt_loss_text) / 2

#     def training_step(self, batch, batch_idx):
#         graph_logits, text_logits = self(**batch)
#         loss = self.clac_loss(graph_logits, text_logits, 'train')
#         self.log('train_loss', loss, prog_bar=True)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         graph_logits, text_logits = self(**batch)
#         loss = self.clac_loss(graph_logits, text_logits, 'val')
#         self.log('val_loss', loss, prog_bar=True)
#         return loss
    
#     def test_step(self, batch, batch_idx):
#         graph_logits, text_logits = self(**batch)
#         loss = self.clac_loss(graph_logits, text_logits, 'test')
#         self.log('test_loss', loss, prog_bar=True)
#         return loss
    
#     def configure_optimizers(self) -> OptimizerLRScheduler:
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)        
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
#         return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}


class GNNSSL(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Instantiate the encoders and tokenizers
        self.graph_encoder = hydra.utils.instantiate(self.hparams.graph_encoder, _recursive_=False)
                
        # Instantiate the augmentation modules
        self.graph_aug_1 = hydra.utils.instantiate(self.hparams.graph_augmentation1, _recursive_=False)
        self.graph_aug_2 = hydra.utils.instantiate(self.hparams.graph_augmentation2, _recursive_=False)

        # Instantiate the loss module via Hydra for ablation flexibility
        # Use graph encoder out_dim for both branches in SSL
        self.loss = hydra.utils.instantiate(
            self.hparams.loss,
            image_dim=self.hparams.graph_encoder.out_dim,
            text_dim=self.hparams.graph_encoder.out_dim,
            _recursive_=False,
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        graphs, texts = batch
        # Augment the graphs and texts only if augmentation is enabled
        aug_graphs_1 = self.graph_aug_1(graphs) if self.hparams.augmentation else None
        aug_graphs_2 = self.graph_aug_2(graphs) if self.hparams.augmentation else None
        return aug_graphs_1, aug_graphs_2
    
    def forward(self, inputs):
        '''
        graph_logits: (batch_size, clac_dim)
        text_logits: (batch_size, clac_dim)
        '''
        aug_graphs_1, aug_graphs_2 = inputs
        graph_feat_1 = self.graph_encoder(aug_graphs_1)
        graph_feat_2 = self.graph_encoder(aug_graphs_2)
        return graph_feat_1, graph_feat_2
        
    def clac_loss(self, graph_feat_1, graph_feat_2, mode):
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
        loss_dict = self.clac_loss(graph_feat_1, graph_feat_2, 'train')
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        graph_feat_1, graph_feat_2 = self(batch)
        loss_dict = self.clac_loss(graph_feat_1, graph_feat_2, 'val')
        return loss_dict['total_loss']
    
    def test_step(self, batch, batch_idx):
        graph_feat_1, graph_feat_2 = self(batch)
        loss_dict = self.clac_loss(graph_feat_1, graph_feat_2, 'test')
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

class CLaCLite(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Instantiate the encoders and tokenizers
        self.graph_encoder = hydra.utils.instantiate(self.hparams.graph_encoder, _recursive_=False)

        self.text_encoder = TextEncoder(self.hparams.datamodule.tokenizer_model)
        self.text_out_dim = self.text_encoder.config.hidden_size
        
        self.tokenizer = CLaCTokenizer(self.hparams.datamodule.tokenizer_model)

        # Text augmentation module with vocab_size and special_token_ids for MLM
        # Override mask_token with the actual mask_token_id from tokenizer if not set in config
        text_aug_config = dict(self.hparams.text_augmentation)
        if 'mask_token' not in text_aug_config and self.tokenizer.mask_token_id is not None:
            text_aug_config['mask_token'] = self.tokenizer.mask_token_id

        self.text_aug = hydra.utils.instantiate(
            text_aug_config,
            vocab_size=self.tokenizer.vocab_size,
            special_token_ids=self.tokenizer.get_special_token_ids(),
            _recursive_=False)

        # Instantiate the loss module via Hydra for ablation flexibility
        # Support both old (use_intramodal_loss) and new (separate visual/textual) configs
        # for backward compatibility
        if hasattr(self.hparams, "use_visual_intramodal_loss") or hasattr(self.hparams, "use_textual_intramodal_loss"):
            # New fine-grained control
            self.use_visual_intramodal_loss = getattr(self.hparams, "use_visual_intramodal_loss", True)
            self.use_textual_intramodal_loss = getattr(self.hparams, "use_textual_intramodal_loss", True)
        else:
            # Backward compatibility: use_intramodal_loss applies to both
            use_intramodal_loss = getattr(self.hparams, "use_intramodal_loss", True)
            self.use_visual_intramodal_loss = use_intramodal_loss
            self.use_textual_intramodal_loss = use_intramodal_loss

        self.loss = hydra.utils.instantiate(
            self.hparams.loss,
            image_dim=self.hparams.graph_encoder.out_dim,
            text_dim=self.text_out_dim,
            _recursive_=False,
        )
        # Language model loss weighting for ablation (default 0.1)
        # Encoder: MLM (Masked Language Modeling), Decoder: CLM (Causal Language Modeling)
        self.lm_weight = getattr(self.hparams, "lm_weight", 0.1)

    def on_load_checkpoint(self, checkpoint):
        """
        Automatically convert old checkpoint keys to new format for backward compatibility.
        This allows loading checkpoints trained with 'use_intramodal_loss' to work with
        the new 'use_visual_intramodal_loss' and 'use_textual_intramodal_loss' keys.
        """
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']

            # Convert old use_intramodal_loss to new format
            if 'use_intramodal_loss' in hparams and 'use_visual_intramodal_loss' not in hparams:
                use_intramodal = hparams['use_intramodal_loss']
                hparams['use_visual_intramodal_loss'] = use_intramodal
                hparams['use_textual_intramodal_loss'] = use_intramodal

    def encode_text(self, texts: dict):
        """
        Encapsulates the logic for encoding text tokens into features, including label preparation.

        Training mode (self.training == True):
            - Encoder: MLM with masking (15%, 80/10/10 rule)
            - Decoder: CLM

        Evaluation mode (self.training == False):
            - Encoder: Self-reconstruction (all tokens, no masking)
            - Decoder: CLM (same as training)
        """
        def _ensure_labels_encoder_train(batch: dict) -> dict:
            """For encoder models (BERT) during training: apply MLM masking."""
            if 'labels' in batch:
                return batch
            batch = {k: v for k, v in batch.items()}

            # Apply MLM masking for encoder models
            input_ids = batch['input_ids'].clone()
            labels = batch['input_ids'].clone()

            # Create probability matrix for masking (15% by default)
            # IMPORTANT: Must be on the same device as input_ids
            probability_matrix = torch.full(input_ids.shape, 0.15, device=input_ids.device)

            # Don't mask special tokens (PAD, CLS, SEP, etc.)
            special_token_ids = self.tokenizer.get_special_token_ids()
            special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in special_token_ids:
                special_tokens_mask |= (input_ids == special_token_id)
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

            # Select tokens to mask
            masked_indices = torch.bernoulli(probability_matrix).bool()
            # Set labels: only compute loss on masked tokens
            labels[~masked_indices] = -100

            # 80% of the time: replace with [MASK]
            indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=input_ids.device)).bool() & masked_indices
            input_ids[indices_replaced] = self.tokenizer.mask_token_id

            # 10% of the time: replace with random word
            indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(
                self.tokenizer.vocab_size,
                input_ids.shape,
                dtype=input_ids.dtype,
                device=input_ids.device
            )
            input_ids[indices_random] = random_words[indices_random]

            # Remaining 10%: keep original token

            batch['labels'] = labels
            batch['input_ids'] = input_ids
            return batch

        def _ensure_labels_encoder_eval(batch: dict) -> dict:
            """For encoder models (BERT) during evaluation: no masking, all tokens."""
            if 'labels' in batch:
                return batch
            batch = {k: v for k, v in batch.items()}

            # No masking: use original input_ids
            labels = batch['input_ids'].clone()

            # Only ignore padding tokens
            if 'attention_mask' in batch:
                labels = labels.masked_fill(batch['attention_mask'] == 0, -100)

            batch['labels'] = labels
            # input_ids remain unchanged (no masking)
            return batch

        def _ensure_labels_decoder(batch: dict) -> dict:
            """For decoder models (GPT, OPT), use input_ids as labels for CLM."""
            if 'labels' in batch:
                return batch
            batch = {k: v for k, v in batch.items()}
            labels = batch['input_ids'].clone()
            # For decoder models, padding tokens should be ignored
            if 'attention_mask' in batch:
                labels = labels.masked_fill(batch['attention_mask'] == 0, -100)
            batch['labels'] = labels
            return batch

        # Choose the appropriate label preparation based on model type and training mode
        if self.text_encoder.is_encoder_model:
            if self.training:
                text_inputs = _ensure_labels_encoder_train(texts)
            else:
                text_inputs = _ensure_labels_encoder_eval(texts)
        elif self.text_encoder.is_decoder_model:
            text_inputs = _ensure_labels_decoder(texts)
        else:
            # Fallback: use decoder logic
            print(f"WARNING: Unknown model type for {self.text_encoder.pretrained_model_name_or_path}. Using decoder logic as fallback.")
            text_inputs = _ensure_labels_decoder(texts)

        output = self.text_encoder(**text_inputs)
        # Return both the loss and the features
        return output.loss, output.hidden_states[-1][:, self.text_encoder.feature_idx]

    def forward(self, graphs, texts, aug_graphs=None, aug_texts=None, do_text_aug=False):
        '''
        Processes graph and text data, with optional augmentations, to produce features and LM loss.
        '''
        # Apply text augmentation if specified (only for encoder models)
        # Decoder models use CLM and don't need MLM-style masking augmentation
        if do_text_aug and self.text_encoder.is_encoder_model:
            texts = self.text_aug(texts)
            if aug_texts:
                aug_texts = self.text_aug(aug_texts)

        # Encode base graph and text
        graph_feat = self.graph_encoder(graphs)
        loss_texts, text_feat = self.encode_text(texts)

        # Handle augmented data if provided
        aug_graph_feat, aug_text_feat, loss_aug_texts = None, None, None
        if aug_graphs is not None:
            aug_graph_feat = self.graph_encoder(aug_graphs)
        if aug_texts is not None:
            loss_aug_texts, aug_text_feat = self.encode_text(aug_texts)

        lm_loss = (loss_texts + loss_aug_texts) / 2 if loss_aug_texts is not None else loss_texts

        return graph_feat, text_feat, aug_graph_feat, aug_text_feat, lm_loss
        
    def clac_loss(self, graph_feat, text_feat, aug_graph_feat, aug_text_feat, lm_loss, mode):
        # Compute Jensen-Shannon Divergence loss
        loss_dict, gt_logits = self.loss(
            image_features=graph_feat,
            text_features=text_feat,
            neg_image_features=None,
            neg_text_features=None,
            aug_image_features=aug_graph_feat if self.use_visual_intramodal_loss else None,
            aug_text_features=aug_text_feat if self.use_textual_intramodal_loss else None,
        )

        loss_dict['lm_loss'] = lm_loss
        loss_dict['total_loss'] = loss_dict['total_loss'] + self.lm_weight * lm_loss
        
        on_step = True if mode == 'train' else False
        self.log_dict(loss_dict, prog_bar=False, on_step=on_step, on_epoch=not on_step, batch_size=graph_feat.shape[0], sync_dist=True)
        if mode != 'train':
            self_mask = torch.eye(gt_logits.shape[0], device=gt_logits.device, dtype=torch.bool)
            comb_sim = torch.cat([gt_logits[self_mask][:,None], gt_logits.masked_fill(self_mask, -torch.inf)], dim=-1)
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), prog_bar=True, batch_size=graph_feat.shape[0], sync_dist=True)
            self.log(mode + "_acc_top3", (sim_argsort < 3).float().mean(), prog_bar=True, batch_size=graph_feat.shape[0], sync_dist=True)
        self.log(mode + "_loss", loss_dict['total_loss'], prog_bar=True, on_step=on_step, on_epoch=not on_step, batch_size=graph_feat.shape[0], sync_dist=True)
        return loss_dict

    def training_step(self, batch, batch_idx):
        graphs1, graphs2, texts1, texts2 = batch
        graph_feat, text_feat, aug_graph_feat, aug_text_feat, lm_loss = self.forward(
            graphs=graphs1,
            texts=texts1,
            aug_graphs=graphs2 if self.use_visual_intramodal_loss else None,
            aug_texts=texts2 if self.use_textual_intramodal_loss else None,
            do_text_aug=self.hparams.augmentation
        )
        loss_dict = self.clac_loss(graph_feat, text_feat, aug_graph_feat, aug_text_feat, lm_loss, 'train')
        return loss_dict['total_loss']

    def validation_step(self, batch, batch_idx):
        graphs, texts = batch
        graph_feat, text_feat, _, _, lm_loss = self.forward(
            graphs=graphs,
            texts=texts,
            do_text_aug=False
        )
        loss_dict = self.clac_loss(graph_feat, text_feat, None, None, lm_loss, 'val')
        return loss_dict['total_loss']

    def test_step(self, batch, batch_idx):
        graphs, texts = batch
        graph_feat, text_feat, _, _, lm_loss = self.forward(
            graphs=graphs,
            texts=texts,
            do_text_aug=False
        )
        loss_dict = self.clac_loss(graph_feat, text_feat, None, None, lm_loss, 'test')
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
        **kwargs,
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

    def _calculate_jsd_loss(self, discriminator, features1, features2):
        batch_size = features1.shape[0]
        # Positive pairs
        similarity, f1_out, f2_out = discriminator(features1=features1, features2=features2)
        Ej = -F.softplus(-similarity).mean()

        # Negative pairs
        i = torch.randint(1, batch_size, (1,)).item()
        features2_prime = torch.roll(features2.clone(), i, dims=0)
        similarity, _, _ = discriminator(features1=features1, features2=features2_prime)
        Em = F.softplus(similarity).mean()
        return Em - Ej, f1_out, f2_out

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
        if self.image_prior:
            image_prior = torch.rand_like(image_features)
            term_a = torch.log(self.prior_d(image_prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(image_features)).mean()
            IMAGE_PRIOR = -(term_a + term_b)
            PRIOR = PRIOR + IMAGE_PRIOR
        if self.text_prior:
            text_prior = torch.rand_like(text_features)
            term_a = torch.log(self.text_prior_d(text_prior)).mean()
            term_b = torch.log(1.0 - self.text_prior_d(text_features)).mean()
            TEXT_PRIOR = -(term_a + term_b)
            PRIOR = PRIOR + TEXT_PRIOR

        # --- Integrated Cross-modal loss ---
        cross_modal_loss_terms = []
        # Base loss (I, T)
        loss_i_t, image_out, text_out = self._calculate_jsd_loss(self.global_d, image_features, text_features)
        cross_modal_loss_terms.append(loss_i_t)

        if aug_image_features is not None:
            loss_ai_t, _, _ = self._calculate_jsd_loss(self.global_d, aug_image_features, text_features)
            cross_modal_loss_terms.append(loss_ai_t)
        if aug_text_features is not None:
            loss_i_at, _, _ = self._calculate_jsd_loss(self.global_d, image_features, aug_text_features)
            cross_modal_loss_terms.append(loss_i_at)
        if aug_image_features is not None and aug_text_features is not None:
            loss_ai_at, _, _ = self._calculate_jsd_loss(self.global_d, aug_image_features, aug_text_features)
            cross_modal_loss_terms.append(loss_ai_at)

        CROSS_MODAL_LOSS = torch.sum(torch.stack(cross_modal_loss_terms))

        # --- Visual self supervised loss ---
        VISUAL_LOSS = torch.tensor(0.0).cuda()
        if aug_image_features is not None:
            VISUAL_LOSS, _, _ = self._calculate_jsd_loss(self.visual_d, image_features, aug_image_features)

        # --- Textual self supervised loss ---
        TEXTUAL_LOSS = torch.tensor(0.0).cuda()
        if aug_text_features is not None:
            TEXTUAL_LOSS, _, _ = self._calculate_jsd_loss(self.textual_d, text_features, aug_text_features)

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


class GlobalDiscriminatorSim(nn.Module):
    """
    Computes similarity matrix for InfoNCE-style loss after projecting features.
    """

    def __init__(self, image_sz, text_sz, temperature, units=512, bln=True):
        super().__init__()
        self.img_block = MILinearBlock(image_sz, units=units, bln=bln)
        self.text_block = MILinearBlock(text_sz, units=units, bln=bln)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, features1, features2):
        feat1 = self.img_block(features1)
        feat2 = self.text_block(features2)

        feat1 = F.normalize(feat1, p=2, dim=-1)
        feat2 = F.normalize(feat2, p=2, dim=-1)

        similarity = torch.matmul(feat1, feat2.T) * self.temperature.exp()
        return similarity, feat1, feat2


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        temperature=0.07,
        visual_self_supervised=False,
        textual_self_supervised=False,
        **kwargs,  # to ignore other args from yaml
    ):
        super().__init__()
        self.temperature = temperature
        self.visual_self_supervised = visual_self_supervised
        self.textual_self_supervised = textual_self_supervised
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # Use the new GlobalDiscriminatorSim for each path
        self.global_d = GlobalDiscriminatorSim(
            image_sz=image_dim, text_sz=text_dim, temperature=self.temperature
        )

        if self.visual_self_supervised:
            self.visual_d = GlobalDiscriminatorSim(
                image_sz=image_dim, text_sz=image_dim, temperature=self.temperature
            )
        if self.textual_self_supervised:
            self.textual_d = GlobalDiscriminatorSim(
                image_sz=text_dim, text_sz=text_dim, temperature=self.temperature
            )

    def forward(
        self,
        image_features,
        text_features,
        neg_image_features=None,
        neg_text_features=None,
        aug_image_features=None,
        aug_text_features=None,
    ):
        labels = torch.arange(len(image_features)).to(image_features.device)

        # --- Integrated Cross-modal loss ---
        cross_modal_loss_terms = []
        # Base loss (I, T)
        logits_i_t, _, _ = self.global_d(image_features, text_features)
        loss_i_t = (self.cross_entropy_loss(logits_i_t, labels) + self.cross_entropy_loss(logits_i_t.T, labels)) / 2
        cross_modal_loss_terms.append(loss_i_t)

        # Expanded losses if augmentations exist
        if aug_image_features is not None:
            logits_ai_t, _, _ = self.global_d(aug_image_features, text_features)
            loss_ai_t = (self.cross_entropy_loss(logits_ai_t, labels) + self.cross_entropy_loss(logits_ai_t.T, labels)) / 2
            cross_modal_loss_terms.append(loss_ai_t)

        if aug_text_features is not None:
            logits_i_at, _, _ = self.global_d(image_features, aug_text_features)
            loss_i_at = (self.cross_entropy_loss(logits_i_at, labels) + self.cross_entropy_loss(logits_i_at.T, labels)) / 2
            cross_modal_loss_terms.append(loss_i_at)

        if aug_image_features is not None and aug_text_features is not None:
            logits_ai_at, _, _ = self.global_d(aug_image_features, aug_text_features)
            loss_ai_at = (self.cross_entropy_loss(logits_ai_at, labels) + self.cross_entropy_loss(logits_ai_at.T, labels)) / 2
            cross_modal_loss_terms.append(loss_ai_at)
        
        cross_modal_loss = torch.sum(torch.stack(cross_modal_loss_terms))
        total_loss = cross_modal_loss 

        # --- Visual self-supervised loss (Symmetric) ---
        visual_loss = torch.tensor(0.0).to(total_loss.device)
        if self.visual_self_supervised and aug_image_features is not None:
            visual_logits, _, _ = self.visual_d(image_features, aug_image_features)
            loss_v1 = self.cross_entropy_loss(visual_logits, labels)
            loss_v2 = self.cross_entropy_loss(visual_logits.T, labels)
            visual_loss = (loss_v1 + loss_v2) / 2
            total_loss += visual_loss

        # --- Textual self-supervised loss (Symmetric) ---
        textual_loss = torch.tensor(0.0).to(total_loss.device)
        if self.textual_self_supervised and aug_text_features is not None:
            textual_logits, _, _ = self.textual_d(text_features, aug_text_features)
            loss_t1 = self.cross_entropy_loss(textual_logits, labels)
            loss_t2 = self.cross_entropy_loss(textual_logits.T, labels)
            textual_loss = (loss_t1 + loss_t2) / 2
            total_loss += textual_loss

        loss_dict = {
            "total_loss": total_loss,
            "cross_modal_loss": cross_modal_loss,
            "visual_loss": visual_loss,
            "textual_loss": textual_loss,
        }

        # Return original logits for accuracy calculation
        return loss_dict, logits_i_t

    

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
            # Load checkpoint with strict option
            strict_loading = getattr(self.hparams, 'strict_loading', True)

            if self.hparams.base_model_type == 'clac':
                model = CLaCLite.load_from_checkpoint(
                    self.hparams.fine_tune_from,
                    map_location={'cuda:0': 'cpu'},
                    strict=strict_loading
                )
            elif self.hparams.base_model_type == 'ssl':
                model = GNNSSL.load_from_checkpoint(
                    self.hparams.fine_tune_from,
                    map_location={'cuda:0': 'cpu'},
                    strict=strict_loading
                )
            self.graph_encoder = model.graph_encoder

            # Freeze graph encoder if specified
            freeze_graph_encoder = getattr(self.hparams, 'freeze_graph_encoder', False)
            if freeze_graph_encoder:
                for param in self.graph_encoder.parameters():
                    param.requires_grad = False
                print(f"Graph encoder frozen: all parameters set to requires_grad=False")

            # Freeze specific layers if specified
            freeze_layers = getattr(self.hparams, 'freeze_layers', None)
            if freeze_layers is not None and not freeze_graph_encoder:
                self._freeze_specific_layers(freeze_layers)
        else:
            self.graph_encoder = hydra.utils.instantiate(self.hparams.graph_encoder, _recursive_=False)

        self.projection_head = nn.Linear(self.hparams.graph_encoder.out_dim, self.hparams.num_classes)
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Get task from datamodule config
        self.task = getattr(self.hparams.datamodule, 'task', 'regression')

    def _freeze_specific_layers(self, freeze_layers):
        """Freeze specific layers in the graph encoder.

        Args:
            freeze_layers: List of layer names or indices to freeze
        """
        # For CGCNN: convs[0], convs[1], etc.
        # For PaiNN: list_message[0], list_update[0], etc.
        # For ORB: orb_model.gnn_stacks[0], orb_model.atom_emb, etc.

        if hasattr(self.graph_encoder, 'convs'):  # CGCNN
            for idx in freeze_layers:
                if isinstance(idx, int) and idx < len(self.graph_encoder.convs):
                    for param in self.graph_encoder.convs[idx].parameters():
                        param.requires_grad = False
                    print(f"Frozen CGCNN layer: convs[{idx}]")
        elif hasattr(self.graph_encoder, 'list_message'):  # PaiNN
            for idx in freeze_layers:
                if isinstance(idx, int):
                    if idx < len(self.graph_encoder.list_message):
                        for param in self.graph_encoder.list_message[idx].parameters():
                            param.requires_grad = False
                        for param in self.graph_encoder.list_update[idx].parameters():
                            param.requires_grad = False
                        print(f"Frozen PaiNN layer: list_message[{idx}] and list_update[{idx}]")
        elif hasattr(self.graph_encoder, 'orb_model'):  # ORB
            for item in freeze_layers:
                if isinstance(item, int):
                    # Freeze specific GNN stack layer
                    if hasattr(self.graph_encoder.orb_model, 'gnn_stacks') and item < len(self.graph_encoder.orb_model.gnn_stacks):
                        for param in self.graph_encoder.orb_model.gnn_stacks[item].parameters():
                            param.requires_grad = False
                        print(f"Frozen ORB GNN stack: gnn_stacks[{item}]")
                elif item == 'atom_emb':
                    # Freeze atom embedding
                    if hasattr(self.graph_encoder.orb_model, 'atom_emb'):
                        for param in self.graph_encoder.orb_model.atom_emb.parameters():
                            param.requires_grad = False
                        print("Frozen ORB atom_emb layer")
                elif item == 'conditioner':
                    # Freeze conditioner if exists
                    if hasattr(self.graph_encoder.orb_model, 'conditioner'):
                        for param in self.graph_encoder.orb_model.conditioner.parameters():
                            param.requires_grad = False
                        print("Frozen ORB conditioner layer")

        # Freeze embedding layer if specified (for CGCNN/PaiNN)
        if 'embedding' in freeze_layers:
            if hasattr(self.graph_encoder, 'embedding'):
                for param in self.graph_encoder.embedding.parameters():
                    param.requires_grad = False
                print("Frozen embedding layer")

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
            # Training: MSE for stable gradients
            # Validation/Test: MAE for interpretable metric
            if mode == 'train':
                loss = F.mse_loss(y_pred.squeeze(), label.squeeze())
            else:  # val or test
                loss = F.l1_loss(y_pred.squeeze(), label.squeeze())
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
    

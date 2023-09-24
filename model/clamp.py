import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torch.nn.functional as F


class CLaMP(pl.LightningModule):
    def __init__(\
            self,
            graph_encoder: torch.nn.Module,
            text_encoder: torch.nn.Module,
            graph_encoder_dim: int,
            text_encoder_dim: int,
            clamp_dim: int,
            lr: float = 1e-4
        ):        
        super().__init__()
        self.save_hyperparameters(ignore=['graph_encoder', 'text_encoder'])
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
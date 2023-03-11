import torch
import torch.nn as nn
import numpy as np

class SCE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def __instance_bce_with_logits(logits: torch.Tensor, labels: torch.Tensor, keep_shape: bool = False)->torch.Tensor:
        assert logits.dim() == 2
        if not keep_shape:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            loss = loss.sum(1) / labels.size(1)
        loss *= labels.size(1)
        return loss
    
    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        Embedding = torch.from_numpy(np.load('data/cache/bert_init_answer_768d.npy')) # num_cls dim
        Embedding_sim = Embedding.mm(Embedding.T) # num_cls num_cls
        E1_norm = torch.sqrt((Embedding_sim**2).sum(1).view(-1, 1))
        E2_norm = torch.sqrt(((Embedding_sim)**2).sum(1).view(-1, 1))
        Embedding_sim = Embedding_sim / (E1_norm.mm(E2_norm.T)+1e-8)

        diagoal_mask = torch.eye(Embedding_sim.size(0), Embedding_sim.size(0),device=Embedding_sim.device).bool()
        Embedding_sim_masked = Embedding_sim.masked_fill(diagoal_mask, -2.)
        mapping_dic = torch.argmax(Embedding_sim_masked, dim=1, keepdim=True) # num_cls 1 

        semantic_a = torch.zeros_like(label)

        for i in range(label.size(0)):
            gt_index = torch.nonzero(label[i]).view(-1) # [len_gt]
            for g_index in gt_index:
                semantic_a[i][mapping_dic[g_index]] = label[i][g_index]
                
        return self.__instance_bce_with_logits(prediction, semantic_a)
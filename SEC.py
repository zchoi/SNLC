import torch
import torch.nn as nn
from typing import Optional

class SEC(nn.Module):
    '''compute contrastive loss
    '''
    def __init__(self, margin: int = 0, max_violation: bool = False, direction: str ='bi', topk: int =1) -> None:
        '''Args:
        direction: i2t for negative sentence, t2i for negative image, bi for both
        '''
        super(SEC, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.direction = direction
        self.topk = topk

    def forward(self, scores: torch.Tensor, margin: Optional[int] = None, label: torch.Tensor = None) -> torch.Tensor:
        '''
        Args:
        scores: image-label score matrix, (num_imgs, num_labels)
        label: imgs' ground truth index, [0,1,8...]
            the same row of im and s are positive pairs, different rows are negative pairs
            there may have multiple correct captions in each row
        '''
        batch_size = scores.size(0)
        if margin is None:
            margin = self.margin
        loss = 0.
        scores = torch.exp(scores)
        
        for i in range(label.size(0)):
            pos = 0.0
            gt_index = torch.nonzero(label[i]).view(1,-1) # [len_gt]
            mask_pos_cap = scores[i:i+1].scatter(1, torch.cuda.LongTensor(gt_index), float('-inf')) # mask gt (1, num_cap)
            semantic_pos = torch.max(mask_pos_cap)
            mask_pos_cap = mask_pos_cap.scatter(1,torch.cuda.LongTensor(torch.argmax(mask_pos_cap).view(1,-1)), float('-inf'))

            for j in gt_index[0]:
                pos += scores[i][j] 
            pos += semantic_pos
            loss += -torch.log(pos / torch.sum(scores[i]) + 0.05)

        return loss / batch_size

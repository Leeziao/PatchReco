from config import *

from typing import Any, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, trainer_utils, BertModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

class PatchModel(nn.Module):
    def __init__(self,
                 msgModelCheckpoint: str,
                 codeModelCheckpoint: str,
                 trainBase: bool=False,
                 dropout=0.1) -> None:
        super().__init__()
        trainedMsgModel = AutoModelForSequenceClassification.from_pretrained(msgModelCheckpoint)
        trainedCodeModel = AutoModelForSequenceClassification.from_pretrained(codeModelCheckpoint)

        self.msgModel = BertModel(trainedMsgModel.config, add_pooling_layer=True)
        self.msgModel.load_state_dict(trainedMsgModel.bert.state_dict())

        self.codeModel = RobertaModel(trainedCodeModel.config, add_pooling_layer=False)
        self.codeModel.load_state_dict(trainedCodeModel.roberta.state_dict())

        self.trainBase = trainBase
        self.numLabels = 2
        self.concatDim = self.msgModel.pooler.dense.out_features + \
            2 * self.codeModel.encoder.layer[-1].output.dense.out_features
        self.dropout = nn.Dropout(dropout)

        self.dense1 = nn.Linear(self.concatDim, 768)
        self.dense2 = nn.Linear(768, 768)
        self.outProj = nn.Linear(768, self.numLabels)

        for param in self.msgModel.parameters():
            param.requires_grad_(trainBase)
        for param in self.codeModel.parameters():
            param.requires_grad_(trainBase)
 
    def forwardBaseModel(self,
        msg_input_ids,
        msg_token_type_ids,
        msg_attention_mask,
        msg_special_tokens_mask,
        code_input_ids,
        code_token_type_ids,
        code_attention_mask,
        code_special_tokens_mask,
    ) -> torch.Tensor:
        batchSize = msg_input_ids.shape[0]
        msgOutputs = self.msgModel(
            msg_input_ids,
            attention_mask=msg_attention_mask,
            token_type_ids=msg_token_type_ids,
        )
        msgOutputs = msgOutputs[1] # pooler_output

        code_input_ids = code_input_ids.reshape(-1, code_input_ids.shape[-1])
        code_token_type_ids = code_token_type_ids.reshape(-1, code_token_type_ids.shape[-1])
        code_attention_mask = code_attention_mask.reshape(-1, code_attention_mask.shape[-1])
        codeOutputs = self.codeModel(
            code_input_ids,
            attention_mask=code_attention_mask,
            token_type_ids=code_token_type_ids,
        )
        codeOutputs = codeOutputs[0] # last_hidden_state
        codeOutputs = codeOutputs[:, 0, :]
        codeOutputs = codeOutputs.reshape(batchSize, -1, codeOutputs.shape[-1])

        x = torch.concat([msgOutputs,
                          codeOutputs.max(dim=1).values,
                          codeOutputs.mean(dim=1)], dim=-1)
        return x

    def forward(
        self,
        msg_input_ids,
        msg_token_type_ids,
        msg_attention_mask,
        msg_special_tokens_mask,
        code_input_ids,
        code_token_type_ids,
        code_attention_mask,
        code_special_tokens_mask,
        labels,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if not self.trainBase:
            with torch.no_grad():
                x = self.forwardBaseModel(
                    msg_input_ids,
                    msg_token_type_ids,
                    msg_attention_mask,
                    msg_special_tokens_mask,
                    code_input_ids,
                    code_token_type_ids,
                    code_attention_mask,
                    code_special_tokens_mask,
                )
        else:
            x = self.forwardBaseModel(
                msg_input_ids,
                msg_token_type_ids,
                msg_attention_mask,
                msg_special_tokens_mask,
                code_input_ids,
                code_token_type_ids,
                code_attention_mask,
                code_special_tokens_mask,
            )

        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.outProj(x)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(x.view(-1, self.numLabels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=x
        )

if __name__ == '__main__':
    m = PatchModel(
        trainer_utils.get_last_checkpoint(msgCHECKPOINTDIR),
        trainer_utils.get_last_checkpoint(codeCHECKPOINTDIR)
    )

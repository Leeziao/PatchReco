import dataset
from config import *
import utils

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path

import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, RobertaForSequenceClassification, trainer_utils

def evaluteSeperateModel(modelType: str):
    assert(modelType in ['msg', 'code'])

    PD = dataset.getHGDataset(modelType)
    evalData = PD['eval']

    if modelType == 'code':
        checkPointDir = codeCHECKPOINTDIR
        batchSize = utils.hpJ['codeBatchSize']
    elif modelType == 'msg':
        checkPointDir = msgCHECKPOINTDIR
        batchSize = utils.hpJ['msgBatchSize']

    m = AutoModelForSequenceClassification.from_pretrained(
        trainer_utils.get_last_checkpoint(checkPointDir)
    )

    trainingArgs = TrainingArguments(
        output_dir=checkPointDir,
        evaluation_strategy='epoch',
        eval_steps=1,

        report_to='tensorboard',
        logging_strategy='steps',
        logging_steps=utils.hpJ['logging_steps'],

        save_strategy='steps',
        save_steps=utils.hpJ['save_steps'],
        save_total_limit=10,

        per_device_train_batch_size=batchSize,
        per_device_eval_batch_size=batchSize,
        num_train_epochs=utils.hpJ['epoch'],

        ignore_data_skip=utils.hpJ['ignore_data_skip'],
        dataloader_drop_last=True,
    )

    metric = evaluate.load('accuracy')
    def compute_metric(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        return metric.compute(predictions=pred, references=labels)
    
    trainer = Trainer(
        model=m,
        args=trainingArgs,
        eval_dataset=evalData,
        compute_metrics=compute_metric
    )

    trainer.evaluate()

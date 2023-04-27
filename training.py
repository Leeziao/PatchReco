import models
import dataset
from config import *
import utils

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path

import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, RobertaForSequenceClassification

def isResumeTraining(p: Path):
    fs = list(p.iterdir())
    fs = [f for f in fs if f.is_dir() and 'checkpoint' in f.name]
    return len(fs) != 0

def trainModel(modelType: str):
    assert(modelType in ['msg', 'code'])

    PD = dataset.PatchDataset()
    trainData = PD.getSplit('train', modelType)
    # testData = PD.getSplit('test', modelType)
    evalData = PD.getSplit('eval', modelType)

    if modelType == 'code':
        m = AutoModelForSequenceClassification.from_pretrained("neulab/codebert-cpp", num_labels=2)
        checkPointDir = codeCHECKPOINTDIR
        batchSize = utils.hpJ['codeBatchSize']
    elif modelType == 'msg':
        m = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        checkPointDir = msgCHECKPOINTDIR
        batchSize = utils.hpJ['msgBatchSize']

    trainingArgs = TrainingArguments(
        output_dir=checkPointDir,
        evaluation_strategy='epoch',
        eval_steps=1,

        report_to='azure_ml',
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
        train_dataset=trainData,
        eval_dataset=evalData,
        compute_metrics=compute_metric
    )

    if isResumeTraining(checkPointDir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
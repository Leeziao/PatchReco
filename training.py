from typing import Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_utils import EvalPrediction, seed_worker
from transformers.training_args import TrainingArguments

import dataset
from config import *
import models
import utils

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path

import evaluate
import datasets
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, trainer_utils

def isResumeTraining(p: Path):
    fs = list(p.iterdir())
    fs = [f for f in fs if f.is_dir() and 'checkpoint' in f.name]
    return len(fs) != 0

def trainSeperateModel(modelType: str):
    assert(modelType in ['msg', 'code'])

    PD = dataset.getHGDataset(modelType)
    trainData = PD['train']
    testData = PD['test']
    evalData = PD['eval']

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
        train_dataset=trainData,
        eval_dataset=evalData,
        compute_metrics=compute_metric
    )

    if isResumeTraining(checkPointDir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


class TrainerForAll(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

def trainAllModel():
    PD = dataset.getHGDataset('all')
    trainData = PD['train']
    testData = PD['test']
    evalData = PD['eval']

    m = models.PatchModel(
        trainer_utils.get_last_checkpoint(msgCHECKPOINTDIR),
        trainer_utils.get_last_checkpoint(codeCHECKPOINTDIR),
        trainBase=utils.hpJ['trainBase'],
    )

    batchSize = utils.hpJ['allBatchSize']
    checkPointDir = allCHECKPOINTDIR
    trainingArgs = TrainingArguments(
        output_dir=checkPointDir,
        evaluation_strategy='epoch',
        eval_steps=1,

        report_to='tensorboard',
        logging_strategy='steps',
        logging_steps=utils.hpJ['logging_steps'],

        save_strategy='steps',
        save_steps=utils.hpJ['save_steps'],
        save_total_limit=3,

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
    
    trainer = TrainerForAll(
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

import dataset
from config import *
import utils
import models

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path

import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, RobertaForSequenceClassification, trainer_utils
import datasets
import json

def evaluteModel(modelType: str):
    assert(modelType in ['msg', 'code', 'hunk', 'all', 'allHunk'])

    resultJ = json.loads(resultFilePath.read_text()) if resultFilePath.exists() else dict()
    if modelType in resultJ:
        print(f'{modelType} already evaluated, skipped')
        return

    tgtSplit = 'test'
    PD = dataset.getHGDataset(modelType)
    fileList = json.loads(shuffleFileListFilePath.read_text())[tgtSplit]
    evalData = PD[tgtSplit]

    if modelType == 'code':
        checkPointDir = codeCHECKPOINTDIR
        m = AutoModelForSequenceClassification.from_pretrained(
            trainer_utils.get_last_checkpoint(checkPointDir)
        )
    elif modelType == 'msg':
        checkPointDir = msgCHECKPOINTDIR
        m = AutoModelForSequenceClassification.from_pretrained(
            trainer_utils.get_last_checkpoint(checkPointDir)
        )
    elif modelType == 'hunk':
        checkPointDir = hunkCHECKPOINTDIR
        modelPath = Path(trainer_utils.get_last_checkpoint(checkPointDir)) / 'pytorch_model.bin'
        if torch.cuda.is_available():
            st_dict = torch.load(modelPath)
        else:
            st_dict = torch.load(modelPath, map_location='cpu')
        m = models.HunkClassifierModel('neulab/codebert-cpp')
        m.load_state_dict(st_dict)
    elif modelType == 'all':
        checkPointDir = allCHECKPOINTDIR
        modelPath = Path(trainer_utils.get_last_checkpoint(checkPointDir)) / 'pytorch_model.bin'
        if torch.cuda.is_available():
            st_dict = torch.load(modelPath)
        else:
            st_dict = torch.load(modelPath, map_location='cpu')
        m = models.PatchModel(
            trainer_utils.get_last_checkpoint(msgCHECKPOINTDIR),
            trainer_utils.get_last_checkpoint(codeCHECKPOINTDIR)
        )
        m.load_state_dict(st_dict)
    elif modelType == 'allHunk':
        checkPointDir = allHunkCHECKPOINTDIR
        modelPath = Path(trainer_utils.get_last_checkpoint(checkPointDir)) / 'pytorch_model.bin'
        if torch.cuda.is_available():
            st_dict = torch.load(modelPath)
        else:
            st_dict = torch.load(modelPath, map_location='cpu')
        m = models.HunkPatchModel(
            trainer_utils.get_last_checkpoint(msgCHECKPOINTDIR),
            trainer_utils.get_last_checkpoint(codeCHECKPOINTDIR)
        )
        m.load_state_dict(st_dict)

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

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=utils.hpJ['epoch'],

        ignore_data_skip=utils.hpJ['ignore_data_skip'],
        dataloader_drop_last=False,
    )

    metric = evaluate.combine(['accuracy', 'f1', 'precision', 'recall'])
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

    # FOR TESTING ONLY
    # evalData = evalData.to_dict()
    # for k in evalData.keys():
    #     evalData[k] = evalData[k][:100]
    # evalData = datasets.Dataset.from_dict(evalData)
    # fileList = fileList[:100]

    # ------------- MODEL INFERENCE ------------------ #
    result = trainer.predict(evalData)
    # ------------- MODEL INFERENCE ------------------ #

    preds, gts = result.predictions.argmax(axis=-1).tolist(), result.label_ids.tolist()
    assert(len(preds) == len(gts) == len(fileList))
    resultMetrics = result.metrics

    j = {}
    metrics = {
        'accuracy': resultMetrics['test_accuracy'],
        'precision': resultMetrics['test_precision'],
        'recall': resultMetrics['test_recall'],
        'f1': resultMetrics['test_f1'],
    }
    j['metrics'] = metrics

    stats = dict()
    for i, filePath in enumerate(fileList):
        stats[filePath] = {
            'pred': preds[i],
            'gt': gts[i],
        }
    j['stats'] = stats
    resultJ[modelType] = j

    resultFilePath.write_text(json.dumps(resultJ, indent=2))

from config import *
import models
import dataset

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, RobertaForSequenceClassification, trainer_utils
import datasets
import json
import torch
import tempfile
from pathlib import Path
import os

def predictItem(modelType: str, patch: str) -> bool:
    assert(modelType in ['msg', 'code', 'hunk', 'all', 'allHunk'])

    predictPatchFilePath.write_text(patch)
    predictFileListFilePath.write_text(f'{json.dumps([str(predictPatchFilePath)], indent=2)}')

    if predictRawDataFilePath.exists():
        os.remove(predictRawDataFilePath)
    dataset.createRawDataset(fileListInput=str(predictFileListFilePath), 
                             tgtFileListPath=predictFileListFilePath, 
                             tgtRawDataPath=predictRawDataFilePath,
                             force=True)
    
    if predictProcessedDataFilePath.exists():
        os.remove(predictProcessedDataFilePath)
    dataset.processRawDataset(tgtFilePath=predictProcessedDataFilePath,
                              srcFilePath=predictRawDataFilePath)
    
    j = json.loads(predictProcessedDataFilePath.read_text())
    j = {'train': j}
    predictProcessedDataFilePath.write_text(json.dumps(j, indent=2))

    PD = dataset.PatchDataset(predictProcessedDataFilePath)
    split = PD.getSplit('train', modelType)

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
    
    data = split.to_dict()
    data.pop('special_tokens_mask')
    data = {k: torch.tensor(v) for k, v in data.items()}

    m.eval()
    with torch.no_grad():
        result = m(**data)
        result = result.logits[0].argmax().item()
    
    return bool(result)

if __name__ == '__main__':
    modelType = 'msg'
    patch = '''
commit b03da72d7c11ae1b207cae3e1318ce31fc6f4c19
Author: Jakub Filak <jfilak@redhat.com>
Date:   Wed Jun 5 13:51:53 2013 +0200

    ureport: add conversion from abrt vmcore type to ureport KERNELOOPS type
    
    Closes #163
    
    Signed-off-by: Jakub Filak <jfilak@redhat.com>
    Signed-off-by: Martin Milata <mmilata@redhat.com>

diff --git a/src/lib/json.c b/src/lib/json.c
index bf1e0494..d5c6f2d6 100644
--- a/src/lib/json.c
+++ b/src/lib/json.c
@@ -139,7 +139,7 @@ static bool ureport_add_type(struct json_object *ur, problem_data_t *pd)
         ureport_add_str(ur, "type", "USERSPACE");
     else if (strcmp(pd_item, "Python") == 0)
         ureport_add_str(ur, "type", "PYTHON");
-    else if (strcmp(pd_item, "Kerneloops") == 0)
+    else if (strcmp(pd_item, "Kerneloops") == 0 || strcmp(pd_item, "vmcore") == 0)
         ureport_add_str(ur, "type", "KERNELOOPS");
     else
     {
    '''
    result = predictItem(modelType, patch)
    print(result)

import os
import re
import numpy as np
import random

from transformers import AutoTokenizer
import datasets
from datasets import Dataset

from pathlib import Path
from typing import List
import json
import torch.utils.data

from myCmd import CMD
from config import *
import utils

@utils.statusNotifier
def preprocessZipData():
    for filePath in RAWDATADIR.glob('**/*'):
        if filePath.name.endswith('.zip'):
            CMD(f'unzip {filePath}', path=filePath.parent)()
            CMD(f'rm {filePath}', path=filePath.parent)()

@utils.statusNotifier
def createRawDataset():
    '''
        Read data from the files.
        :return: data - a set of commit message, diff code, and labels.
        [[['', ...], [['', ...], ['', ...], ...], 0/1], ...]
        [
            [
                ['msg Line1', 'msg Line2', ...],
                [['hunk1 line1', 'hunk1 line2', ...], ['hunk2 line1', 'hunk2 line2', ...]],
                0/1
            ],
            [
                ...
            ]
        ]
    '''
    if rawDataFilePath.exists(): return

    def ReadCommitMsg(filename):
        '''
        Read commit message from a file.
        :param filename: file name (string).
        :return: commitMsg - commit message.
        '''

        fp = open(filename, encoding='utf-8', errors='ignore')  # get file point.
        lines = fp.readlines()  # read all lines.
        #numLines = len(lines)   # get the line number.
        #print(lines)

        # initialize commit message.
        commitMsg = []
        # get the wide range of commit message.
        for line in lines:
            if line.startswith('diff --git'):
                break
            else:
                commitMsg.append(line)

        # process the head of commit message.
        while (1):
            headMsg = commitMsg[0]
            if (headMsg.startswith('From') or headMsg.startswith('Date:') or headMsg.startswith('Subject:')
                    or headMsg.startswith('commit') or headMsg.startswith('Author:')):
                commitMsg.pop(0)
            else:
                break

        # process the tail of commit message.
        dashLines = [i for i in range(len(commitMsg))
                     if commitMsg[i].startswith('---')]  # finds all lines start with ---.
        if (len(dashLines)):
            lnum = dashLines[-1]  # last line number of ---
            marks = [1 if (' file changed, ' in commitMsg[i] or ' files changed, ' in commitMsg[i]) else 0
                     for i in range(lnum, len(commitMsg))]
            if (sum(marks)):
                for i in reversed(range(lnum, len(commitMsg))):
                    commitMsg.pop(i)

        return ''.join(commitMsg)

    def ReadDiffLines(filename):
        '''
        Read diff code from a file.
        :param filename:  file name (string).
        :return: diffLines - diff code.
        '''

        fp = open(filename, encoding='utf-8', errors='ignore')  # get file point.
        lines = fp.readlines()  # read all lines.
        numLines = len(lines)  # get the line number.

        atLines = [i for i in range(numLines) if lines[i].startswith('@@ ')]  # find all lines start with @@.
        atLines.append(numLines)

        diffLines = []
        for nh in range(len(atLines) - 1):  # find all hunks.
            # print(atLines[nh], atLines[nh + 1])
            hunk = []
            for nl in range(atLines[nh] + 1, atLines[nh + 1]):
                # print(lines[nl], end='')
                if lines[nl].startswith('diff --git '):
                    break
                else:
                    hunk.append(lines[nl])
            diffLines.append(hunk)

        # process the last hunk.
        lastHunk = diffLines[-1]
        numLastHunk = len(lastHunk)
        dashLines = [i for i in range(numLastHunk) if lastHunk[i].startswith('--')]
        if (len(dashLines)):
            lnum = dashLines[-1]
            for i in reversed(range(lnum, numLastHunk)):
                lastHunk.pop(i)

        return [''.join(hunk) for hunk in diffLines]

    # initialize data.
    data = []
    # read security patch data.
    for filePath in sRAWDATADIR.glob('**/*'):
        if not filePath.is_file(): continue

        commitMsg = ReadCommitMsg(filePath)
        diffLines = ReadDiffLines(filePath)
        data.append([commitMsg, diffLines, 1])

    # read positive data.
    for filePath in pRAWDATADIR.glob('**/*'):
        if not filePath.is_file(): continue

        commitMsg = ReadCommitMsg(filePath)
        diffLines = ReadDiffLines(filePath)
        data.append([commitMsg, diffLines, 1])

    # read negative data.
    for filePath in nRAWDATADIR.glob('**/*'):
        if not filePath.is_file(): continue

        commitMsg = ReadCommitMsg(filePath)
        diffLines = ReadDiffLines(filePath)
        data.append([commitMsg, diffLines, 0])

    rawDataFilePath.write_text(json.dumps(data, indent=2))
    
    return data

@utils.statusNotifier
def processRawDataset():
    if processedDataFilePath.exists(): return

    rawJ = json.loads(rawDataFilePath.read_text())

    msgs = [j[0].strip() for j in rawJ]
    codes = [j[1] for j in rawJ]
    labels = [j[2] for j in rawJ]

    def isMsgLineKept(line: str):
        line = line.strip().lower()
        if len(line) == 0:
            return False
        
        r = re.match(r'^([a-zA-Z-]+):', line)
        if not r: return True

        prefix = r.group(1)
        if prefix.endswith('by') or prefix == 'cc': return False

        return True
    
    def rewriteMsg(msg: str):
        lines = msg.split('\n')
        msg = '\n'.join([line for line in lines if isMsgLineKept(line)])
        return msg
    
    def rewriteCode(code: List[str]):
        def rewriteHunk(hunk: str):
            return '\n'.join([line for line in hunk.split('\n') if len(line.strip()) != 0])
        code = [rewriteHunk(hunk) for hunk in code]
        return code

    msgs = [rewriteMsg(msg) for msg in msgs]
    codes = [rewriteCode(code) for code in codes]

    processedJ = list(zip(msgs, codes, labels))
    processedDataFilePath.write_text(json.dumps(processedJ, indent=2))

@utils.statusNotifier
def splitProcessedData(testSplit=0.1, evalSplit=0.1):
    if splitDataFilePath.exists():
        return

    d = json.loads(processedDataFilePath.read_text())
    random.shuffle(d)
    dSize = len(d)
    testSize = int(dSize * testSplit)
    evalSize = int(dSize * evalSplit)
    trainSize = dSize - testSize - evalSize

    trainD = d[:trainSize]
    evalD = d[trainSize: trainSize+evalSize]
    testD = d[trainSize+evalSize: trainSize+evalSize+testSize]

    j = {
        'train': trainD,
        'eval': evalD,
        'test': testD,
    }

    splitDataFilePath.write_text(json.dumps(j, indent=2))

class PatchDataset:
    def __init__(self):
        self.dataset = json.loads(splitDataFilePath.read_text())
        self.msgTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.codeTokenizer = AutoTokenizer.from_pretrained("neulab/codebert-cpp")

    def getSplit(self, splitType: str='train', dataType: str='msg'):
        assert(splitType in ['train', 'test', 'eval'])
        assert(dataType in ['msg', 'code'])

        if dataType == 'msg':
            data = self.dataset[splitType]
            codes = self.msgTokenizer([d[0] for d in data],
                                truncation=True,
                                padding='max_length',
                                add_special_tokens=True,
                                max_length=utils.hpJ['msgTokenLength'],

                                return_tensors='pt',
                                return_attention_mask=True,
                                return_special_tokens_mask=True,
                                return_token_type_ids=True,
                                return_length=True)

            labels = torch.LongTensor([d[2] for d in data])

            result = {k: codes[k] for k in codes.keys()}
            result['labels'] = labels
            return datasets.Dataset.from_dict(result).shuffle()

        elif dataType == 'code':
            data = self.dataset[splitType]
            codes_unpack, labels_unpack = [], []
            for d in data:
                codes_unpack.extend(d[1])
                labels_unpack.extend([d[2] for _ in range(len(d[1]))])
            codes = self.codeTokenizer(codes_unpack,
                                truncation=True,
                                padding='max_length',
                                add_special_tokens=True,
                                max_length=utils.hpJ['codeTokenLength'],

                                return_tensors='pt',
                                return_attention_mask=True,
                                return_special_tokens_mask=True,
                                return_token_type_ids=True,
                                return_length=True)

            labels = torch.LongTensor(labels_unpack)

            result = {k: codes[k] for k in codes.keys()}
            result['labels'] = labels
            return datasets.Dataset.from_dict(result).shuffle()

def buildDataLoader():
    raise NotImplemented()
    # def collate_fn():
    #     msgTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #     def _collate_fn(data):
    #         msgs = msgTokenizer([d[0] for d in data],
    #                             truncation=True,
    #                             padding='max_length',
    #                             add_special_tokens=True,
    #                             max_length=400,

    #                             return_tensors='pt',
    #                             return_attention_mask=True,
    #                             return_special_tokens_mask=True,
    #                             return_token_type_ids=True,
    #                             return_length=True)
    #         labels = torch.LongTensor([d[2] for d in data])

    #         return msgs['input_ids'], msgs['attention_mask'], labels
    #     return _collate_fn

    # dataset = PatchDataset('train')

    # loader = torch.utils.data.DataLoader(dataset,
    #                                      batch_size=4,
    #                                      collate_fn=collate_fn(),
    #                                      shuffle=True)
    
    # return loader

if __name__ == '__main__':
    preprocessZipData()
    createRawDataset()
    processRawDataset()
    splitProcessedData()

    patch = PatchDataset()
    # patch.getSplit('train', 'code')

    # buildDataLoader()

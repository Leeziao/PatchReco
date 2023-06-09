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
            pathWithoutExt = filePath.parent / filePath.stem
            if pathWithoutExt.exists(): continue

            CMD(f'unzip {filePath}', path=filePath.parent)()
            # CMD(f'rm {filePath}', path=filePath.parent)()

@utils.statusNotifier
def createRawDataset(fileListInput: str='', tgtFileListPath=fileListFilePath, tgtRawDataPath=rawDataFilePath, force=False):
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
    if tgtRawDataPath.exists() and tgtFileListPath.exists() and not force: return

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
    data, fileList = [], []
    if fileListInput == '':
        # read security patch data.
        for filePath in sRAWDATADIR.glob('**/*'):
            if not filePath.is_file() or filePath.suffix == '.zip': continue

            fileList.append(str(filePath.relative_to(PROJECT_ROOT)))
            commitMsg = ReadCommitMsg(filePath)
            diffLines = ReadDiffLines(filePath)
            data.append([commitMsg, diffLines, 1])

        # read positive data.
        for filePath in pRAWDATADIR.glob('**/*'):
            if not filePath.is_file() or filePath.suffix == '.zip': continue

            fileList.append(str(filePath.relative_to(PROJECT_ROOT)))
            commitMsg = ReadCommitMsg(filePath)
            diffLines = ReadDiffLines(filePath)
            data.append([commitMsg, diffLines, 1])

        # read negative data.
        for filePath in nRAWDATADIR.glob('**/*'):
            if not filePath.is_file() or filePath.suffix == '.zip': continue

            fileList.append(str(filePath.relative_to(PROJECT_ROOT)))
            commitMsg = ReadCommitMsg(filePath)
            diffLines = ReadDiffLines(filePath)
            data.append([commitMsg, diffLines, 0])
    else:
        fileList = json.loads(Path(fileListInput).read_text())

        for filePath in fileList:
            s = str(filePath)
            if 'data/raw/negatives' in s:
                tgtLabel = 0
            elif 'data/raw/positives' in s or 'data/raw/security_patch' in s:
                tgtLabel = 1
            else:
                tgtLabel = 0
                print("Unknown Label Rule")
                # raise ValueError("Unknown Label Rule")
            commitMsg = ReadCommitMsg(filePath)
            diffLines = ReadDiffLines(filePath)
            data.append([commitMsg, diffLines, tgtLabel])

    tgtRawDataPath.write_text(json.dumps(data, indent=2))
    tgtFileListPath.write_text(json.dumps(fileList, indent=2))
    
    return data

@utils.statusNotifier
def processRawDataset(tgtFilePath=processedDataFilePath, srcFilePath=rawDataFilePath):
    if tgtFilePath.exists(): return

    rawJ = json.loads(srcFilePath.read_text())

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
        keepHunkNum = 5
        if len(code) > keepHunkNum: code = code[:keepHunkNum]
        else: code = code + ['' for _ in range(keepHunkNum - len(code))]
        return code
    
    def rewriteHunk(code: List[str]):
        def _rewriteHunk(hunk: str):
            lines = hunk.split('\n')
            postHunk, preHunk = [], []
            for line in lines:
                if line.startswith('+'):
                    line = ' ' + line[1:]
                    postHunk.append(line)
                elif line.startswith('-'):
                    line= ' ' + line[1:]
                    preHunk.append(line)
                else:
                    postHunk.append(line)
                    preHunk.append(line)
            return ['\n'.join(preHunk), '\n'.join(postHunk)]
        return [_rewriteHunk(c) for c in code]

    msgs = [rewriteMsg(msg) for msg in msgs]
    codes = [rewriteCode(code) for code in codes]
    hunks = [rewriteHunk(code) for code in codes]

    processedJ = list(zip(msgs, codes, labels, hunks))
    tgtFilePath.write_text(json.dumps(processedJ, indent=2))

@utils.statusNotifier
def splitProcessedData(testSplit=0.1, evalSplit=0.1):
    if splitDataFilePath.exists() and shuffleFileListFilePath.exists():
        return

    d = json.loads(processedDataFilePath.read_text())
    fileList = json.loads(fileListFilePath.read_text())
    dSize = len(d)

    indices = list(range(dSize))
    random.Random(42).shuffle(indices)

    testSize = int(dSize * testSplit)
    evalSize = int(dSize * evalSplit)
    trainSize = dSize - testSize - evalSize

    trainIndices = set(indices[:trainSize])
    evalIndices = set(indices[trainSize: trainSize+evalSize])
    testIndices = set(indices[trainSize+evalSize: trainSize+evalSize+testSize])

    trainD = [dd for i, dd in enumerate(d) if i in trainIndices]
    evalD = [dd for i, dd in enumerate(d) if i in evalIndices]
    testD = [dd for i, dd in enumerate(d) if i in testIndices]

    trainFileList = [f for i, f in enumerate(fileList) if i in trainIndices]
    evalFileList = [f for i, f in enumerate(fileList) if i in evalIndices]
    testFileList = [f for i, f in enumerate(fileList) if i in testIndices]


    j = {'train': trainD, 'eval': evalD, 'test': testD}
    jF = {'train': trainFileList, 'eval': evalFileList, 'test': testFileList}

    splitDataFilePath.write_text(json.dumps(j, indent=2))
    shuffleFileListFilePath.write_text(json.dumps(jF, indent=2))

class PatchDataset:
    def __init__(self, datasetPath=splitDataFilePath):
        self.dataset = json.loads(datasetPath.read_text())
        self.msgTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.codeTokenizer = AutoTokenizer.from_pretrained("neulab/codebert-cpp")
    
    def getMsgSplit(self, splitType: str):
        data = self.dataset[splitType]
        codes = self.msgTokenizer([d[0] for d in data],
                            truncation=True,
                            padding='max_length',
                            add_special_tokens=True,
                            max_length=utils.hpJ['msgTokenLength'],

                            return_attention_mask=True,
                            return_special_tokens_mask=True,
                            return_token_type_ids=True)

        labels = torch.LongTensor([d[2] for d in data])

        result = {k: codes[k] for k in codes.keys()}
        result['labels'] = labels
        return datasets.Dataset.from_dict(result)
    
    def getCodeSplit(self, splitType: str, flatten: bool=True):
        data = self.dataset[splitType]
        codes_unpack, labels_unpack, hunkNum = [], [], []
        for d in data:
            codes_unpack.extend(d[1])
            labels_unpack.extend([d[2] for _ in range(len(d[1]))])
            hunkNum.append(len(d[1]))

        codes = self.codeTokenizer(codes_unpack,
                            truncation=True,
                            padding='max_length',
                            add_special_tokens=True,
                            max_length=utils.hpJ['codeTokenLength'],

                            return_attention_mask=True,
                            return_special_tokens_mask=True,
                            return_token_type_ids=True)

        labels = torch.LongTensor(labels_unpack)

        if flatten:
            result = {k: codes[k] for k in codes.keys()}
            result['labels'] = labels
            return datasets.Dataset.from_dict(result)
        else:
            index, keys = 0, codes.keys()
            result = {k: [] for k in keys}
            for hn in hunkNum:
                for k in keys:
                    result[k].append(codes[k][index: index + hn])
                index += hn
            result['labels'] = torch.LongTensor([d[2] for d in data])
            return datasets.Dataset.from_dict(result)

    def getHunkSplit(self, splitType: str):
        data = self.dataset[splitType]
        hunks, hunks_f = [d[3] for d in data], []
        hunkShape = [len(hunks), len(hunks[0]), len(hunks[0][0])]
        hunks_f = [item for h in hunks for hh in h for item in hh]

        labels = torch.LongTensor([d[2] for d in data])

        hunks = self.codeTokenizer(hunks_f,
                            truncation=True,
                            padding='max_length',
                            add_special_tokens=True,
                            max_length=utils.hpJ['codeTokenLength'],
                            return_tensors='np',

                            return_attention_mask=True,
                            return_special_tokens_mask=True,
                            return_token_type_ids=True)
        result = dict()
        for k in hunks.keys():
            result[k] = hunks[k].reshape(hunkShape[0], hunkShape[1] * 2, -1)
        result['labels'] = labels

        return datasets.Dataset.from_dict(result)

    @utils.statusNotifier
    def getSplit(self, splitType: str='train', dataType: str='msg'):
        assert(splitType in ['train', 'test', 'eval'])
        assert(dataType in ['msg', 'code', 'all', 'hunk', 'allHunk'])

        if dataType == 'msg':
            return self.getMsgSplit(splitType)
        elif dataType == 'code':
            return self.getCodeSplit(splitType)
        elif dataType == 'hunk':
            return self.getHunkSplit(splitType)
        elif dataType == 'allHunk':
            msgSplit = self.getMsgSplit(splitType).to_dict()
            hunkSplit = self.getHunkSplit(splitType).to_dict()

            msgLabels = msgSplit['labels']
            hunkLabels = hunkSplit['labels']
            equalLabelNum = sum([msgLabels[i] == hunkLabels[i] for i in range(len(msgLabels))])
            assert(equalLabelNum == len(msgLabels) == len(hunkLabels))

            result = dict()
            for k in msgSplit.keys():
                if k == 'labels': continue
                result[f'msg_{k}'] = msgSplit[k]
            for k in hunkSplit.keys():
                if k == 'labels': continue
                result[f'hunk_{k}'] = hunkSplit[k]
            result['labels'] = msgLabels

            return datasets.Dataset.from_dict(result)

        elif dataType == 'all':
            msgSplit = self.getMsgSplit(splitType).to_dict()
            codeSplit = self.getCodeSplit(splitType, flatten=False).to_dict()

            msgLabels = msgSplit['labels']
            codeLabels = codeSplit['labels']
            equalLabelNum = sum([msgLabels[i] == codeLabels[i] for i in range(len(msgLabels))])
            assert(equalLabelNum == len(msgLabels) == len(codeLabels))

            result = dict()
            for k in msgSplit.keys():
                if k == 'labels': continue
                result[f'msg_{k}'] = msgSplit[k]
            for k in codeSplit.keys():
                if k == 'labels': continue
                result[f'code_{k}'] = codeSplit[k]
            result['labels'] = msgLabels

            return datasets.Dataset.from_dict(result)

@utils.statusNotifier
def createHGDataset(dataType: str='msg'):
    assert(dataType in ['msg', 'code', 'all', 'hunk', 'allHunk'])
    targetDirPath = hgDATADIR / dataType
    if targetDirPath.exists(): return

    patch = PatchDataset()

    dtest   = patch.getSplit('test', dataType)
    deval   = patch.getSplit('eval', dataType)
    dtrain  = patch.getSplit('train', dataType)

    dd = datasets.DatasetDict({
        'train': dtrain,
        'test': dtest,
        'eval': deval
    })
    dd.save_to_disk(targetDirPath)

@utils.statusNotifier
def getHGDataset(dataType: str='msg'):
    assert(dataType in ['msg', 'code', 'all', 'hunk', 'allHunk'])
    targetDirPath = hgDATADIR / dataType
    assert(targetDirPath.exists())

    dd = datasets.load_from_disk(targetDirPath)

    return dd

if __name__ == '__main__':
    preprocessZipData()
    createRawDataset()
    processRawDataset()
    splitProcessedData()

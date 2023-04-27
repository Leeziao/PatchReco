from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent

DATADIR = PROJECT_ROOT / 'data'
RAWDATADIR = DATADIR / 'raw'
sRAWDATADIR = RAWDATADIR / 'security_patch'
pRAWDATADIR = RAWDATADIR / 'positives'
nRAWDATADIR = RAWDATADIR / 'negatives'

if os.environ.get('OUTPUT_DIR',None) is None:
    RESULTDIR = PROJECT_ROOT / 'result'
else:
    RESULTDIR = Path(os.environ['OUTPUT_DIR'])

if os.environ.get('CHECKPOINT_DIR',None) is None:
    CHECKPOINTDIR = PROJECT_ROOT / 'trainer'
else:
    CHECKPOINTDIR = Path(os.environ['CHECKPOINT_DIR'])

msgCHECKPOINTDIR = CHECKPOINTDIR / 'msgTrainer'
codeCHECKPOINTDIR = CHECKPOINTDIR / 'codeTrainer'

DATADIR.mkdir(exist_ok=True, parents=True)
RESULTDIR.mkdir(exist_ok=True, parents=True)
CHECKPOINTDIR.mkdir(exist_ok=True, parents=True)

hpFilePath = PROJECT_ROOT / 'hp.json'

rawDataFilePath = DATADIR / 'rawData.json'
processedDataFilePath = DATADIR / 'processedData.json'
splitDataFilePath = DATADIR / 'splitProcessedData.json'


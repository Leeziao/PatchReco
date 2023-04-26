from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATADIR = PROJECT_ROOT / 'data'
RAWDATADIR = DATADIR / 'raw'
sRAWDATADIR = RAWDATADIR / 'security_patch'
pRAWDATADIR = RAWDATADIR / 'positives'
nRAWDATADIR = RAWDATADIR / 'negatives'

RESULTDIR = PROJECT_ROOT / 'result'

CHECKPOINTDIR = PROJECT_ROOT / 'trainer'
msgCHECKPOINTDIR = CHECKPOINTDIR / 'msgTrainer'
codeCHECKPOINTDIR = CHECKPOINTDIR / 'codeTrainer'

DATADIR.mkdir(exist_ok=True, parents=True)
RESULTDIR.mkdir(exist_ok=True, parents=True)
CHECKPOINTDIR.mkdir(exist_ok=True, parents=True)

hpFilePath = PROJECT_ROOT / 'hp.json'

rawDataFilePath = DATADIR / 'rawData.json'
processedDataFilePath = DATADIR / 'processedData.json'
splitDataFilePath = DATADIR / 'splitProcessedData.json'


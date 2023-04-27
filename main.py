import dataset
import training
import evaluating
import argparse

from transformers import RobertaForSequenceClassification

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modelType', choices=['code', 'msg', 'all'], type=str)
    args = parser.parse_args()

    dataset.preprocessZipData()
    dataset.createRawDataset()
    dataset.processRawDataset()
    dataset.splitProcessedData()

    modelType = args.modelType
    dataset.createHGDataset(modelType)

    if modelType == 'all':
        training.trainAllModel()
    else:
        training.trainSeperateModel(args.modelType)
    # evaluating.evaluteSeperateModel(args.modelType)

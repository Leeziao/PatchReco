import dataset
import training
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modelType', choices=['code', 'msg'], type=str)
    args = parser.parse_args()

    dataset.preprocessZipData()
    dataset.createRawDataset()
    dataset.processRawDataset()
    dataset.splitProcessedData()

    training.trainModel(args.modelType)

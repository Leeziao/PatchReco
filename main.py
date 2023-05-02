import dataset
import training
import evaluating
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        choices=['train', 'eval'],
                        type=str)
    parser.add_argument('modelType',
                        choices=['code', 'msg', 'all', "hunk", 'allHunk'],
                        type=str)
    parser.add_argument('--fileList', type=str, default='',
                        help='The fileList specify the order of creating the rawDataset, thus affects the data sequences')
    args = parser.parse_args()

    dataset.preprocessZipData()
    dataset.createRawDataset(args.fileList)
    dataset.processRawDataset()
    dataset.splitProcessedData()

    modelType, mode = args.modelType, args.mode
    dataset.createHGDataset(modelType)

    if mode == 'eval':
        evaluating.evaluteModel(args.modelType)
    else:
        if modelType in ['all', 'allHunk']:
            training.trainAllModel(modelType)
        else:
            training.trainSeperateModel(args.modelType)

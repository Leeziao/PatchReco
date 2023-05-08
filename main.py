import dataset
import training
import evaluating
import argparse
import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        choices=['train', 'eval', 'predict'],
                        type=str)
    parser.add_argument('modelType',
                        choices=['code', 'msg', 'all', "hunk", 'allHunk'],
                        type=str)
    parser.add_argument('--fileList', type=str, default='',
                        help='The fileList specify the order of creating the rawDataset, thus affects the data sequences')
    parser.add_argument('--patch', type=str, default='',
                        help='The patch file for predict')
    args = parser.parse_args()

    dataset.preprocessZipData()
    dataset.createRawDataset(args.fileList)
    dataset.processRawDataset()
    dataset.splitProcessedData()

    modelType, mode = args.modelType, args.mode
    dataset.createHGDataset(modelType)

    if mode == 'eval':
        evaluating.evaluteModel(args.modelType)
    elif mode == 'train':
        if modelType in ['all', 'allHunk']:
            training.trainAllModel(modelType)
        else:
            training.trainSeperateModel(args.modelType)
    elif mode == 'predict':
        result = predict.predictItem(modelType, args.patch)
        resultStr = 'Security Patch' if result else 'Non Security Patch'
        print(f'Prediction result is [{resultStr}]')

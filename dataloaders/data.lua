local M = {}

local options = require '../train_options'

local image = require 'image'
local tnt = require 'torchnet'

-- Current directory changes to the running directory
local trainingMetadata = torch.load('./data/train_data/train_full_metadata.t7')
local trainSize = trainingMetadata:size(1)

local TRAIN_PATH = './data/train_cropped_'
if options.imageSize <= 128 then
    TRAIN_PATH = TRAIN_PATH .. '128/'
else
    TRAIN_PATH = TRAIN_PATH .. '256/'
end

function resize(img)
    if options.imageSize == 128 or options.imageSize == 256 then
        return img
    end

    return image.scale(img, options.imageSize, options.imageSize)
end

function getTrainSample(idx)
    sample = trainingMetadata[idx]
    if sample[2] == 1 then
        file_name = tostring(sample[1]) .. '_left.jpeg'
    else
        file_name = tostring(sample[1]) .. '_right.jpeg'
    end

    return resize(image.load(TRAIN_PATH .. file_name))
end

function getTrainLabel(idx)
    return torch.LongTensor{trainingMetadata[idx][8] + 1}
end

-- Pre-Load all the data!
trainImages = torch.Tensor(trainSize, 3, options.imageSize, options.imageSize)
trainLabels = torch.LongTensor(trainSize, 1)
for i=1, trainSize do
    trainImages[i] = getTrainSample(i)
    trainLabels[i] = getTrainLabel(i)
end

--Creating the datasets
trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',
   
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, trainSize):long(),
            load = function(idx)
                return {
                    input = trainImages[idx],
                    target = trainLabels[idx]
                }
            end
        }
    }
}

function getBatchIterator(dataset_to_iterate)
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = options.batchsize,
            dataset = dataset_to_iterate
        }
    }
end

function M.getValIterator()
    trainDataset:select('val')
    return getBatchIterator(trainDataset)
end

function M.getTrainIterator()
    trainDataset:select('train')
    return getBatchIterator(trainDataset)
end

function M.getTestIterator()
    --We currently don't have the test set (It's still undergoing pre-processing)
    trainDataset:select('val')
    return getBatchIterator(trainDataset)
end

return M
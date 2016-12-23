local M = {}

local options = require '../train_options'

local image = require 'image'
local tnt = require 'torchnet'

local perturb = require '../lib/perturbations'

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
local trainImages = torch.Tensor(trainSize, 3, options.imageSize, options.imageSize)
local trainLabels = torch.LongTensor(trainSize, 1)
for i=1, trainSize do
    trainImages[i] = getTrainSample(i)
    trainLabels[i] = getTrainLabel(i)
end

print("Data loaded, Now normalizing...")

perturb.normalize_parameters = torch.load(TRAIN_PATH .. 'normalize_parameters.t7')
perturb.pca = torch.load(TRAIN_PATH .. 'PCA.t7')

local trainSetSize = (torch.Tensor{0.9 * trainSize}):floor()[1]

local perturbImage = function (inp)
    f = tnt.transform.compose {
        perturb.BrightnessJitter,
        perturb.ContrastJitter,
        perturb.SaturationJitter,
        perturb.RandomRotate,
        perturb.RandomTranslate,
        perturb.RandomHFlip,
        perturb.RandomVFlip,
        perturb.LightingJitter,
        perturb.normalize
    }
    return f(inp)
end

--Creating the datasets
local trainDataset = tnt.ListDataset{
        list = torch.range(1, trainSetSize):long(),
        load = function(idx)
            return {
                input = perturbImage(trainImages[idx]),
                target = trainLabels[idx]
            }
        end
}

local validationDataset = tnt.ListDataset {
    list = torch.range(trainSetSize+1, trainSize):long(),
    load = function(idx)
        return {
            input = trainImages[idx],
            target = trainLabels[idx]
        }
    end
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
    return getBatchIterator(validationDataset)
end

if options.cuda then
    M.getTrainIterator = function ()
        return tnt.ParallelDatasetIterator {
            nthread = options.nCudaThreads,

            init = function ()
                local tnt = require 'torchnet'
            end,

            closure = function ()
                local image = require 'image'
                local perturb = require '../lib/perturbations'

                local perturbImage = function (inp)
                    f = tnt.transform.compose {
                        perturb.BrightnessJitter,
                        perturb.ContrastJitter,
                        perturb.SaturationJitter,
                        perturb.RandomRotate,
                        perturb.RandomTranslate,
                        perturb.RandomHFlip,
                        perturb.RandomVFlip,
                        perturb.LightingJitter,
                        perturb.normalize
                    }
                    return f(inp)
                end

                return tnt.BatchDataset {
                    batchsize = options.batchsize,

                    dataset = tnt.ListDataset{
                            list = torch.range(1, trainSetSize):long(),
                            load = function(idx)
                                return {
                                    input = perturbImage(trainImages[idx]),
                                    target = trainLabels[idx]
                                }
                            end
                    }
                }
            end
        }
    end
else
    M.getTrainIterator = function ()
        return getBatchIterator(trainDataset)
    end
end

return M
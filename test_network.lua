require 'nn'
require 'optim'
require 'os'
require 'xlua'

local tnt = require 'torchnet'
local image = require 'image'

local cmd = torch.CmdLine();

cmd:text()
cmd:text('Diabetic Retinopathy - submission fle generation: ')
cmd:text()
cmd:text('Options:')
cmd:option('-model',                '',             'Model to use for generating ')
cmd:option('-imageSize',            128,            'Image size for the network input')
cmd:option('-batchsize',            32,             'Batch Size')
cmd:option('-cuda',                 false,          'Switch on CUDA')

local options = cmd:parse(arg or {})

if options.model == '' or not paths.filep('run_models/'..options.model..'.model') then
    cmd:error('Invalid model ' .. options.model)
end

if options.cuda then
    require 'cunn'
    require 'cudnn' -- faster convolutions

    cudnn.benchmark = true
    cudnn.fastest = true
    cudnn.verbose = true
end

local file_name = options.model

local trained_network = torch.load('run_models/'..options.model..'.model')


local testingMetadata = torch.load('./data/test_data/test_full_metadata.t7')
local testingSize = testingMetadata:size(1)

local TEST_PATH = './data/test_cropped_'
if options.imageSize <= 128 then
    TEST_PATH = TEST_PATH .. '128/'
else
    TEST_PATH = TEST_PATH .. '256/'
end
function resize(img)
    if options.imageSize == 128 or options.imageSize == 256 then
        return img
    end

    return image.scale(img, options.imageSize, options.imageSize)
end

function getTestSample(idx)
    sample = testingMetadata[idx]
    if sample[2] == 1 then
        file_name = tostring(sample[1]) .. '_left.jpeg'
    else
        file_name = tostring(sample[1]) .. '_right.jpeg'
    end

    return resize(image.load(TEST_PATH .. file_name))
end

function getTestLabel(idx)
    sample = testingMetadata[idx]
    if sample[2] == 1 then
        file_name = tostring(sample[1]) .. '_left.jpeg'
    else
        file_name = tostring(sample[1]) .. '_right.jpeg'
    end

    return file_name
end

function getBatchIterator(dataset_to_iterate)
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = options.batchsize,
            dataset = dataset_to_iterate
        }
    }
end

function getTestIterator()
    return getBatchIterator(tnt.ListDataset{
        list = torch.range(1, testingSize):long(),
        load = function(idx)
            return {
                input = getTestSample(idx),
                target = getTestLabel(idx)
            }
        end
    })
end

--[[
--  This piece of code creates the submission
--  file that has to be uploaded on kaggle.
--]]
local submission = assert(io.open("submissions/" .. file_name .. ".csv", "w"))
submission:write("image,level\n")
batch = 1

local engine = tnt.OptimEngine()

engine.hooks.onForward = function(state)
    local fileNames  = state.sample.target
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end

    if options.verbose == true then
        print(string.format("%s Batch: %d/%d;", "test", batch, state.iterator.dataset:size()))
    end

    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network = trained_network,
    iterator = getTestIterator()
}
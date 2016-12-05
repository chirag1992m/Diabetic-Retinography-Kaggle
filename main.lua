require 'nn'
require 'optim'
require 'os'

local tnt = require 'torchnet'
local options = require 'options'
local image = require 'image'

if options.cuda then
    require 'cunn'
    require 'cudnn' -- faster convolutions

    cudnn.benchmark = true
    cudnn.fastest = true
    cudnn.verbose = true
end

local file_suffix = options.suffix .. string.format("_%d", os.time());

torch.setdefaulttensortype('torch.DoubleTensor')

torch.setnumthreads(options.nThreads)
torch.manualSeed(options.manualSeed)

if options.cuda then
    cutorch.manualSeedAll(options.manualSeed)
end


print("Fetching data... ")
local data = require ('dataloader/' .. options.dataloader)

print("Loading Network... ")
local model = require("models/".. options.model)

local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

local trainingLosses, trainingErrors = {}, {}
local validationLosses, validationErrors = {}, {}
local timeVals = {}


if options.cuda then
    model = model:cuda()
    criterion = criterion:cuda()
end

print('\nModel: ' .. tostring(model) .. '\n')

local epoch = 1

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end


if options.cuda then
    local inputGPU = torch.CudaTensor()
    local targetGPU = torch.CudaTensor()
    
    engine.hooks.onSample = function(state)
        inputGPU:resize(state.sample.input:size() ):copy(state.sample.input)
        targetGPU:resize(state.sample.target:size()):copy(state.sample.target)
        state.sample.input  = inputGPU
        state.sample.target = targetGPU
    end
end

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if options.verbose == true then
        print(string.format("%s Batch: %d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, meter:value(), clerr:value{k = 1}))
    end

    if mode == 'Train' then
        intermediateTL[batch] = meter:value()
        intermediateTE[batch] = clerr:value{k = 1}
    else
        intermediateVL[batch] = meter:value()
        intermediateVE[batch] = clerr:value{k = 1}
    end

    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))

    if mode == 'Train' then
        timeVals[epoch] = timer:value()
        trainingLosses[epoch] = meter:value()
	    trainingErrors[epoch] = clerr:value{k = 1}
    else
    	validationLosses[epoch] = meter:value()
	    validationErrors[epoch] = clerr:value{k = 1}
    end
end

print("Training Started")
while epoch <= options.nEpochs do
    engine:train{
        network = model,
        criterion = criterion,
        iterator = data.getTrainIterator(),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = options.LR,
            momentum = options.momentum,
            weightDecay = options.weightDecay,
            learningRateDecay = options.lrDecay
        }
    }

    engine:test{
        network = model,
        criterion = criterion,
        iterator = data.getValIterator()
    }

    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

local submission = assert(io.open("submissions/submission_" .. file_suffix .. ".csv", "w"))
submission:write("image,level\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded on kaggle.
--]]
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
    network = model,
    iterator = data.getTestIterator()
}

-- Dump the results in files
model:clearState()
torch.save("run_models/model_" .. file_suffix .. ".model", model)

torch.save("logs/trainingErrors_" .. file_suffix .. ".log", torch.Tensor(trainingErrors))
torch.save("logs/trainingLosses_" .. file_suffix .. ".log", torch.Tensor(trainingLosses))
torch.save("logs/validationErrors_" .. file_suffix .. ".log", torch.Tensor(validationErrors))
torch.save("logs/validationLosses_" .. file_suffix .. ".log", torch.Tensor(validationLosses))
torch.save("logs/timers" .. file_suffix .. ".log", torch.Tensor(timeVals))
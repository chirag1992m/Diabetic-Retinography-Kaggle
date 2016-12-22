require 'nn'
require 'optim'
require 'os'

local tnt = require 'torchnet'
local options = require 'train_options'
local image = require 'image'

if options.cuda then
    print("Turning CUDA on...")

    require 'cunn'
    require 'cudnn' -- faster convolutions

    cudnn.benchmark = true
    cudnn.fastest = true
    cudnn.verbose = true
end

local file_name = ''
if options.suffix == '' then 
    file_name = string.format("_%d", os.time()) 
else 
    file_name = options.suffix
end

torch.setdefaulttensortype('torch.DoubleTensor')

torch.setnumthreads(options.nThreads)
torch.manualSeed(options.manualSeed)

if options.cuda then
    cutorch.manualSeedAll(options.manualSeed)
end

print("Loading Network... ")
local model = require("models/".. options.model)
print('\nModel: ' .. tostring(model) .. '\n')
print("Model Fetched!")

print("\nFetching data... ")
local data = require("dataloaders/" .. options.dataloader)
print("Data Fetched!\n")

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
        inputGPU:resize(state.sample.input:size()):copy(state.sample.input)
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

    if not options.noValidation then
        engine:test{
            network = model,
            criterion = criterion,
            iterator = data.getValIterator()
        }
    end

    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

-- Dump the results in files
if not options.dontSave then
    model:clearState()
    torch.save("run_models/" .. file_name .. ".model", model)

    torch.save("logs/trainingErrors/" .. file_name .. ".log", torch.Tensor(trainingErrors))
    torch.save("logs/trainingLosses/" .. file_name .. ".log", torch.Tensor(trainingLosses))
    if not options.noValidation then
        torch.save("logs/validationErrors/" .. file_name .. ".log", torch.Tensor(validationErrors))
        torch.save("logs/validationLosses/" .. file_name .. ".log", torch.Tensor(validationLosses))
    end
    torch.save("logs/timers/" .. file_name .. ".log", torch.Tensor(timeVals))
end
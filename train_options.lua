local cmd = torch.CmdLine();

cmd:text()
cmd:text('Diabetic Retinopathy Modelling: ')
cmd:text()
cmd:text('Options:')
cmd:option('-dataloader',           '',             'Dataloader to use')
cmd:option('-model',                '',             'Model to use for training')
cmd:option('-imageSize',			128,			'Image Size as input')
cmd:option('-nEpochs',              50,             'Maximum epochs')
cmd:option('-batchsize',            32,             'Batch size for epochs')
cmd:option('-nThreads',             1,              'Number of dataloading threads')
cmd:option('-manualSeed',           '0',            'Manual seed for RNG')
cmd:option('-LR',                   0.1,            'initial learning rate')
cmd:option('-momentum',             0.9,            'momentum')
cmd:option('-weightDecay',          1e-4,           'weight decay')
cmd:option('-lrDecay',          	1e-4,           'Learning Rate decay')
cmd:option('-verbose',              false,          'Print stats for every batch')
cmd:option('-suffix',               '',             'Suffix to add on all output files')
cmd:option('-cuda',					false,			'Use cuda tensor')
cmd:option('-nCudaThreads',         1,              'Number of CUDA dataloading threads')
cmd:option('-noValidation',			false,			'Only training needs to be done?')
cmd:option('-dontSave',				false,			'Don\'t save the readings')

local opt = cmd:parse(arg or {})

if opt.model == '' or not paths.filep('models/'..opt.model..'.lua') then
    cmd:error('Invalid model ' .. opt.model)
end

if opt.dataloader == '' or not paths.filep('dataloaders/'..opt.dataloader..'.lua') then
    cmd:error('Invalid dataloader ' .. opt.dataloader)
end

if opt.imageSize < 1 or opt.imageSize > 256 then
	cmd:error('Invalid imageSize ' .. opt.imageSize)
end

return opt
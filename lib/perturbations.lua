local image = require 'image'

local M = {}

M.channels = 3
M.jitterVariation = 0.4
M.jitterProbability = 0.5


M.normalize_parameters = {
    mean = torch.Tensor{ 0.0, 0.0, 0.0},
    std = { 1.0, 1.0, 1.0}
}
function M.normalize(inp)
	inp = inp:clone()

	for i=1, M.channels do
        inp[{i, {}, {}}]:add(-M.normalize_parameters.mean[i])
        inp[{i, {}, {}}]:div(M.normalize_parameters.std[i])
    end
    return inp
end


M.pca = {
    eigval = torch.Tensor{0.0, 0.0, 0.0},

    eigvec = torch.Tensor{
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}
    }
}
M.lighting_alphastd = 0.1
function M.LightingJitter(inp)
    if torch.uniform() < M.jitterProbability then
        return inp
    end

    if not M.pca then
    	return inp
    end

    local alpha = torch.Tensor(3):normal(0, M.lighting_alphastd)
    local channel_inp = M.pca.eigvec:clone()
        :cmul(alpha:view(1, 3):expand(3, 3))
        :cmul(M.pca.eigval:view(1, 3):expand(3, 3))
        :sum(2)
        :squeeze()

    inp = inp:clone()
    for i=1, M.channels do
        inp[i]:add(channel_inp[i])
    end
    return inp
end


function M.BrightnessJitter(inp)
    if torch.uniform() < M.jitterProbability then
        return inp
    end

    inp = inp:clone()
    jitter = 1.0 - (torch.uniform(-M.jitterVariation, M.jitterVariation))
    inp:mul(jitter)
    return inp
end

function M.ContrastJitter(inp)
    if torch.uniform() < M.jitterProbability then
        return inp
    end

    inp = inp:clone()
    local contrast = torch.Tensor(inp:size())
    for i=1, inp:size()[1] do
        contrast[i]:fill(inp[i]:mean())
    end
    jitter = 1.0 - (torch.uniform(-M.jitterVariation, M.jitterVariation))

    return inp:mul(jitter):add(1 - jitter, contrast)
end

function M.SaturationJitter(inp)
    if torch.uniform() < M.jitterProbability then
        return inp
    end

    inp = inp:clone()
    local saturation = torch.Tensor(inp:size())
    saturation[1]:zero()
    saturation[1]:add(0.299, inp[1]):add(0.587, inp[2]):add(0.114, inp[3])
    saturation[2]:copy(saturation[1])
    saturation[3]:copy(saturation[1])
    jitter = 1.0 - (torch.uniform(-M.jitterVariation, M.jitterVariation))

    return inp:mul(jitter):add(1 - jitter, saturation)
end

function M.RandomRotate(inp)
    if torch.uniform() < M.jitterProbability then
        return inp
    end

    inp = inp:clone()
    return image.rotate(inp, torch.uniform(-1.0, 1.0) * math.pi, 'bilinear')
end

function M.RandomTranslate(inp)
    if torch.uniform() < M.jitterProbability then
        return inp
    end

    inp = inp:clone()
    jitter = torch.uniform(-M.jitterVariation*10, M.jitterVariation*10)
    return image.translate(inp, jitter, jitter)
end

function M.RandomHFlip(inp)
	if torch.uniform() < M.jitterProbability then
        return inp
    end

    inp = inp:clone()
    return image.hflip(inp)
end

function M.RandomVFlip(inp)
	if torch.uniform() < M.jitterProbability then
        return inp
    end
    inp = inp:clone()
    return image.vflip(inp)
end

return M
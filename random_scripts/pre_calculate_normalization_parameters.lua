require 'torch'
require 'xlua'
image = require 'image'

sample = torch.load('../data/sample/sample_full_metadata.t7')

normalize_parameters = {
    mean = {
        0.0,
        0.0,
        0.0
    },
    
    std = {
        1.0,
        1.0,
        1.0
    }
}

size = sample:size(1)

--For pre_processes 128x128 files
images = torch.Tensor(size, 3, 128, 128)
for i=1, size do
    if sample[i][2] == 1 then
        image_name = tostring(sample[i][1]) .. '_left.jpeg'
    else
        image_name = tostring(sample[i][1]) .. '_right.jpeg'
    end
    images[i] = image.load('../data/sample_cropped_128/' .. image_name)

    xlua.progress(i, size)
end
for i=1, 3 do
    normalize_parameters.mean[i] = images[{{}, i, {}, {}}]:mean()
    normalize_parameters.std[i] = images[{{}, i, {}, {}}]:std()
end
torch.save('../data/sample_cropped_128/normalize_parameters.t7', normalize_parameters)

--For pre_processes 256x256 files
images = torch.Tensor(size, 3, 256, 256)
collectgarbage()
for i=1, size do
    if sample[i][2] == 1 then
        image_name = tostring(sample[i][1]) .. '_left.jpeg'
    else
        image_name = tostring(sample[i][1]) .. '_right.jpeg'
    end
    images[i] = image.load('../data/sample_cropped_256/' .. image_name)

    xlua.progress(i, size)
end
for i=1, 3 do
    normalize_parameters.mean[i] = images[{{}, i, {}, {}}]:mean()
    normalize_parameters.std[i] = images[{{}, i, {}, {}}]:std()
end
torch.save('../data/sample_cropped_256/normalize_parameters.t7', normalize_parameters)

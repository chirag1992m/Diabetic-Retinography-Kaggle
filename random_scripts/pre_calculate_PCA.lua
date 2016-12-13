require 'torch'
require 'xlua'
image = require 'image'

sample = torch.load('../data/sample/sample_full_metadata.t7')
size = sample:size(1)


--Calculating PCA for 128x128 images
images = torch.Tensor(size, 3, 128, 128)
for i=1,size do
    if sample[i][2] == 1 then
        file_name = tostring(sample[i][1]) .. '_left.jpeg'
    else
        file_name = tostring(sample[i][1]) .. '_right.jpeg'
    end
    
    images[i] = image.load('../data/sample_cropped_128/' .. file_name)

    xlua.progress(i, size)
end
images = images:permute(1, 3, 4, 2)
images:resize(size*128*128, 3)

covariance = torch.Tensor(3, 3)
covariance = (covariance:addmm(images:t(), images))/images:size(1)

PCA = {}
PCA.eigval, PCA.eigvec = torch.symeig(covariance)
PCA.eigval:sqrt()
torch.save('../data/sample_cropped_128/PCA.t7', PCA)


--Calculating PCA for 256x256 images
images = torch.Tensor(size, 3, 256, 256)
collectgarbage()
for i=1,size do
    if sample[i][2] == 1 then
        file_name = tostring(sample[i][1]) .. '_left.jpeg'
    else
        file_name = tostring(sample[i][1]) .. '_right.jpeg'
    end
    
    images[i] = image.load('../data/sample_cropped_256/' .. file_name)

    xlua.progress(i, size)
end
images = images:permute(1, 3, 4, 2)
images:resize(size*256*256, 3)

covariance = torch.Tensor(3, 3)
covariance = (covariance:addmm(images:t(), images))/images:size(1)

PCA = {}
PCA.eigval, PCA.eigvec = torch.symeig(covariance)
PCA.eigval:sqrt()
torch.save('../data/sample_cropped_256/PCA.t7', PCA)

require 'torch'
require 'xlua'
image = require 'image'

data = torch.load('../data/sample/sample_full_metadata.t7')
size = data:size(1)

for i=1, size do
    collectgarbage() --remove unnecessary used memory. Greatly increases performance too!

    if data[i][2] == 1 then
        file_name = string.format("%d_left.jpeg", data[i][1])
    else
        file_name = string.format("%d_right.jpeg", data[i][1])
    end
    
    img = image.load('../data/sample/' .. file_name)
    
    --Crop the image according to the parameters given and then resize and save
    cropped = image.crop(img, data[i][4], data[i][5], data[i][6], data[i][7])
    
    resized_1 = image.scale(cropped, 128, 128)
    image.save('../data/sample_cropped_128/' .. file_name, resized_1)
    
    resized_2 = image.scale(cropped, 256, 256)
    image.save('../data/sample_cropped_256/' .. file_name, resized_2)

-- Removing as no longer using 512x512 images
--    resized_3 = image.scale(cropped, 512, 512)
--    image.save('../data/sample_cropped_512/' .. file_name, resized_3)

    xlua.progress(i, size)
end

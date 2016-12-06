require 'torch'

local dir = io.popen("ls ./test_data/")

lines = {}
for line in dir:lines() do
    lines[#lines + 1] = line
end

eye_test_data = {}
for i=1, #lines do
    track, eye, file_extension = lines[i]:match("([^_]+)_([^.]+).([^,]+)")
    index = tonumber(track)
    if eye == "left" then
        eyetype = 1
    else
        eyetype = 2
    end
    if eye_test_data[index] == nil then
        eye_test_data[index] = {}
    end
    eye_test_data[index][eyetype] = 1
end

length = 0
for k, v in pairs(eye_test_data) do
    length = length+1
end

-- index 1 --> the track of file
-- 2 --> Left is available
-- 3 --> Right is available
eye_test_tensor = torch.LongTensor(length, 3)


index = 1
for k, v in pairs(eye_test_data) do
    eye_test_tensor[index][1] = k
    if v[1] == nil then
        eye_test_tensor[index][2] = -1
    else
        eye_test_tensor[index][2] = 1
    end
    
    if v[2] == nil then
        eye_test_tensor[index][3] = -1
    else
        eye_test_tensor[index][3] = 1
    end
    index = index + 1
end

torch.save('test.t7', eye_test_tensor)
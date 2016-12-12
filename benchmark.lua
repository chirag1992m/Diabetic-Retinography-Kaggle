--[[
The purpose of this file is to create a benchmark submission file with majority class model

label 0 - 25810
label 1 - 2443
label 2 - 5292
label 3 - 873
label 4 - 708
]]--

-- Accuracy of the majority model: 0.73478335 (According to training data)

-- Kappa (Competition Evaluation Method): 0.00000 (After submission. submissions/submission_benchmark.png)

local dir = io.popen("ls ./data/test_data/")

lines = {}
for line in dir:lines() do
    lines[#lines + 1] = line
end

local submission = assert(io.open("./submissions/submission_benchmark.csv", "w"))
submission:write("image,level\n")

for i=1, #lines do
    filename, file_extension = lines[i]:match("([^.]+).([^,]+)")
    submission:write(filename .. ',0\n')	--Writing 0 label for every class
end

submission:close()
-- A 9 Layer model proposed by Deepsense.io: https://deepsense.io/diagnosing-diabetic-retinopathy-with-deep-learning/
-- There is a bit difference as Deepsense uses 512x512 images and we are using 256x256
--Takes in input of size 3*256*256

require 'nn'

local Conv = nn.SpatialConvolution
local BatchNorm = nn.SpatialBatchNormalization
local NonLin = nn.ReLU
local MaxPool = nn.SpatialMaxPooling
local View = nn.View
local Lin = nn.Linear
local Reg = nn.Dropout

local model  = nn.Sequential()

model:add(Conv(3, 16, 3, 3)) --> 16 * 254 * 254
model:add(BatchNorm(16))
model:add(NonLin())

model:add(Conv(16, 16, 3, 3)) --> 16 * 252 * 252
model:add(BatchNorm(16))
model:add(NonLin())

model:add(MaxPool(2, 2, 2, 2)) --> 16 * 126 * 126

model:add(Conv(16, 32, 3, 3)) --> 32 * 124 * 124
model:add(BatchNorm(32))
model:add(NonLin())

model:add(Conv(32, 32, 3, 3)) --> 32 * 122 * 122
model:add(BatchNorm(32))
model:add(NonLin())

model:add(MaxPool(2, 2, 2, 2)) --> 32 * 61 * 61

model:add(Conv(32, 64, 3, 3)) --> 64 * 59 * 59
model:add(BatchNorm(64))
model:add(NonLin())

model:add(Conv(64, 64, 3, 3)) --> 64 * 57 * 57
model:add(BatchNorm(64))
model:add(NonLin())

model:add(MaxPool(2, 2, 2, 2)) --> 64 * 28 * 28

model:add(Conv(64, 96, 3, 3)) --> 96 * 26 * 26
model:add(BatchNorm(96))
model:add(NonLin())

model:add(MaxPool(2, 2, 2, 2)) --> 96 * 13 * 13

model:add(Conv(96, 96, 3, 3)) --> 96 * 11 * 11
model:add(BatchNorm(96))
model:add(NonLin())

model:add(MaxPool(2, 2, 2, 2)) --> 96 * 5 * 5

model:add(Conv(96, 128, 3, 3)) --> 128 * 3 * 3
model:add(BatchNorm(128))
model:add(NonLin())

model:add(View(1152))
model:add(Lin(1152, 96))
model:add(NonLin())

model:add(Reg(0.5))

model:add(Lin(96, 5))

return model
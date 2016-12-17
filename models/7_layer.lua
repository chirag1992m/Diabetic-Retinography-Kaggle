-- A basic 7_layer ConvNet.

-- This time we use LeakyRelU for faster convergence

--Takes in input of size 128x128

require 'nn'

local Conv = nn.SpatialConvolution
local NonLin = nn.LeakyReLU
local leakyReluParameter = 0.1
local MaxPool = nn.SpatialMaxPooling
local View = nn.View
local Lin = nn.Linear
local SBNorm = nn.SpatialBatchNormalization

local model  = nn.Sequential()

model:add(Conv(3, 16, 3, 3)) -- 16*126*126
model:add(SBNorm(16))
model:add(NonLin(leakyReluParameter))

model:add(Conv(16, 16, 3, 3)) -- 16*124*124
model:add(SBNorm(16))
model:add(NonLin(leakyReluParameter))

model:add(MaxPool(2,2,2,2)) -- 16*62*62

model:add(Conv(16, 32, 3, 3)) -- 32*60*60
model:add(SBNorm(32))
model:add(NonLin(leakyReluParameter))

model:add(MaxPool(2,2,2,2)) -- 32*30*30

model:add(Conv(32, 64, 3, 3)) -- 64*28*28
model:add(SBNorm(64))
model:add(NonLin(leakyReluParameter))

model:add(Conv(64, 64, 3, 3)) -- 64*26*26
model:add(SBNorm(64))
model:add(NonLin(leakyReluParameter))

model:add(MaxPool(2,2,2,2)) -- 64*13*13

model:add(Conv(64, 128, 3, 3)) -- 128*11*11
model:add(SBNorm(128))
model:add(NonLin(leakyReluParameter))

model:add(Conv(128, 128, 3, 3)) -- 128*9*9
model:add(SBNorm(128))
model:add(NonLin(leakyReluParameter))

model:add(MaxPool(2,2,2,2)) -- 128*4*4

model:add(View(2048)) -- 2048*1
model:add(Lin(2048, 128)) -- 128*1
model:add(NonLin(leakyReluParameter))
model:add(Lin(128, 5)) -- 5*1

return model
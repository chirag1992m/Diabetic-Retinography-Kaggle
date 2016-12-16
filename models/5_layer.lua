-- A basic/shallow 5_layer ConvNet.

--Takes in input of size 128x128

require 'nn'

local Conv = nn.SpatialConvolution
local NonLin = nn.Tanh
local MaxPool = nn.SpatialMaxPooling
local View = nn.View
local Lin = nn.Linear
local Reg1 = nn.Dropout

local model  = nn.Sequential()

model:add(Conv(3, 16, 3, 3)) -- 16*126*126
model:add(NonLin())
model:add(MaxPool(2,2,2,2)) -- 16*63*63

model:add(Conv(16, 32, 3, 3)) -- 32*61*61
model:add(NonLin())
model:add(MaxPool(2,2,2,2)) -- 32*30*30

model:add(Conv(32, 64, 3, 3)) -- 64*28*28
model:add(NonLin())
model:add(MaxPool(2,2,2,2)) -- 64*14*14

model:add(Conv(64, 128, 3, 3)) -- 128*12*12
model:add(NonLin())
model:add(MaxPool(2,2,2,2)) -- 128*6*6

model:add(Conv(128, 128, 3, 3)) -- 128*4*4
model:add(NonLin())
model:add(MaxPool(2,2,2,2)) -- 128*2*2

model:add(View(512)) -- 512*1
model:add(Lin(512, 128)) -- 128*1
model:add(NonLin())
model:add(Lin(128, 5)) -- 5*1

return model
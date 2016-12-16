-- A basic/shallow 4_layer ConvNet for testing code

--Takes in input of size 128x128

require 'nn'

local Conv = nn.SpatialConvolution
local NonLin = nn.Tanh
local MaxPool = nn.SpatialMaxPooling
local View = nn.View
local Lin = nn.Linear

local model  = nn.Sequential()

model:add(Conv(3, 16, 3, 3))
model:add(NonLin())
model:add(MaxPool(2,2,2,2))

model:add(Conv(16, 32, 3, 3))
model:add(NonLin())
model:add(MaxPool(2,2,2,2))

model:add(Conv(32, 64, 3, 3))
model:add(NonLin())
model:add(MaxPool(2,2,2,2))

model:add(Conv(64, 128, 3, 3))
model:add(NonLin())
model:add(MaxPool(2,2,2,2))

model:add(View(4608))
model:add(Lin(4608, 32))
model:add(NonLin())
model:add(Lin(32, 5))

return model
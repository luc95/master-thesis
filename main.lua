require "torch" 

cmd = torch.CmdLine()
cmd:option('-dataset', 'yeast', 'name of dataset used for training')

cmd:option('-optimizationMethod', 'SGD', 'optimization method: SGD, ADAM or RMSProp')
cmd:option('-learningRate', 0.01, 'initial learning rate')
cmd:option('-batchSize', 100, 'mini-batch size')
cmd:option('-noOfEpochs', 10, 'number of epochs')

params = cmd:parse(arg)
print(params)


-- Load data from dataset with provided name
-- dofile("./utils.lua")
dofile("./load_data.lua")
dofile("./model_test.lua")
dofile("./eval.lua")
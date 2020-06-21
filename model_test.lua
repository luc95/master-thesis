require "nn"
require "xlua"
require "optim"
require "nngraph"

print("\n---------------------------------------------------------------------------------")
print("=> Creating the model...")
print("---------------------------------------------------------------------------------")

local number_of_features = 13

model = nn.Sequential()

neural_network = nn.Sequential()
neural_network:add(nn.TemporalConvolution(number_of_features, 16, 3, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.TemporalMaxPooling(3))

neural_network:add(nn.TemporalConvolution(16, 64, 3, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.TemporalMaxPooling(3))

neural_network:add(nn.TemporalConvolution(64, 128, 3, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.TemporalMaxPooling(3))

neural_network:add(nn.TemporalConvolution(128, 256, 3, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.TemporalMaxPooling(3))

neural_network:add(nn.TemporalConvolution(256, 512, 3, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.TemporalMaxPooling(3))

neural_network:add(nn.Reshape(512))

linear_module_1 = nn.Sequential()
linear_module_1:add(nn.Linear(512, 512) )
linear_module_1:add(nn.BatchNormalization(512))
linear_module_1:add(nn.ReLU(true))

linear_module_2 = nn.Sequential()
linear_module_2:add(nn.Linear(512, 512) )
linear_module_2:add(nn.BatchNormalization(512))
linear_module_2:add(nn.ReLU(true))

parallel = nn.Sequential()
siamese_neural_network = nn.ParallelTable()
siamese_neural_network:add(neural_network):add(neural_network:clone('weight', 'bias', 'gradWeight', 'gradBias'))

random_projection_module_1 = nn.Sequential()
random_projection_module_1:add(nn.ConcatTable():add(linear_module_1):add(linear_module_2))
random_projection_module_1:add(nn.JoinTable(2))

random_projection_module_2 = nn.Sequential()
random_projection_module_2:add(nn.ConcatTable():add(linear_module_2:clone('weight', 'bias', 'gradWeight', 'gradBias')):add(linear_module_1:clone('weight', 'bias', 'gradWeight', 'gradBias')))
random_projection_module_2:add(nn.JoinTable(2))

parallel:add(siamese_neural_network)
parallel:add(nn.ParallelTable():add(random_projection_module_1):add(random_projection_module_2))
parallel:add(nn.CMulTable())
parallel:add(nn.View(1024))
parallel:add(nn.Linear(1024, 1))
parallel:add(nn.View(1))
parallel:add(nn.Sigmoid())

model:add(parallel)

print("Created the model:")
-- print(model)

for k, v in pairs(model:findModules("nn.TemporalConvolution")) do
    local n = v.kW*v.outputFrameSize
    v.weight:normal(0, math.sqrt(2/n))
    v.bias:zero()
end

for k, v in pairs(model:findModules("nn.Linear")) do
    v.bias:zero()
end


print("\n---------------------------------------------------------------------------------")
print("=> Training on "..params.dataset.." dataset...")
print("---------------------------------------------------------------------------------")

criterion = nn.BCECriterion()
-- model:cuda()
-- criterion:cuda()

local total_batches = math.floor(#input_data[1] / params.batchSize)

for e = 1, params.noOfEpochs do
    model:training()
    -- xlua.progress(e, params.noOfEpochs)

    print("=> Epoch "..e.."/"..params.noOfEpochs)
    -- print(parameters)
    -- print(input_data)
    parameters, gradParameters = model:getParameters()

    for t = 1, total_batches*params.batchSize, params.batchSize do
        xlua.progress(t + params.batchSize - 1, total_batches*params.batchSize)

        local input_batch_1 = input_data[1][t]
        local input_batch_2 = input_data[2][t]
        local labels_batch = {}
        labels_batch[1] = labels[t]
        for i = 2, params.batchSize do
            input_batch_1 = torch.cat(input_batch_1, input_data[1][t + i - 1], 1)
            input_batch_2 = torch.cat(input_batch_2, input_data[2][t + i - 1], 1)

            table.insert(labels_batch, labels[t + i - 1])
        end
        local input_batch = {input_batch_1, input_batch_2}
        -- print(input_batch)
        -- print(labels_batch)

        local labels_tensor = torch.Tensor(labels_batch)
        model:zeroGradParameters()
        -- print("labels")
        -- print(labels_tensor)
        function feval(parameters)
            gradParameters:zero()
            local model_output = model:forward(input_batch)
            -- print("output")
            -- print(model_output)
            local loss = criterion:forward(model_output, labels_tensor)
            local dloss_doutput = criterion:backward(model_output, labels_tensor)
            model:backward(input_batch, dloss_doutput)

            -- model:updateParameters(params.learningRate)
            return loss, gradParameters
        end
        optim.sgd(feval, parameters, {learningRate = params.learningRate})
    end

    collectgarbage()
end

print("=> Finished training!")

print("Test train set:")

local tptn = 0
for t = 1, total_batches*params.batchSize, params.batchSize do
    xlua.progress(t + params.batchSize - 1, total_batches*params.batchSize)


    local input_batch_1 = input_data[1][t]
    local input_batch_2 = input_data[2][t]
    local labels_batch = {}
    labels_batch[1] = labels[t]
    for i = 2, params.batchSize do
        input_batch_1 = torch.cat(input_batch_1, input_data[1][t + i - 1], 1)
        input_batch_2 = torch.cat(input_batch_2, input_data[2][t + i - 1], 1)
        table.insert(labels_batch, labels[t + i - 1])
    end
    local input_batch = {input_batch_1, input_batch_2}

    local labels_tensor = torch.Tensor(labels_batch)
   
    local output = model:forward(input_batch)

    for i = 1, output:size(1) do
        local prediction = 0
        if output[i][1] > 0.5 then
            prediction = 1
        end
        if (prediction == labels_tensor[i]) then
            tptn = tptn + 1
        end
    end
end
print('Accuracy train:')
print(tptn/(total_batches*params.batchSize))


print("\n---------------------------------------------------------------------------------")
print("=> Testing on "..params.dataset.." dataset...")
print("---------------------------------------------------------------------------------")
local test_total_batches = math.floor(#test_data[1] / params.batchSize)

local tptn = 0
for t = 1, test_total_batches*params.batchSize, params.batchSize do
    xlua.progress(t + params.batchSize - 1, test_total_batches*params.batchSize)

    local test_batch_1 = test_data[1][t]
    local test_batch_2 = test_data[2][t]
    local test_labels_batch = {}
    test_labels_batch[1] = test_labels[t]
    for i = 2, params.batchSize do
        test_batch_1 = torch.cat(test_batch_1, test_data[1][t + i - 1], 1)
        test_batch_2 = torch.cat(test_batch_2, test_data[2][t + i - 1], 1)
        table.insert(test_labels_batch, test_labels[t + i - 1])
    end
    local test_batch = {test_batch_1, test_batch_2}

    local test_labels_tensor = torch.Tensor(test_labels_batch)
   
    local output = model:forward(test_batch)

    for i = 1, output:size(1) do
        local prediction = 0
        if output[i][1] > 0.5 then
            prediction = 1
        end
        if (prediction == test_labels_tensor[i]) then
            tptn = tptn + 1
        end
    end
end

print('Accuracy:')
print(tptn/(test_total_batches*params.batchSize))

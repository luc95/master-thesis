require "nn"
require "xlua"

print("\n---------------------------------------------------------------------------------")
print("=> Creating the model...")
print("---------------------------------------------------------------------------------")

local number_of_features = 13

model = nn.Sequential()

neural_network = nn.Sequential()
neural_network:add(nn.JoinTable(1))

neural_network:add(nn.TemporalConvolution(number_of_features, 64, 1, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.TemporalMaxPooling(1))

neural_network:add(nn.TemporalConvolution(64, 128, 1, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.TemporalMaxPooling(1))

neural_network:add(nn.TemporalConvolution(128, 256, 1, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.TemporalMaxPooling(1))

neural_network:add(nn.TemporalConvolution(256, 512, 1, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.TemporalMaxPooling(1))

neural_network:add(nn.Reshape(512*512))

model:add(nn.MapTable():add(neural_network))

linear_module_1 = nn.Sequential()
linear_module_1:add(nn.Linear(512*512, 512) )
linear_module_1:add(nn.BatchNormalization(512, 1e-6, 0.1, false))
linear_module_1:add(nn.ReLU(true))

linear_module_2 = nn.Sequential()
linear_module_2:add(nn.Linear(512*512, 512) )
linear_module_2:add(nn.BatchNormalization(512, 1e-6, 0.1, false))
linear_module_2:add(nn.ReLU(true))


random_projection_module_1 = nn.Sequential()
random_projection_module_1:add(nn.ConcatTable():add(linear_module_1):add(linear_module_2))
random_projection_module_1:add(nn.JoinTable(2) )
random_projection_module_1:add(nn.Reshape(1024) )

random_projection_module_2 = nn.Sequential()
random_projection_module_2:add(nn.ConcatTable():add(linear_module_2:clone('weight', 'bias', 'gradWeight', 'gradBias')):add(linear_module_1:clone('weight', 'bias', 'gradWeight', 'gradBias')))
random_projection_module_2:add(nn.JoinTable(2))
random_projection_module_2:add(nn.Reshape(1024))

model:add(nn.ParallelTable():add(random_projection_module_1):add(random_projection_module_2))
model:add(nn.CMulTable())
model:add(nn.View(1024))
model:add(nn.Linear(1024, 1))
model:add(nn.View(1))
model:add(nn.Sigmoid())


print("Created the model:")
print(model)
-- print(model:forward(input_data))
-- print(input_data)
print("\n---------------------------------------------------------------------------------")
print("=> Training on "..params.dataset.." dataset...")
print("---------------------------------------------------------------------------------")
criterion = nn.MSECriterion()
local total_batches = math.floor(#input_data[1] / params.batchSize)

for t = 1, total_batches*params.batchSize, params.batchSize do
    xlua.progress(t, total_batches*params.batchSize)

    input_batch = {}
    input_batch[1] = {}
    input_batch[2] = {}
    labels_batch = {}

    for i = 1, params.batchSize do
        input_batch[1][i] = input_data[1][t + i - 1]
        input_batch[2][i] = input_data[2][t + i - 1]
        table.insert(labels_batch, labels[t + i - 1])
    end

    print(input_batch)

    labels_tensor = torch.Tensor(labels_batch)
    print(labels_tensor)


    model:zeroGradParameters()

    criterion:forward(model:forward(input_batch), labels_tensor)

    model:backward(input_batch, criterion:backward(model.output, labels_tensor))

    model:updateParameters(params.learningRate)

    collectgarbage()
end

print("=> Finished!")

-- criterion:forward(model:forward(input_data), labels)
-- model:zeroGradParameters()
-- model:backward(input_data, criterion:backward(model.output, labels))
-- model:updateParameters(params.learningRate)

print(model:forward(test_data))

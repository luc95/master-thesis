require 'optim'
require 'math'

print("\n---------------------------------------------------------------------------------")
print("=> Evaluating on "..params.dataset.." dataset - 5-fold cross validation")
print("---------------------------------------------------------------------------------")
function create_folds()
    local k = 5
    folds = {}
    folds_labels = {}

    for i=1, k do
        folds[i] = {}
        folds[i][1] = {}
        folds[i][2] = {}
        folds_labels[i] = {}
    end

    local positive_interactions = {}
    positive_interactions[1] = {}
    positive_interactions[2] = {}
    local negative_interactions = {}
    negative_interactions[1] = {}
    negative_interactions[2] = {}

    for i=1, labels:size(1) do
        if labels[i] == 1 then
            table.insert(positive_interactions[1], input_data[1][i])
            table.insert(positive_interactions[2], input_data[2][i])
        else 
            table.insert(negative_interactions[1], input_data[1][i])
            table.insert(negative_interactions[2], input_data[2][i])
        end
    end

    -- print(positive_interactions)

    local positive_fold_size = math.floor(#positive_interactions[1] / k)
    local negative_fold_size = math.floor(#negative_interactions[1] / k)

    local ratio = math.floor(positive_fold_size / negative_fold_size)
    local fold_size = math.floor(labels:size(1) / k)

    local fold_idx = 0
    for i=1, fold_size * k, fold_size do
        fold_idx = fold_idx + 1
        local count = 0
        local pos_count = 0
        local neg_count = 0
        for j=1, fold_size do
            count = count + 1
            if count <= ratio then
                pos_count = pos_count + 1
                table.insert(folds[fold_idx][1], positive_interactions[1][(fold_idx-1)*positive_fold_size + pos_count])
                table.insert(folds[fold_idx][2], positive_interactions[2][(fold_idx-1)*positive_fold_size + pos_count])
                table.insert(folds_labels[fold_idx], 1)
            else
                neg_count = neg_count + 1
                table.insert(folds[fold_idx][1], negative_interactions[1][(fold_idx-1)*negative_fold_size + neg_count])
                table.insert(folds[fold_idx][2], negative_interactions[2][(fold_idx-1)*negative_fold_size + neg_count])
                table.insert(folds_labels[fold_idx], 0)
            end
            if count > ratio then
                count = 0
            end
        end
    end

    -- local fold_idx = 0
    -- for i=1, positive_fold_size * k, positive_fold_size do
    --     fold_idx = fold_idx + 1
    --     for j=1, positive_fold_size do
    --         table.insert(folds[fold_idx][1], positive_interactions[1][i + j - 1])
    --         table.insert(folds[fold_idx][2], positive_interactions[2][i + j - 1])
    --         table.insert(folds_labels[fold_idx], 1)
    --     end
    -- end

    -- local fold_idx = 0
    -- for i=1, negative_fold_size * k, negative_fold_size do
    --     fold_idx = fold_idx + 1
    --     for j=1, negative_fold_size do
    --         table.insert(folds[fold_idx][1], negative_interactions[1][i + j - 1])
    --         table.insert(folds[fold_idx][2], negative_interactions[2][i + j - 1])
    --         table.insert(folds_labels[fold_idx], 0)
    --     end
    -- end

    
    -- for l=1, k do
    --     for i = #folds[l][1], 2, -1 do
    --         local j = math.random(i)
    --         folds[l][1][i], folds[l][1][j] = folds[l][1][j], folds[l][1][i]
    --         folds[l][2][i], folds[l][2][j] = folds[l][2][j], folds[l][2][i]
    --         folds_labels[l][i], folds_labels[l][j] = folds_labels[l][j], folds_labels[l][i]
    --     end
    -- end

end

create_folds()


local optimizationMethod = optim.sgd
local optimizationConfig = {
    learningRate = params.learningRate,
    momentum = 0.9,
    learningRateDecay = 0.000001
    -- nesterov = true,
    -- dampening = 0
}

if params.optimzationMethod == "ADAM" then
    optimizationMethod = optim.adam
    optimizationConfig = {
        learningRate = params.learningRate,
        epsilon = 1e-6
    }
elseif params.optimzationMethod == "RMSProp" then
    optimizationMethod = optim.rmsprop
    optimizationConfig = {
        learningRate = params.learningRate,
        epsilon = 1e-6,
        alpha = 0.9
    }
end


local avg_acc = 0
local avg_p = 0
local avg_r = 0
local avg_s = 0
local avg_f1 = 0

for test_idx = 1, 5 do

    local train_data = {}
    train_data[1] = {}
    train_data[2] = {}
    local train_labels = {}

    local test_data = {}
    test_data[1] = {}
    test_data[2] = {}
    local test_labels = {}

    for fold_idx = 1, 5 do
        if fold_idx == test_idx then
            for _, v in ipairs(folds[fold_idx][1]) do
                table.insert(test_data[1], v)
            end
            for _, v in ipairs(folds[fold_idx][2]) do
                table.insert(test_data[2], v)
            end
            for _, v in ipairs(folds_labels[fold_idx]) do
                table.insert(test_labels, v)
            end
        else
            for _, v in ipairs(folds[fold_idx][1]) do
                table.insert(train_data[1], v)
            end
            for _, v in ipairs(folds[fold_idx][2]) do
                table.insert(train_data[2], v)
            end
            for _, v in ipairs(folds_labels[fold_idx]) do
                table.insert(train_labels, v)
            end
        end
    end

    local total_batches = math.floor(#train_data[1] / params.batchSize)

    local shuffle = torch.randperm(total_batches)

    for e = 1, params.noOfEpochs do
        print("=> Epoch "..e.."/"..params.noOfEpochs)

        local time = sys.clock()
        parameters, gradParameters = model:getParameters()

        model:training()
        local k = 0
        for t = 1, total_batches*params.batchSize, params.batchSize do
            xlua.progress(t + params.batchSize - 1, total_batches*params.batchSize)
            
            k = k + 1
            local idx = shuffle[k]

            local input_batch_1 = train_data[1][idx]
            local input_batch_2 = train_data[2][idx]
            local labels_batch = {}
            labels_batch[1] = train_labels[idx]
            for i = 2, params.batchSize do
                input_batch_1 = torch.cat(input_batch_1, train_data[1][idx + i - 1], 1)
                input_batch_2 = torch.cat(input_batch_2, train_data[2][idx + i - 1], 1)
                table.insert(labels_batch, train_labels[idx + i - 1])
            end
            local input_batch = {input_batch_1, input_batch_2}
            local labels_tensor = torch.Tensor(labels_batch)
            -- print(labels_tensor)
            model:zeroGradParameters()
            local feval = function(x)
                if x ~= parameters then
                    parameters:copy(x)
                end

                gradParameters:zero()
                local model_output = model:forward(input_batch)
                local loss = criterion:forward(model_output, labels_tensor)
                -- print(loss)
                local dloss_doutput = criterion:backward(model_output, labels_tensor)
                model:backward(input_batch, dloss_doutput)
                return loss, gradParameters
            end
            optimizationMethod(feval, parameters, optimizationConfig)
        end
        print("Time needed for completing the epoch: "..(sys.clock() - time).." seconds")

        parameters, gradParameters = nil, nil
        collectgarbage()
    end

    print("=> Finished training!")

    print("\n---------------------------------------------------------------------------------")
    print("=> Testing on "..params.dataset.." dataset...")
    print("---------------------------------------------------------------------------------")
    local test_total_batches = math.floor(#test_data[1] / params.batchSize)
    local shuffle = torch.randperm(test_total_batches)

    local tp = 0
    local tn = 0
    local fp = 0
    local fn = 0
    local k = 0
    model:evaluate()
    for t = 1, test_total_batches*params.batchSize, params.batchSize do
        xlua.progress(t + params.batchSize - 1, test_total_batches*params.batchSize)

        k = k + 1
        local idx = shuffle[k]
        local test_batch_1 = test_data[1][idx]
        local test_batch_2 = test_data[2][idx]
        local test_labels_batch = {}
        test_labels_batch[1] = test_labels[idx]
        for i = 2, params.batchSize do
            test_batch_1 = torch.cat(test_batch_1, test_data[1][idx + i - 1], 1)
            test_batch_2 = torch.cat(test_batch_2, test_data[2][idx + i - 1], 1)
            table.insert(test_labels_batch, test_labels[idx + i - 1])
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
                if prediction == 1 then
                    tp = tp + 1
                else
                    tn = tn + 1
                end
            else
                if prediction == 1 then
                    fp = fp + 1
                else
                    fn = fn + 1
                end
            end
        end
    end

    local total = test_total_batches*params.batchSize
    local acc = (tp+tn)/total
    avg_acc = avg_acc + acc
    print("Acc: "..acc)

    local precision = tp/(tp+fp)
    avg_p = avg_p + precision
    print("P: "..precision)

    local recall = tp/(tp+fn)
    avg_r = avg_r + recall
    print("R: "..recall)

    local specificiy = tn/(tn+fn)
    avg_s = avg_s + specificiy
    print("S: "..specificiy)

    local f1 = 2*precision*recall/(precision + recall)
    avg_f1 = avg_f1 + f1
    print("F1: "..f1)
    
end

print("Average accuracy: ".. avg_acc / 5)
print("Average precision: ".. avg_p / 5)
print("Average recall: ".. avg_r / 5)
print("Average specificity: ".. avg_s / 5)
print("Average F1: ".. avg_f1 / 5)


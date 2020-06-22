require 'xlua'

print("\n---------------------------------------------------------------------------------")
print("=> Loading data...")
print("---------------------------------------------------------------------------------")

amino_acid_embeddings = {}

local function get_amino_acid_embeddings(file_name)
    io.input(file_name)
    -- local amino_acid_embeddings = {}
    for line in io.lines() do
        local amino_acid
        local amino_acid_vector = {}
        for element in string.gmatch(line, "%S+") do
            if string.find(element, "%a") then
                amino_acid = element
            else 
                table.insert(amino_acid_vector, element) 
            end
        end
        amino_acid_embeddings[amino_acid] = amino_acid_vector
    end
    -- return amino_acid_embeddings
end


local function convert_tsv_to_dat(file_name)
    print("   => *.dat format does not exist")
    print("   => Converting *.tsv to *.dat")

    io.input(file_name)

    local dataset = {}
    for line in io.lines() do
        local protein_pair_interaction = {}
        for element in string.gmatch(line, "%S+") do
            table.insert(protein_pair_interaction, element)
        end
        dataset[#dataset+1] = protein_pair_interaction
    end
    -- './'..params.dataset..'/'..params.dataset..'.protein.actions.dat'
    torch.save(string.gsub(file_name, ".tsv", ".dat"), dataset)
end

local function convert_tsv_to_dat_with_keys(file_name)
    print("   => *.dat format does not exist")
    print("   => Converting *.tsv to *.dat with keys")

    io.input(file_name)

    local dataset = {}
    for line in io.lines() do
        local protein_pair_interaction = {}
        for element in string.gmatch(line, "%S+") do
            table.insert(protein_pair_interaction, element)
        end
        dataset[protein_pair_interaction[1]] = protein_pair_interaction[2]
    end
    -- './'..params.dataset..'/'..params.dataset..'.protein.actions.dat'
    torch.save(string.gsub(file_name, ".tsv", ".dat"), dataset)
end

local function load_protein_pairs_interactions(file_name)
    local protein_pairs_interactions_data = torch.load(file_name)
    -- print(protein_pairs_interactions_data)
    return protein_pairs_interactions_data
end

local function load_protein_sequences(file_name)
    local protein_sequences = torch.load(file_name)
    -- print(protein_sequences)
    return protein_sequences
end

local function get_protein_profile(protein_sequence, length)
    -- Replace every amino acid (char) to its numeric represenation
    length = length or 512
    local amino_acid_embedding_dimen = table.getn(amino_acid_embeddings["A"])
    local protein_profile = torch.Tensor(1, length, amino_acid_embedding_dimen):zero()   
    local i = 1 
    for amino_acid in string.gmatch(protein_sequence, ".") do
        if i == length+1 then
            break
        end        
        for j=1, amino_acid_embedding_dimen do
            protein_profile[1][i][j] = amino_acid_embeddings[amino_acid][j]
        end
        i = i + 1
    end
    return protein_profile
end

-- protein_sequence_indexes = {}
-- other_protein_sequence_indexes = {}
input_data = {}
input_data[1] = {}
input_data[2] = {}


train_data = {}
train_data[1] = {}
train_data[2] = {}

test_data = {}
test_data[1] = {}
test_data[2] = {}

local function create_protein_sequences_tensors(protein_pairs_interactions, protein_sequences)
    -- Sequences of proteins in pair added to one table
    local interactions = {}
    local train_interactions = {}
    local test_interactions = {}

    local negative_interactions = {}
    local positive_interactions = {}
    for index, protein_pair in ipairs(protein_pairs_interactions) do
        if protein_pair[3] == "0" then 
            table.insert(negative_interactions, protein_pair)
        else 
            table.insert(positive_interactions,protein_pair)
        end
    end

    print("=> Creating the tensors...")
    for index, protein_pair in ipairs(positive_interactions) do
        local protein_profile = get_protein_profile(protein_sequences[protein_pair[1]])
        local other_protein_profile = get_protein_profile(protein_sequences[protein_pair[2]])

        if (index < #positive_interactions - 0.2*#positive_interactions) then 
            table.insert(train_data[1], protein_profile )
            table.insert(train_data[2], other_protein_profile )
            table.insert(train_interactions, protein_pair[3])
        else
            table.insert(test_data[1], protein_profile)
            table.insert(test_data[2], other_protein_profile)
            table.insert(test_interactions, protein_pair[3])
        end

        table.insert(input_data[1], protein_profile )
        table.insert(input_data[2], other_protein_profile )
        table.insert(interactions, protein_pair[3])

        xlua.progress(index, #positive_interactions)
    end

    for index, protein_pair in ipairs(negative_interactions) do
        local protein_profile = get_protein_profile(protein_sequences[protein_pair[1]])
        local other_protein_profile = get_protein_profile(protein_sequences[protein_pair[2]])

        if (index < #negative_interactions - 0.2*#negative_interactions) then 
            table.insert(train_data[1], protein_profile )
            table.insert(train_data[2], other_protein_profile )
            table.insert(train_interactions, protein_pair[3])
        else
            table.insert(test_data[1], protein_profile)
            table.insert(test_data[2], other_protein_profile)
            table.insert(test_interactions, protein_pair[3])
        end

        table.insert(input_data[1], protein_profile )
        table.insert(input_data[2], other_protein_profile )
        table.insert(interactions, protein_pair[3])

        xlua.progress(index, #negative_interactions)
    end

    -- shuffle data
    for i = #input_data[1], 2, -1 do
        local j = math.random(i)
        input_data[1][i], input_data[1][j] = input_data[1][j], input_data[1][i]
        input_data[2][i], input_data[2][j] = input_data[2][j], input_data[2][i]
        interactions[i], interactions[j] = interactions[j], interactions[i]
    end

    for i = #train_data[1], 2, -1 do
        local j = math.random(i)
        train_data[1][i], train_data[1][j] = train_data[1][j], train_data[1][i]
        train_data[2][i], train_data[2][j] = train_data[2][j], train_data[2][i]
        train_interactions[i], train_interactions[j] = train_interactions[j], train_interactions[i]
    end

    for i = #test_data[1], 2, -1 do
        local j = math.random(i)
        test_data[1][i], test_data[1][j] = test_data[1][j], test_data[1][i]
        test_data[2][i], test_data[2][j] = test_data[2][j], test_data[2][i]
        test_interactions[i], test_interactions[j] = test_interactions[j], test_interactions[i]
    end

    print("\n")
    labels = torch.Tensor(interactions)
    train_labels = torch.Tensor(train_interactions)
    test_labels = torch.Tensor(test_interactions)

    torch.save(params.dataset..'.protein.features.all.t7', input_data)
    torch.save(params.dataset..'.protein.labels.all.t7', labels)
    torch.save(params.dataset..'.protein.features.train.t7', train_data)
    torch.save(params.dataset..'.protein.features.test.t7', test_data)
    torch.save(params.dataset..'.protein.labels.train.t7', train_labels)
    torch.save(params.dataset..'.protein.labels.test.t7', test_labels)

end

-- loading the dataset
if params.dataset then 

    local time = sys.clock()

    get_amino_acid_embeddings('./amino_acid_embeddings.txt')

    local file_path = './'..params.dataset..'/'..params.dataset..'.protein.actions'
    if path.exists(file_path..'.dat') == false then
        convert_tsv_to_dat(file_path..'.tsv')
    end

    local protein_pairs_interactions = load_protein_pairs_interactions(file_path..'.dat')
    -- print(#protein_pairs_interactions)

    file_path = './'..params.dataset..'/'..params.dataset..'.protein.dictionary'
    if path.exists(file_path..'.dat') == false then
        convert_tsv_to_dat_with_keys(file_path..'.tsv')
    end

    local protein_sequences = load_protein_sequences(file_path..'.dat')

    local train_data_file = './'..params.dataset..'.protein.features.train.t7'
    if path.exists(train_data_file) == false then
        create_protein_sequences_tensors(protein_pairs_interactions, protein_sequences)
    else
        train_data = torch.load(train_data_file)
        test_data = torch.load('./'..params.dataset..'.protein.features.test.t7')
        train_labels = torch.load('./'..params.dataset..'.protein.labels.train.t7')
        test_labels = torch.load('./'..params.dataset..'.protein.labels.test.t7')
        input_data = torch.load('./'..params.dataset..'.protein.features.all.t7')
        labels = torch.load('./'..params.dataset..'.protein.labels.all.t7')
        print(labels:size(1))
    end

   
    -- print(input_data)
    -- print("Time needed for creating and loading train and test data: ")
    -- print(sys.clock() - time)


else
  error( 'You have to provide the dataset name!')
end
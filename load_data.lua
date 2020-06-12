require 'xlua'

print("\n---------------------------------------------------------------------------------")
print("=> Loading data...")
print("---------------------------------------------------------------------------------")

amino_acid_embeddings = {}

-- amino_acid_embeddings["A"]=0.0799912015849807 --A
-- amino_acid_embeddings["R"]=0.0484482507611578 --R
-- amino_acid_embeddings["N"]=0.044293531582512 --N
-- amino_acid_embeddings["D"]=0.0578891399707563 --D
-- amino_acid_embeddings["C"]=0.0171846021407367 --C
-- amino_acid_embeddings["Q"]=0.0380578923048682 --Q
-- amino_acid_embeddings["E"]=0.0638169929675978 --E
-- amino_acid_embeddings["G"]=0.0760659374742852 --G
-- amino_acid_embeddings["H"]=0.0223465499452473 --H
-- amino_acid_embeddings["I"]=0.0550905793661343 --I
-- amino_acid_embeddings["L"]=0.0866897071203864 --L
-- amino_acid_embeddings["K"]=0.060458245507428 --K
-- amino_acid_embeddings["M"]=0.0215379186368154 --M
-- amino_acid_embeddings["F"]=0.0396348024787477 --F
-- amino_acid_embeddings["P"]=0.0465746314476874 --P
-- amino_acid_embeddings["S"]=0.0630028230885602 --S
-- amino_acid_embeddings["T"]=0.0580394726014824 --T
-- amino_acid_embeddings["W"]=0.0144991866213453 --W
-- amino_acid_embeddings["Y"]=0.03635438623143 --Y
-- amino_acid_embeddings["V"]=0.0700241481678408 --V

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

protein_sequence_indexes = {}
other_protein_sequence_indexes = {}
input_data = {}
input_data[1] = {}
input_data[2] = {}

test_data = {}
test_data[1] = {}
test_data[2] = {}

local function create_protein_sequences_tensors(protein_pairs_interactions, protein_sequences)
    -- Sequences of proteins in pair added to one table
    local protein_pairs_sequences = {}
    local interactions = {}

    print("=> Creating the tensors...")
    for index, protein_pair in ipairs(protein_pairs_interactions) do
        local protein_profile = get_protein_profile(protein_sequences[protein_pair[1]])
        local other_protein_profile = get_protein_profile(protein_sequences[protein_pair[2]])

        table.insert(protein_sequence_indexes, #protein_pairs_sequences+1)
        protein_pairs_sequences[#protein_pairs_sequences+1] = protein_profile
        table.insert(other_protein_sequence_indexes, #protein_pairs_sequences+1)
        protein_pairs_sequences[#protein_pairs_sequences+1] = other_protein_profile
        
        if (index < #protein_pairs_interactions - 0.3*#protein_pairs_interactions) then 
            table.insert(input_data[1], protein_profile )
            table.insert(input_data[2], other_protein_profile )
            table.insert(interactions, protein_pair[3])
        else
            table.insert(test_data[1], protein_profile)
            table.insert(test_data[2], other_protein_profile)
        end

        -- table.insert(interactions, protein_pair[3])
        xlua.progress(index, #protein_pairs_interactions)
    end

    print("\n")
    labels = torch.Tensor(interactions)
    torch.save(params.dataset..'.protein.features.train.t7', input_data)
    torch.save(params.dataset..'.protein.features.test.t7', test_data)
    torch.save(params.dataset..'.protein.labels.train.t7', labels)
end

-- loading the dataset
if params.dataset then 

    local time = sys.clock()

    get_amino_acid_embeddings('./amino_acid_embeddings.txt')

    local file_path = './'..params.dataset..'/'..params.dataset..'_small.protein.actions'
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
        input_data = torch.load(train_data_file)
        test_data = torch.load('./'..params.dataset..'.protein.features.test.t7')
        labels = torch.load('./'..params.dataset..'.protein.labels.train.t7')
    end

    -- print(input_data)
    -- print("Time needed for creating and loading train and test data: ")
    -- print(sys.clock() - time)


else
  error( 'You have to provide the dataset name!')
end
function [training_data, validation_data, testing_data] = split_dataset(data)
    % split_dataset: function to perform a dataset split
    % perform a random split of data 
    % 60% training data, 20% validation data, 20% testing data

    % get length of data
    n = length(data);

    % firstly, randomize data
    random_order = randperm(n);
    random_data = data(random_order, :);

    % now split
    first_split_point = floor(0.6 * n);
    second_split_point = floor(0.8 * n);

    training_data = random_data(1:first_split_point, :);
    validation_data = random_data((first_split_point + 1):second_split_point, :);
    testing_data = random_data((second_split_point + 1):end, :);
end

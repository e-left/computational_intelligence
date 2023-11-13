function [x, y] = extract_xy_data(data)
    % extract_xy_data: extract training inputs and labels from data

    % assuming that final column is label and the others are inputs
    x = data(:, 1:(end - 1));
    y = data(:, end);
end


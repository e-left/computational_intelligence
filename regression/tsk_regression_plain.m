% clear
clear;
close all;
clc;

% load data
data = importdata('airfoil_self_noise.dat');

% split data
[training_data, validation_data, testing_data] = split_dataset(data);

% construct xy data
[train_x, train_y] = extract_xy_data(training_data);
[validate_x, validate_y] = extract_xy_data(validation_data);
[test_x, test_y] = extract_xy_data(testing_data);

% create cell array to hold models
tsk_models = cell(4, 1);

% create metrics functions
MSE = @(yhat, y) mean((yhat - y).^2);
RMSE = @(yhat, y) sqrt(MSE(yhat,y));
R2 = @(yhat, y) 1-sum((y-yhat).^2)/sum((y-mean(y)).^2);

NMSE = @(yhat, y) 1 - R2(yhat,y);
NDEI = @(yhat, y) sqrt(NMSE(yhat,y));

% construct models, train them and extract training curves, error curves
% and metrics

metrics = ones(4, 5);

for iModel = 1:4
    % setup fis options 
    
    % grid search
    fis_options = genfisOptions('GridPartition');
    
    % number of membership functions
    if mod(iModel, 2) == 0
        number_of_mf = 3;
    else
        number_of_mf = 2;
    end

    fis_options.NumMembershipFunctions = number_of_mf;

    % type of output
    if iModel < 3
        output_mf_type = 'constant';
    else
        output_mf_type = 'linear';
    end

    fis_options.OutputMembershipFunctionType = output_mf_type;

    fis_model = genfis(train_x, train_y, fis_options);

    % plot membership functions
    figure("Name", sprintf("Model no %d", iModel));
    for iMF = 1:size(train_x, 2)
        subplot(size(train_x, 2), 1, iMF);
        plotmf(fis_model, 'input', iMF);
        xlabel(sprintf("Input MF no %d", iMF));
    end
    saveas(gcf, [pwd sprintf('/plain_mf_untrained_%d.png', iModel)]);

    % train fis
    % create options for anfis
    anfis_options = anfisOptions;
    anfis_options.EpochNumber = 100; 
    anfis_options.ValidationData = validation_data;
    anfis_options.InitialFIS = fis_model;

    % create anfis - trained fis
    [trained_fis , training_error, step_size, validation_fis, validation_error] = anfis(training_data, anfis_options);

    % plot membership functions for tuned fis
    figure("Name", sprintf("Model no %d (trained)", iModel));
    for iMF = 1:size(train_x, 2)
        subplot(size(train_x, 2), 1, iMF);
        plotmf(validation_fis, 'input', iMF);
        xlabel(sprintf("Input MF no %d", iMF));
    end
    saveas(gcf, [pwd sprintf('/plain_mf_trained_%d.png', iModel)]);

    % plot training and validation error
    figure("Name", sprintf("Training and Validation error no %d", iModel));
    plot([training_error, validation_error], "LineWidth", 2.5);
    grid on;
    xlabel("Number of epochs");
    ylabel("RMS Error");
    legend("Training", "Validation");
    title(sprintf("RMS error curves for model %d", iModel));
    saveas(gcf, [pwd sprintf('/plain_training_validation_error_%d.png', iModel)]);

    % predict on test data
    y_predicted = evalfis(validation_fis, test_x);

    % keep metrics from test data
    model_metrics = [MSE(y_predicted, test_y), RMSE(y_predicted, test_y), R2(y_predicted, test_y), NMSE(y_predicted, test_y), NDEI(y_predicted, test_y)];
    metrics(iModel, :) = model_metrics(:);

    % plot test data error
    test_error = y_predicted - test_y;
    figure("Name", sprintf("Test error no %d", iModel));
    plot(test_error, "LineWidth", 2.5);
    grid on;
    xlabel("Sample no");
    ylabel("error");
    title(sprintf("Test error for model %d", iModel));
    saveas(gcf, [pwd sprintf('/plain_test_error_%d.png', iModel)]);
end

% display metrics
for iModel = 1:4
    metrics_model = metrics(iModel, :);
    fprintf("For model %d: MSE = %f, RMSE = %f, R2 = %f, NMSE = %f, NDEI = %f\n", iModel, metrics_model(1), metrics_model(2), metrics_model(3), metrics_model(4), metrics_model(5));
end
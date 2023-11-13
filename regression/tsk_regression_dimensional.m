% clear
clear;
close all;
clc;

% load data
data = importdata('superconduct.csv');

% split data
[training_data, validation_data, testing_data] = split_dataset(data);

% construct xy data
[train_x, train_y] = extract_xy_data(training_data);
[validate_x, validate_y] = extract_xy_data(validation_data);
[test_x, test_y] = extract_xy_data(testing_data);

% create metrics functions
MSE = @(yhat, y) mean((yhat - y).^2);
RMSE = @(yhat, y) sqrt(MSE(yhat,y));
R2 = @(yhat, y) 1-sum((y-yhat).^2)/sum((y-mean(y)).^2);

NMSE = @(yhat, y) 1 - R2(yhat,y);
NDEI = @(yhat, y) sqrt(NMSE(yhat,y));

% determine values for number of clusters
[idx, weight] = relieff(data(:, 1:end - 1),data(:, end), 10, 'method', 'regression');

% plot the weights of the most significant predictors
bar(weight(idx));
ylabel("Weights");
saveas(gcf, [pwd '/relieff_plot.png']);

% create grid for parameters to be searched
number_of_features = [6, 8, 10];
cluster_radius = [0.3, 0.6, 0.9];
k_folds = 5;

grid_search_errors = zeros(length(number_of_features), length(cluster_radius), 5);
number_of_rules = zeros(length(number_of_features), length(cluster_radius));

% perform grid search
for iNumberOfFeatures = 1:length(number_of_features)
    number_of_features_s = number_of_features(iNumberOfFeatures);

    % keep only the respective most important features for input
    x_train_feat = train_x(:, idx(1:number_of_features_s));
    x_test_feat = test_x(:, idx(1:number_of_features_s));

    for iClusterRadius = 1:length(cluster_radius)
        cluster_radius_s = cluster_radius(iClusterRadius);
        % partition into 5 folds
        train_cross_idx = cvpartition(train_y, "KFold", k_folds);
        cross_validation_errors = zeros(k_folds, 5);

        % for every k fold calculate metrics
        for iK = 1:k_folds
            fprintf("features = %d, cluster radius = %f, k fold = %d\n", number_of_features_s, cluster_radius_s, iK);
            train_x_kfold = x_train_feat(training(train_cross_idx, iK), :);
            train_y_kfold = train_y(training(train_cross_idx, iK), :);

            test_x_kfold = x_train_feat(training(train_cross_idx, iK), :);
            test_y_kfold = train_y(training(train_cross_idx, iK), :);

            training_data_kfold = [train_x_kfold train_y_kfold];
            validation_data_kfold = [test_x_kfold test_y_kfold];

            % create fis
            fis_options = genfisOptions("SubtractiveClustering", "ClusterInfluenceRange", cluster_radius_s);
            fis = genfis(train_x_kfold, train_y_kfold, fis_options);

            % perform this check to prevent an error
            if (size(fis.Rules,2) < 2)
                continue;
            end

            % trained anfis
            anfis_options = anfisOptions;
            anfis_options.InitialFIS = fis;
            anfis_options.EpochNumber = 50; 

            anfis_options.ValidationData = validation_data_kfold;

            [~, ~, ~, validation_fis, ~] = anfis(training_data_kfold, anfis_options);

            y_predicted = evalfis(validation_fis, x_test_feat);
            test_errors = [MSE(y_predicted, test_y)
                           RMSE(y_predicted, test_y)
                           NMSE(y_predicted, test_y)
                           NDEI(y_predicted, test_y) 
                           R2(y_predicted, test_y)];

            cross_validation_errors(iK, :) = test_errors(:);
        end

        % take average of all folds for this grid cell and fill it
        grid_cell_mse = mean(cross_validation_errors(:, 1));
        grid_cell_rmse = mean(cross_validation_errors(:, 2));
        grid_cell_r2 = mean(cross_validation_errors(:, 3));
        grid_cell_nmse = mean(cross_validation_errors(:, 4));
        grid_cell_ndei = mean(cross_validation_errors(:, 5));

        grid_cell_errors = [grid_cell_mse, grid_cell_rmse, grid_cell_r2, grid_cell_nmse, grid_cell_ndei];

        % save to grid
        grid_search_errors(iNumberOfFeatures, iClusterRadius, :) = grid_cell_errors(:);

        % save rule number
        number_of_rules(iNumberOfFeatures, iClusterRadius) = size(validation_fis.Rules, 2);
    end
end

% Error Plots
rmse_errors = grid_search_errors(:, :, 2);

% RMSE vs number of rules
figure("Name", "Errors vs Number of rules");
scatter(reshape(number_of_rules, 1, []), reshape(rmse_errors, 1, []), "r*");
hold on;
grid on;
xlabel("Number of Rules");
ylabel("Error");
legend("RMSE");
title("RMSE relevant to Number of Rules");
saveas(gcf, [pwd '/dimensional_rmse_rules.png']);

% RMSE vs parameters
[x_surface_data, y_surface_data]=meshgrid(number_of_features, cluster_radius);
figure("Name", "RMSE vs parameter values");
surf(x_surface_data, y_surface_data, rmse_errors');
xlabel("Number of Features");
ylabel("Cluster Radius");
zlabel("RMSE");
title("RMSE to Parameter Values");
saveas(gcf, [pwd '/dimensional_rmse_parameters.png']);

% Find minimum error model
invalid_configurations = rmse_errors == 0;
rmse_errors(invalid_configurations) = 200; % set to a large value to search for the minimum

[best_feature_idx, best_cluster_radius_idx] = find(rmse_errors == min(rmse_errors(:)));
best_feature_number = number_of_features(best_feature_idx);
best_cluster_radius = cluster_radius(best_cluster_radius_idx);
fprintf("Best parameters from grid search: number of features = %d, cluster radius = %f\n", best_feature_number, best_cluster_radius);

% keep only most important features
x_train_best = train_x(:, idx(1:best_feature_number));
x_validate_best = validate_x(:, idx(1:best_feature_number));
x_test_best = test_x(:, idx(1:best_feature_number));

training_best = [x_train_best train_y];
validate_best = [x_validate_best validate_y];

% set up best cluster radius
best_fis_options = genfisOptions("SubtractiveClustering", "ClusterInfluenceRange", best_cluster_radius);
best_fis = genfis(x_train_best, train_y, best_fis_options);

best_anfis_options = anfisOptions;
best_anfis_options.EpochNumber = 100;
best_anfis_options.ValidationData = validate_best;
best_anfis_options.InitialFIS = best_fis;

% train fis
[best_train_fis, best_training_error, ~, best_validated_fis, best_validation_error] = anfis(training_best, best_anfis_options);

% grab predictions for test input
best_y_predicted = evalfis(best_validated_fis, x_test_best);

% best model plots
% plot MF
for iMF = 1:length(best_validated_fis.input)
   figure("Name", sprintf("MF no %d", iMF));
   [xmf, ymf] = plotmf(best_fis, 'input', iMF);
   plot(xmf, ymf);
   grid on;
   hold on;
   [xmf_best, ymf_best] = plotmf(best_validated_fis, 'input', iMF);
   plot(xmf_best, ymf_best);
   xlabel('Input');
   ylabel('Degree of membership');
   title(sprintf("MF no %d", iMF));
   legend(sprintf("Input no %d (untrained)", iMF), sprintf("Input no %d (trained)", iMF));
   saveas(gcf, [pwd sprintf('/dimensional_mf_%d.png', iMF)])
end

% learning curves
figure("Name", "Learning curves");
plot([best_training_error, best_validation_error], "LineWidth", 2.5); 
grid on;
legend("Training rrror", "Validation rrror");
xlabel("No of Epochs");
ylabel("Error");
title("Best model learning curve");
saveas(gcf, [pwd '/dimensional_learning_curves.png']);

%Predictions vs targets
figure("Name", "Predictions on test set");
title('Prediction vs target');
xlabel('No of sample');
ylabel("Value");
plot(1:length(best_y_predicted), best_y_predicted, 'x', 'Color', 'red');
hold on;
grid on;
plot(1:length(best_y_predicted), test_y, 'o', 'Color', 'green');
legend("Predicted", "Actual");
saveas(gcf, [pwd '/dimensional_predicted_vs_actual.png']);

figure("Name", "Prediction errors");
title("Prediction errors");
xlabel('No of sample');
ylabel('Error');
plot(1:length(best_y_predicted), best_y_predicted - test_y);

% Display metrics 
metrics_model = [MSE(best_y_predicted, test_y), RMSE(best_y_predicted, test_y), R2(best_y_predicted, test_y), NMSE(best_y_predicted, test_y), NDEI(best_y_predicted, test_y)];
fprintf("For best model: MSE = %f, RMSE = %f, R2 = %f, NMSE = %f, NDEI = %f\n", metrics_model(1), metrics_model(2), metrics_model(3), metrics_model(4), metrics_model(5));
% clear
clear;
close all;
clc;

% load data
data = importdata('epileptic_seizure_data.csv');
data = data.data;

% split data
[training_data, validation_data, testing_data] = split_dataset(data);

% construct xy data
[train_x, train_y] = extract_xy_data(training_data);
[validate_x, validate_y] = extract_xy_data(validation_data);
[test_x, test_y] = extract_xy_data(testing_data);

validation_data(validation_data > 1) = 1;
validation_data(validation_data < 0) = 0;

testing_data(testing_data > 1) = 1;
testing_data(testing_data < 0) = 0;

anfis_options = anfisOptions;
anfis_options.EpochNumber = 50;

% determine values for number of clusters
[idx, weight] = relieff(data(:, 1:end - 1),data(:, end), 10, 'method','classification');

%Plot the weights of the most significant predictors
bar(weight(idx));
ylabel("Weights");
saveas(gcf, [pwd '/relieff_plot.png']);

number_of_features = [5, 10, 15];
cluster_radius = [0.3, 0.6, 0.9];
k_folds = 5; 

grid_oa = zeros(length(number_of_features), length(cluster_radius));
grid_error = zeros(length(number_of_features), length(cluster_radius));
grid_number_of_rules = zeros(length(number_of_features), length(cluster_radius));

for iNumberOfFeatures = 1:length(number_of_features)
    number_of_features_s = number_of_features(iNumberOfFeatures);

    % keep only the respective most important features for input
    x_train_feat = train_x(:, idx(1:number_of_features_s));
    x_test_feat = test_x(:, idx(1:number_of_features_s));

    for iClusterRadius = 1:length(cluster_radius)
        cluster_radius_s = cluster_radius(iClusterRadius);

        % partition into 5 folds
        train_cross_idx = cvpartition(train_y, "KFold", k_folds);

        oa_k_fold = zeros(k_folds, 1);
        cross_validation_errors = zeros(k_folds, 1);    
        number_of_rules_k_fold = zeros(k_folds, 1);

        for iK = 1:k_folds
            fprintf("features = %d, cluster radius = %f, k fold = %d\n", number_of_features_s, cluster_radius_s, iK);
            train_x_kfold = x_train_feat(training(train_cross_idx, iK), :);
            train_y_kfold = train_y(training(train_cross_idx, iK), :);

            test_x_kfold = x_train_feat(training(train_cross_idx, iK), :);
            test_y_kfold = train_y(training(train_cross_idx, iK), :);

            training_data_kfold = [train_x_kfold train_y_kfold];
            validation_data_kfold = [test_x_kfold test_y_kfold];

            [cluster_1, sigma_1] = subclust(training_data_kfold(train_y_kfold == 1, :), cluster_radius_s);
            [cluster_2, sigma_2] = subclust(training_data_kfold(train_y_kfold == 2, :), cluster_radius_s);
            [cluster_3, sigma_3] = subclust(training_data_kfold(train_y_kfold == 3, :), cluster_radius_s);
            [cluster_4, sigma_4] = subclust(training_data_kfold(train_y_kfold == 4, :), cluster_radius_s);
            [cluster_5, sigma_5] = subclust(training_data_kfold(train_y_kfold == 5, :), cluster_radius_s);

            num_of_rules = size(cluster_1, 1) + size(cluster_2, 1) + size(cluster_3, 1) + size(cluster_4, 1) + size(cluster_5, 1);

            % construct fis
            fis = sugfis;

            % add inputs, outpus and membership functions
            for i = 1:size(train_x_kfold, 2)
                name_input = sprintf("in%d", i);

                fis = addInput(fis,[0,1], "Name",name_input);

                for j = 1:size(cluster_1, 1)    
                    fis = addMF(fis, name_input, "gaussmf", [sigma_1(i) cluster_1(j,i)]);
                end

                for j = 1:size(cluster_2, 1)
                    fis = addMF(fis, name_input, "gaussmf", [sigma_2(i) cluster_2(j,i)]);
                end

                for j = 1:size(cluster_3, 1)
                    fis = addMF(fis, name_input, "gaussmf", [sigma_3(i) cluster_3(j,i)]);
                end

                for j = 1:size(cluster_4, 1)
                    fis = addMF(fis, name_input, "gaussmf", [sigma_4(i) cluster_4(j,i)]);
                end

                for j = 1:size(cluster_5, 1)
                    fis = addMF(fis, name_input, "gaussmf", [sigma_5(i) cluster_5(j,i)]);
                end

            end

            fis = addOutput(fis, [0,1], "Name", "out1");

            % add output mf
            params = [zeros(1, size(cluster_1, 1)) 0.25 * ones(1, size(cluster_2, 1)) 0.5 * ones(1, size(cluster_3, 1)) 0.75 * ones(1,size(cluster_4, 1)) ones(1, size(cluster_5, 1))];
            for i=1:num_of_rules
                fis = addMF(fis, "out1", 'constant', params(i));
            end

            % add rules
            rule_list = zeros(num_of_rules, size(training_data_kfold, 2));
            for i = 1:size(rule_list, 1)
                rule_list(i, :) = i;
            end
            rule_list = [rule_list, ones(num_of_rules, 2)];

            fis = addrule(fis, rule_list);

            % train and evaluate anfis
            anfis_options.InitialFIS = fis;
            anfis_options.ValidationData = validation_data_kfold;
            [~, ~, ~, validation_fis, validation_error] = anfis(training_data_kfold, anfis_options);

            y_predicted = evalfis(validation_fis, x_test_feat);
            y_predicted = round(y_predicted);
            y_predicted(y_predicted < 1) = 1;
            y_predicted(y_predicted > 5) = 5;

            error_matrix = confusionmat(test_y, y_predicted);
            oa_k_fold(iK) = trace(error_matrix) / sum(error_matrix, "all");
            number_of_rules_k_fold(iK) = size(validation_fis.Rules,2);
            cross_validation_errors(iK) = sqrt(sum(validation_error .* validation_error));
        end

        grid_oa(iNumberOfFeatures, iClusterRadius) = mean(oa_k_fold);
        grid_error(iNumberOfFeatures, iClusterRadius) = mean(cross_validation_errors);
        grid_number_of_rules(iNumberOfFeatures, iClusterRadius) = mean(number_of_rules_k_fold);

    end
end

% plot training data
% OA vs number of rules
figure("Name", "OA vs number of rules");
grid on;
scatter(reshape(grid_oa, 1, []), reshape(grid_number_of_rules, 1, []), "r*");
xlabel("OA");
ylabel("# of rules");
title("OA vs number of rules");
saveas(gcf, [pwd '/oa_vs_num_rules.png']);

% OA vs number of features vs cluster radius
figure("Name", "OA vs number of features vs cluster radius");
[x, y] = meshgrid(number_of_features, cluster_radius); 
surf(x, y, grid_oa');
xlabel("Number of features");
ylabel("Cluster radius");
zlabel("OA");
title("OA vs number of features vs cluster radius");
saveas(gcf, [pwd '/oa_vs_num_features_vs_cluster_radius.png']);

% find optimal model and train it
[optimal_number_of_features_idx, optimal_cluster_radius_idx] = find(grid_oa == max(grid_oa(:)));
optimal_number_of_features = number_of_features(optimal_number_of_features_idx);
optimal_cluster_radius = cluster_radius(optimal_cluster_radius_idx);

[optimal_number_of_features_error_idx, optimal_cluster_radius_error_idx] = find(grid_error == min(grid_error(:)));
optimal_number_of_features_error = number_of_features(optimal_number_of_features_error_idx);
optimal_cluster_radius_error = cluster_radius(optimal_cluster_radius_error_idx);

fprintf("Optimal model charactheristics from OA(features = %d, cluster_radius = %f)\n",optimal_number_of_features, optimal_cluster_radius);
fprintf("Optimal model charactheristics from mean error (features = %d, cluster_radius = %f)\n",optimal_number_of_features_error, optimal_cluster_radius_error);

% prepare data
train_optimal_x = train_x(:, idx(1:optimal_number_of_features));
train_optimal = [train_optimal_x train_y];
validate_optimal = [validate_x(:, idx(1:optimal_number_of_features)) validate_y];
test_optimal_x = [test_x(:, idx(:,1:optimal_number_of_features))];

[optimal_cluster_1, optimal_sigma_1] = subclust(train_optimal(train_y == 1, :), optimal_cluster_radius);
[optimal_cluster_2, optimal_sigma_2] = subclust(train_optimal(train_y == 2, :), optimal_cluster_radius);
[optimal_cluster_3, optimal_sigma_3] = subclust(train_optimal(train_y == 3, :), optimal_cluster_radius);
[optimal_cluster_4, optimal_sigma_4] = subclust(train_optimal(train_y == 4, :), optimal_cluster_radius);
[optimal_cluster_5, optimal_sigma_5] = subclust(train_optimal(train_y == 5, :), optimal_cluster_radius);
optimal_number_of_rules = size(optimal_cluster_1, 1) + size(optimal_cluster_2, 1) + size(optimal_cluster_3, 1) + size(optimal_cluster_4, 1) + size(optimal_cluster_5, 1);

optimal_fis = sugfis;
optimal_anfis_options = anfisOptions;
optimal_anfis_options.EpochNumber = 100;
optimal_anfis_options.ValidationData = validate_optimal;

% add input variables, and mf
 for i = 1:size(train_optimal_x, 2)
    name_in = sprintf("opt_in%d", i);
    optimal_fis = addInput(optimal_fis, [0,1], "Name", name_in);
    for j = 1:size(optimal_cluster_1, 1)    
        optimal_fis = addMF(optimal_fis, name_in, "gaussmf", [optimal_sigma_1(i) optimal_cluster_1(j, i)]);
    end
    for j = 1:size(optimal_cluster_2, 1)
        optimal_fis = addMF(optimal_fis, name_in, "gaussmf", [optimal_sigma_2(i) optimal_cluster_2(j, i)]);
    end
    for j = 1:size(optimal_cluster_3, 1)
        optimal_fis = addMF(optimal_fis, name_in, "gaussmf", [optimal_sigma_3(i) optimal_cluster_3(j, i)]);
    end
    for j = 1:size(optimal_cluster_4, 1)
        optimal_fis = addMF(optimal_fis, name_in, "gaussmf", [optimal_sigma_4(i) optimal_cluster_4(j, i)]);
    end
    for j = 1:size(optimal_cluster_5, 1)
        optimal_fis = addMF(optimal_fis, name_in, "gaussmf", [optimal_sigma_5(i) optimal_cluster_5(j, i)]);
    end
end
optimal_fis = addOutput(optimal_fis, [0,1], "Name", "opt_out");

% add output membership variables
optimal_parameters = [zeros(1, size(optimal_cluster_1, 1)) 0.25 * ones(1, size(optimal_cluster_2, 1)) 0.5 * ones(1, size(optimal_cluster_3, 1)) 0.75 * ones(1, size(optimal_cluster_4, 1)) ones(1, size(optimal_cluster_5, 1))];
for i = 1:optimal_number_of_rules
    optimal_fis = addMF(optimal_fis, "opt_out", 'constant', optimal_parameters(i));
end
% add FIS rules
rule_list = zeros(optimal_number_of_rules, size(train_optimal, 2));
for i = 1:size(rule_list, 1)
    rule_list(i, :) = i;
end
rule_list = [rule_list, ones(optimal_number_of_rules, 2)];
optimal_fis = addrule(optimal_fis, rule_list);

% train and evaluate optimal anfis
optimal_anfis_options.InitialFIS = optimal_fis;
[optimal_trained_fis, optimal_training_error, ~, optimal_validation_fis, optimal_validation_error] = anfis(train_optimal, optimal_anfis_options);

optimal_y_predicted = evalfis(optimal_validation_fis, test_optimal_x);
optimal_y_predicted = round(optimal_y_predicted);
optimal_y_predicted(optimal_y_predicted < 1) = 1;
optimal_y_predicted(optimal_y_predicted > 5) = 5;

optimal_error_matrix = confusionmat(optimal_y_predicted, test_y);
optimal_OA = trace(optimal_error_matrix) / sum(optimal_error_matrix,"all");
sum_actual = sum(optimal_error_matrix);
sum_predicted = sum(optimal_error_matrix');
optimal_PA = zeros(5, 1);
optimal_UA = zeros(5, 1);
for i = 1:5
    optimal_PA(i) = optimal_error_matrix(i,i)/sum_actual(i);
    optimal_UA(i) = optimal_error_matrix(i,i)/sum_predicted(i);
end

optimal_kHat = (sum(optimal_error_matrix, "all") * trace(optimal_error_matrix) - sum(sum_predicted .* sum_actual) ) / (sum(optimal_error_matrix, "all")^2 - sum(sum_predicted .* sum_actual) );

% display metrics
disp(optimal_error_matrix);
disp(optimal_OA);
disp(optimal_PA);
disp(optimal_UA);
disp(optimal_kHat);

% optimal model plots
% plot MF
figure("Name", "Optimal model MF (before training)");
hold on;
for l = 1:length(optimal_trained_fis.input)
   [xmf, ymf] = plotmf(optimal_fis, 'input', l);
   plot(xmf, ymf);
   xlabel('Input');
   ylabel('Degree of membership');
end
title("Optimal model MF (before training)");
saveas(gcf, [pwd '/optimal_model_mf_before_training.png']);

figure("Name", "Optimal model MF (after training)");
hold on;
for l = 1:length(optimal_validation_fis.input)
   [xmf, ymf] = plotmf(optimal_validation_fis, 'input', l);
   plot(xmf, ymf);
   xlabel('Input');
   ylabel('Degree of membership');
end
title("Optimal model MF (after training)");
saveas(gcf, [pwd '/optimal_model_mf_after_training.png']);

% learning curves
figure("Name", "Learning Curves");
plot([optimal_training_error, optimal_validation_error], "LineWidth", 2.5);
grid on;
legend("Training rrror", "Validation rrror");
xlabel("Epoch");
ylabel("Error");
title("Optimal model learning curves");
saveas(gcf, [pwd '/optimal_model_learning_curves.png']);

% predictions
figure("Name", "Predictions vs actual values");
hold on;
title('Predictions vs actual values');
xlabel('# Sample');
ylabel("Value");
plot(1:length(optimal_y_predicted), optimal_y_predicted, 'x','Color','red');
plot(1:length(optimal_y_predicted), test_y, 'o','Color','blue');
legend("Predictions", "Actual");
saveas(gcf, [pwd '/optimal_model_predictions.png']);

figure("Name", "Prediction errors");
title("Prediction Errors");
xlabel('# Sample');
ylabel('Error');
plot(1:length(optimal_y_predicted), optimal_y_predicted - test_y);
saveas(gcf, [pwd '/optimal_model_errors.png']);

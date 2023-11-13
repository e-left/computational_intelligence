% clear
clear; 
close all;
clc;

% load data
data = importdata('haberman.data');
% split data
[training_data, validation_data, testing_data] = split_dataset(data);

% construct xy data
[train_x, train_y] = extract_xy_data(training_data);
[validate_x, validate_y] = extract_xy_data(validation_data);
[test_x, test_y] = extract_xy_data(testing_data);

% create cell array to hold models
tsk_models = cell(4, 1);

OA = ones(4, 1);
PA = ones(4, 2);
UA = ones(4, 2);
kHat = ones(4, 1);
error_matrices = ones(4, 2, 2);
number_of_rules = ones(4, 1);

cluster_radius = [0.1, 1];

fis_options = genfisOptions('SubtractiveClustering');

anfis_options = anfisOptions;
anfis_options.EpochNumber = 50;
anfis_options.ValidationData = validation_data;

for iR = 1:length(cluster_radius)
    
    r = cluster_radius(iR);
    
    model_idx = [2 * (iR - 1) + 1, 2 * (iR - 1) + 2];
    

    % cluster per class
    [cluster_1, sigma_1]= subclust(training_data(train_y == 1, :), r);
    [cluster_2, sigma_2]= subclust(training_data(train_y == 2, :), r);
    
    num_of_rules = size(cluster_1, 1) + size(cluster_2, 1);

    % construct fis
    fis_cd = sugfis;

    % add inputs and output
    for k = 1:size(train_x, 2)
        name_in = "in" + int2str(k);
        fis_cd = addInput(fis_cd, [0,1], "Name", name_in);
    end
    fis_cd = addOutput(fis_cd, [0,1], "Name", "out1");

    % add input mf
    for k = 1:size(train_x, 2)
        var_name = "in" + int2str(k);
        for j = 1:size(cluster_1, 1)    
            fis_cd = addMF(fis_cd, var_name, "gaussmf", [sigma_1(k) cluster_1(j,k)]);
        end
        for j = 1:size(cluster_2, 1)
            fis_cd = addMF(fis_cd, var_name, "gaussmf", [sigma_2(k) cluster_2(j,k)]);
        end
    end

    % add output mf 
    params = [zeros(1, size(cluster_1, 1)) ones(1,size(cluster_2, 1))];
    for k = 1:num_of_rules
        fis_cd = addMF(fis_cd, "out1", 'constant', params(k));
    end

    % add rules
    list_of_rules = zeros(num_of_rules, size(training_data, 2));
    for k=1:size(list_of_rules,1)
        list_of_rules(k, :) = k;
    end
    
    list_of_rules = [list_of_rules, ones(num_of_rules, 2)];
    fis_cd = addrule(fis_cd, list_of_rules);

    % training
    anfis_options.InitialFIS = fis_cd;
    [trained_fis_cd, training_error_cd, ~, validation_fis_cd, validation_error_cd] = anfis(training_data, anfis_options);
    
    % plot learning curves
    figure("Name", sprintf("Class Dependent SC Learning Curves(r = %f)", r));
    plot([training_error_cd, validation_error_cd], 'LineWidth',2.5); 
    grid on;
    legend('Training error', 'Validation error');
    xlabel("Epoch");
    ylabel("Error");
    title(sprintf("Class Dependent SC Learning Curves(r = %f)", r));
    saveas(gcf, [pwd sprintf('/plain_cd_learning_curves_r_%f.png', r)]);

    y_pred_cd = evalfis(validation_fis_cd, test_x);
    y_pred_cd = round(y_pred_cd);
    y_pred_cd = min(max(1,y_pred_cd), 2);

    % plot membership functions
    figure("Name", sprintf("Class Dependent SC Membership functions(r = %f)", r))
    for k=1:size(train_x, 2)
        plotmf(trained_fis_cd, "input", k);
        hold on;
    end
    title(sprintf("Class Dependent SC Membership functions(r = %f)", r));
    saveas(gcf, [pwd sprintf('/plain_cd_mf_r_%f.png', r)]);

    
    error_matrices(model_idx(1), :, :) = confusionmat(test_y, y_pred_cd);
    
    n = sum( error_matrices(model_idx(1), :, :), "all");

    TP = error_matrices(model_idx(1), 1, 1);
    FP = error_matrices(model_idx(1), 1, 2);
    TN = error_matrices(model_idx(1), 2, 2);
    FN = error_matrices(model_idx(1), 2, 1);

    OA(model_idx(1)) = (TP + TN) / (TP + FP + TN + FN);
    PA(model_idx(1),1) = TP / (FN + TP);
    PA(model_idx(1),2) = TN / (TN + FP);
    UA(model_idx(1),1) = TP / (TP + FP);
    UA(model_idx(1),2) = TN / (FN + TN);
    
    m = ((TP + FP) * (TP + FN) + (TN + FN) * (FP + TN));
    kHat(model_idx(1)) = (n * (TP + TN) - m) / (n^2 - m);

    number_of_rules(model_idx(1)) = size(validation_fis_cd.Rules, 2);

    % class independent SP
    
    fis_options.ClusterInfluenceRange = r;
    fis_ci = genfis(train_x, train_y, fis_options);
    
    anfis_options.InitialFIS = fis_ci;
    [trained_fis_ci, training_error_ci, ~, validation_fis_ci, validation_error_ci] = anfis(training_data, anfis_options);
    
    % plot learning curve
    figure("Name", sprintf("Class Independent SC Learning Curves(r = %f)", r));    
    plot([training_error_ci validation_error_ci], 'LineWidth', 2.5);
    grid on;
    legend("Training Error", "Validation Error");
    xlabel("Epoch");
    ylabel("Error");
    title(sprintf("Class Independent SC Learning Curves(r = %f)", r));
    saveas(gcf, [pwd sprintf('/plain_ci_learning_curves_r_%f.png', r)]);


    y_pred_ci = evalfis(validation_fis_ci, test_x);
    y_pred_ci = round(y_pred_ci);
    y_pred_ci = min(max(1, y_pred_ci), 2);

    % plot membership functions
    figure("Name", sprintf("Class Independent SC Membership functions(r = %f)", r))
    for k=1:size(train_x, 2)
        plotmf(trained_fis_ci, "input", k);
        hold on;
    end
    title(sprintf("Class Independent SC Membership functions(r = %f)", r));
    saveas(gcf, [pwd sprintf('/plain_ci_mf_r_%f.png', r)]);


    error_matrices(model_idx(2), :, :) = confusionmat(test_y, y_pred_ci);
    n = sum(error_matrices(model_idx(2), :, :), "all");
    TP = error_matrices(model_idx(2), 1, 1);
    FP = error_matrices(model_idx(2), 1, 2);
    TN = error_matrices(model_idx(2), 2, 2);
    FN = error_matrices(model_idx(2), 2, 1);

    OA(model_idx(2)) = (TP + TN) / (TP + FP + TN + FN);
    PA(model_idx(2), 1) = TP/(FN + TP);
    PA(model_idx(2), 2) = TN/(TN + FP);
    UA(model_idx(2), 1) = TP/(TP + FP);
    UA(model_idx(2), 2) = TN/(FN + TN);
    
    m = ((TP + FP) * (TP + FN) + (TN + FN) * (FP + TN));
    kHat(model_idx(2)) = (n * (TP + TN) - m) / (n^2 - m);

    number_of_rules(model_idx(2)) = size(validation_fis_ci.Rules, 2);
end

for k = 1:4
    fprintf("Displaying metrics for model %d\n", k);
    disp(OA(k, :));
    disp(PA(k, :));
    disp(UA(k, :));
    disp(kHat(k, :));
    disp(error_matrices(k, :, :));
    disp(number_of_rules(k, :));
end
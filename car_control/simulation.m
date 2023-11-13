fuzzyfff% clear workspace
clc; 
clear;
close all;

% read system
% choose between normal and optimized system
% sys_name = "car_system";
sys_name = "optimized_car_system";
car_fuzzy_system = readfis(sys_name);

% set up initial values
x_init = 3.8;
y_init = 0.5;
theta_init = [0, 45, -45];
u = 0.05;
x_final = 10;
y_final = 3.2;

% epsilon for going close
epsilon = 0.1;

% for each starting theta value
for k = 1:3
    % set up arrays to hold x and y values
    x = [];
    y = [];
    x(end + 1) = x_init;
    y(end + 1) = y_init;

    % get initial theta value
    theta = [];
    theta(end + 1) = theta_init(k);
    
    % keep all delta theta values
    delta_theta = [];

    % perform simulation
    finished = false;

    % status variable
    % 0 means successful
    % 1 means crashed
    % 2 means out of bounds
    finished_status = 0;
    while ~finished
        % check if out of bounds
        if x(end) < 0 || y(end) < 0 || x(end) > 10.5 || y(end) > 4.5
            finished = true;
            finished_status = 2;
            continue;
        end
        % get obstacle distances
        [dh, dv] = get_obstacle_dists([x(end), y(end)]);
        % if any is negative, collided with boundaries
        % aka crashed
        if dh < 0 || dv < 0
            finished = true;
            finished_status = 1;
            continue;
        end

        % compute new angle
        delta_theta_t = evalfis(car_fuzzy_system, [dv, dh, theta(end)]);
        theta_new = theta(end) + delta_theta_t;
        theta(end + 1) = theta_new;
        delta_theta(end + 1) = delta_theta_t;
        
        % compute new x and y coordinates
        x_new = x(end) + u * cos(theta(end) * pi / 180.0);
        y_new = y(end) + u * sin(theta(end) * pi / 180.0);
        
        % add them to list
        x(end + 1) = x_new;
        y(end + 1) = y_new;

        dist_from_final = sqrt((x(end) - x_final)^2 + (y(end) - y_final)^2);
        if dist_from_final < epsilon
            finished = true;
        end
    end
    % display result according to status
    fprintf("Finished simulation with theta_0 = %f\n", theta(1));
    if finished_status == 0
        fprintf("Reached destination\n");
    elseif finished_status == 1
        fprintf("Crashed into wall\n");
    elseif finished_status == 2
        fprintf("Out of bounds\n")
    end

    % plot result
    plot_name = sprintf("Simulation, theta_0 = %f", theta(1));
    figure("Name", plot_name)
    xlim([0, 10]);
    ylim([0, 4]);
    title(plot_name);
    hold on
    % plot points
    plot_color = "green";
    if finished_status == 1
        plot_color = "red";
    elseif finished_status == 2
        plot_color = "magenta";
    end
    plot(x, y, "Color", plot_color);
    plot(x_init, y_init, "x", "Color", "yellow");
    plot(x_final, y_final, "x", "Color", "blue");
    % plot obstacles
    obst_area = area([5, 5, 6, 6, 7, 7 ,10], [0, 1, 1, 2, 2, 3, 3,]);
    set(obst_area, 'FaceColor', [0.4 0.4 0.4]);
    saveas(gcf, [pwd, sprintf('/car_theta_%d.png', theta(1))]);
end



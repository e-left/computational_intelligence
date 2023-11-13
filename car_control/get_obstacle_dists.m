function [dh, dv] = get_obstacle_dists(v)
    % pass in a vector v = [x, y] and get distances from obstacles
    x = v(1);
    y = v(2);

    dh = 0;
    dv = 0;
  
    % firstly determine dh
    % based on y distance and x
    if y <= 1 
        % obstacle at x = 5
        dh = 5 - x;
    elseif y <= 2
        % obstacle at x = 6
        dh = 6 - x;
    elseif y <= 3
        % obstacle at x = 7 
        dh = 7 - x;
    elseif y > 3
        % bounds at 10 plus sth to assume we are okay
        dh = 1 + 10 - x;
    end

    % determine dv
    % based on x distance and y
    if x <= 5
        % dv is y (obstacle at 0)
        dv = y;
    elseif x <= 6
        % obstacle at y = 1
        dv = y - 1;
    elseif x <= 7 
        % obstacle at y = 2
        dv = y - 2;
    elseif x > 7
        % obstacle at y = 3
        dv = y - 3;
    end

    dh = max(0, min(dh, 1));
    dv = max(0, min(dv, 1));
end


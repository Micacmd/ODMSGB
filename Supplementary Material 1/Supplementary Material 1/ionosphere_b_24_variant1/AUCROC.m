% Evaluation outlier detection by ROC and AUC for multiple algorithms
clear all
clc

% Load the dataset
load('ionosphere_b_24_variant1.mat'); 

load('ionosphere_b_24_variant1_idea.mat'); 

% Features and labels
data = trandata(:, 1:end-1); % Features
labels = trandata(:, end);   % Labels

% List of algorithms
algorithms = {'ODMSGB'};
colors = [
    1 0 0;   % Red
]; % Line styles for different algorithms

figure;
hold on;

% Loop over algorithms
for i = 1:length(algorithms)
    % Generate outlier scores using the corresponding algorithm
    switch algorithms{i}
        case 'ODMSGB'
            out_scores = idea_out_scores; 
        otherwise
            error(['Unknown algorithm: ', algorithms{i}]);
    end
    
    % Compute ROC
    [FPR, TPR, ~, AUC] = perfcurve(labels, out_scores, 1);
    disp(['Algorithm: ', algorithms{i}, ', AUC = ', num2str(AUC)]);
    
    % Plot ROC curve
    plot(FPR * 100, TPR * 100, 'Color', colors(i,:),'LineWidth', 1.2,  'DisplayName', algorithms{i}); % 使用 RGB 格式

end

% Customize plot
xlabel('FPR(ν) (%)');
ylabel('TPR(ν) (%)');
legend('Location', 'southeast');
grid off;
xlim([-2, 50]);
ylim([-2, 102]);

% Add secondary axes to simulate axes on all four sides
ax = gca; % Get current axes
ax.Box = 'on'; % Turn on box to enclose axes
ax.XColor = 'k'; % Set x-axis color
ax.YColor = 'k'; % Set y-axis color

% Set the position of the axes so that ticks appear on all sides
ax.TickDir = 'out'; % Make ticks point outward
ax.XAxisLocation = 'bottom'; % Keep x-axis at the bottom
ax.YAxisLocation = 'left'; % Keep y-axis at the left

hold off;
%saveas(gcf, 'ionosphere_b_24_variant1.png');

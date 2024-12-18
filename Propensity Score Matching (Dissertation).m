
data = DataFinalAdj;

% Ever_Treated Variable
ID = data.ID;
Year = data.Year;
S_TR = data.S_TR;
Ever_Treated = zeros(size(ID));
unique_ids = unique(ID);
for i = 1:length(unique_ids)
    idx = ID == unique_ids(i);
    if any(S_TR(idx) == 1)
        Ever_Treated(idx) = 1;
    end
end
data.Ever_Treated = Ever_Treated;

% Treatment Timing
treatment_years = NaN(size(ID));
for i = 1:length(unique_ids)
    idx = ID == unique_ids(i) & Ever_Treated == 1;
    if any(idx)
        first_treatment_year = min(Year(idx & S_TR == 1));
        treatment_years(idx) = first_treatment_year;
    end
end
data.Treatment_Year = treatment_years;

% Select Relevant Data 
treated_data = data(data.Ever_Treated == 1 & data.Year == data.Treatment_Year - 1, :);
control_data = data(data.Ever_Treated == 0 & data.SPD == 0, :);  
selected_data = [treated_data; control_data];

continuous_vars = {'MC', 'ROA', 'DIR', 'IO', 'W', 'NED'};
industry_dummies = {'SS', 'THE', 'SC', 'CDD', 'ME', 'PB', 'B', 'AC', 'FS', 'E', 'HCE', 'HPP', 'CSD', 'FBT', 'CS', 'CG', 'TS', 'UTI', 'TR', 'INS', 'CDA', 'EREI', 'CPS', 'MAT', 'REMD'};

reference_category = 'SS';
industry_dummies = setdiff(industry_dummies, reference_category);

all_vars = [continuous_vars, industry_dummies];

X = table2array(selected_data(:, all_vars));
y = selected_data.Ever_Treated;

% Fit Logistic Regression Model
mdl = fitglm(X, y, 'Distribution', 'binomial', 'Link', 'logit');
% Extract and display p-values
coef_table = mdl.Coefficients;
p_values = coef_table.pValue;

disp('P-values for logistic regression');
for i = 1:length(all_vars)
    fprintf('%s: p-value = %.4f\n', all_vars{i}, p_values(i+1)); % i+1 to skip the intercept
end

% model p-value
chi2_statistic = mdl.Deviance - mdl.DFE;
overall_p_value = 1 - chi2cdf(chi2_statistic, mdl.NumCoefficients - 1);
fprintf('Overall model p-value: %.4f\n', overall_p_value);

%  Propensity Scores
X_pred = table2array(data(:, all_vars));
propensity_scores = predict(mdl, X_pred);

% Add Propensity Scores to dataset
data.PropensityScore = propensity_scores;

% Nearest Neighbor Matching
[matched_data, matched_indices] = nearest_neighbor_matching(data, 'PropensityScore', 'Ever_Treated', 'ID', 'EPD');

% Create Matched Dataset with Two Columns (Treated and Control)
treated_ids = matched_data.ID(matched_data.Ever_Treated == 1);
matched_control_ids = matched_data.ID(ismember(matched_data.ID, matched_indices));
num_pairs = min(length(treated_ids), length(matched_control_ids));
matched_dataset = table();
matched_dataset.Treated_ID = treated_ids(1:num_pairs);
matched_dataset.Control_ID = matched_control_ids(1:num_pairs);

% Write matched dataset to an Excel file
%writetable(matched_dataset, 'matched_id_social.xlsx');

% Multicollinearity check
X = table2array(matched_data(:, all_vars));
correlation_matrix = corr(X);
vif = diag(inv(correlation_matrix))';


disp('Variance Inflation Factors:');
for i = 1:length(all_vars)
    fprintf('%s: VIF = %.2f\n', all_vars{i}, vif(i));
end

% Nearest Neighbor Matching Function
function [matched_data, matched_indices] = nearest_neighbor_matching(data, ps_col, treat_col, id_col, epd_col, caliper)
    if nargin < 6
        caliper = 0.2 * std(logit(data.(ps_col)));  
    end

    % Separate treated and conttol groups
    treated = data(data.(treat_col) == 1, :);
    control = data(data.(treat_col) == 0 & data.(epd_col) == 0, :);

    n_treated = height(treated);
    matched_indices = zeros(n_treated, 1);

    for i = 1:n_treated
        distances = abs(treated.(ps_col)(i) - control.(ps_col));
        [min_dist, min_idx] = min(distances);

        if min_dist <= caliper
            matched_indices(i) = control.(id_col)(min_idx);
            control(min_idx, :) = [];  % Remove the matched control to prevent reuse
        end
    end

    % Remove unmatched treated units
    matched_indices(matched_indices == 0) = [];

    % Matched dataset
    matched_treated = treated;
    matched_control = data(ismember(data.(id_col), matched_indices), :);
    matched_data = [matched_treated; matched_control];
end

% SMD 
function check_balance(original_data, matched_data, vars_to_check, treat_col)
    disp('Balance Before Matching:');
    calculate_smd(original_data, vars_to_check, treat_col);
    
    disp('Balance After Matching:');
    calculate_smd(matched_data, vars_to_check, treat_col);
end

function calculate_smd(data, vars, treat_col)
    for i = 1:length(vars)
        treated = data.(vars{i})(data.(treat_col) == 1);
        control = data.(vars{i})(data.(treat_col) == 0);
        
        diff_means = mean(treated) - mean(control);
        pooled_sd = sqrt((var(treated) + var(control)) / 2);
        smd = abs(diff_means / pooled_sd);
        
        fprintf('%s: SMD = %.4f\n', vars{i}, smd);
    end
end

function y = logit(p)
    y = log(p ./ (1 - p));
end

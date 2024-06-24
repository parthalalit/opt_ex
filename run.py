

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit, minimize


from michalis import michaelis_menten, curve_example
from curve_ex import curve_examplee
from data_gen import data_generator
from train import train_slearner
from treatement_effect import extract_treated_effect
from untreated_effect import extract_untreated_effect
from response_curve import response_curves
from objective_func import objective_function


linear_x_1, nonlinear_y_1 = curve_example(10000, 1000)
linear_x_2, nonlinear_y_2 = curve_example(10000, 500)
linear_x_3, nonlinear_y_3 = curve_example(10000, 250)
linear_x_4, nonlinear_y_4 = curve_example(5000, 1000)
linear_x_5, nonlinear_y_5 = curve_example(5000, 500)
linear_x_6, nonlinear_y_6 = curve_example(5000, 250)
linear_x_7, nonlinear_y_7 = curve_example(2500, 1000)
linear_x_8, nonlinear_y_8 = curve_example(2500, 500)
linear_x_9, nonlinear_y_9 = curve_example(2500, 250)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot scatter plots on each subplot
sns.lineplot(x=linear_x_1, y=nonlinear_y_1, ax=axes[0], label='lamda=1000')
sns.lineplot(x=linear_x_2, y=nonlinear_y_2, ax=axes[0], label='lamda=500')
sns.lineplot(x=linear_x_3, y=nonlinear_y_3, ax=axes[0], label='lamda=250')
axes[0].set_title('alpha=10000')
axes[0].set_xlabel('Count of Orders')
axes[0].set_ylabel('Discount Amount')

sns.lineplot(x=linear_x_4, y=nonlinear_y_4, ax=axes[1], label='lamda=1000')
sns.lineplot(x=linear_x_5, y=nonlinear_y_5, ax=axes[1], label='lamda=500')
sns.lineplot(x=linear_x_6, y=nonlinear_y_6, ax=axes[1], label='lamda=250')
axes[1].set_title('alpha=5000')
axes[1].set_xlabel('Count of Orders')
axes[1].set_ylabel('Discount Amount')

sns.lineplot(x=linear_x_7, y=nonlinear_y_7, ax=axes[2], label='lamda=1000')
sns.lineplot(x=linear_x_8, y=nonlinear_y_8, ax=axes[2], label='lamda=500')
sns.lineplot(x=linear_x_9, y=nonlinear_y_9, ax=axes[2], label='lamda=250')
axes[2].set_title('alpha=2500')
axes[2].set_xlabel('Count of Orders')
axes[2].set_ylabel('Discount Amount')

# Add labels to the entire figure
fig.suptitle('Price Discount Effect')


plt.savefig('D:\opt_ex\plots\price_discount_effect.png')


# # Data generating process

np.random.seed(1234)

n=100000

y_train_1, X_train_1, T_mm_1, tau_1 = data_generator(n, 1.00, 2, 5000)
y_train_2, X_train_2, T_mm_2, tau_2 = data_generator(n, 0.25, 2, 5000)
y_train_3, X_train_3, T_mm_3, tau_3 = data_generator(n, 2.00, 2, 5000)


# # S-Learner
np.random.seed(1234)

model_1, yhat_train_1 = train_slearner(X_train_1, y_train_1)
model_2, yhat_train_2 = train_slearner(X_train_2, y_train_2)
model_3, yhat_train_3 = train_slearner(X_train_3, y_train_3)


# Create a figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot scatter plots on each subplot
sns.scatterplot(x=yhat_train_1, y=y_train_1, ax=axes[0])
axes[0].set_title('Product 1')
axes[0].set_xlabel('Estimated Orders')
axes[0].set_ylabel('Actual Orders')

sns.scatterplot(x=yhat_train_2, y=y_train_2, ax=axes[1])
axes[1].set_title('Product 2')
axes[1].set_xlabel('Estimated Orders')
axes[1].set_ylabel('Actual Orders')

sns.scatterplot(x=yhat_train_3, y=y_train_3, ax=axes[2])
axes[2].set_title('Product 3')
axes[2].set_xlabel('Estimated Orders')
axes[2].set_ylabel('Actual Orders')


fig.suptitle('Actual vs Estimated')

# Show plots
plt.savefig('D://opt_ex//plots//actuval_estimated.png')


# # Extracting treatment effect

treated_1, df_scoring_1 = extract_treated_effect(n, X_train_1, model_1)
treated_2, df_scoring_2 = extract_treated_effect(n, X_train_2, model_2)
treated_3, df_scoring_3 = extract_treated_effect(n, X_train_3, model_3)

#
untreated_1 = extract_untreated_effect(n, X_train_1, model_1)
untreated_2 = extract_untreated_effect(n, X_train_2, model_2)
untreated_3 = extract_untreated_effect(n, X_train_3, model_3)

#
# Calculate treatment effect
treatment_effect_1 = treated_1 - untreated_1
treatment_effect_2 = treated_2 - untreated_2
treatment_effect_3 = treated_3 - untreated_3


# Create a figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot scatter plots on each subplot
sns.scatterplot(x=df_scoring_1['T'], y=treatment_effect_1, ax=axes[0], label='estimated')
sns.lineplot(x=df_scoring_1['T'], y=T_mm_1, ax=axes[0], color='green', label='actual')
axes[0].set_title('Product 1')
axes[0].set_xlabel('Discount')
axes[0].set_ylabel('Treatment Effect')

sns.scatterplot(x=df_scoring_2['T'], y=treatment_effect_2, ax=axes[1], label='estimated')
sns.lineplot(x=df_scoring_2['T'], y=T_mm_2, ax=axes[1], color='green', label='actual')
axes[1].set_title('Product 2')
axes[1].set_xlabel('Discount')
axes[1].set_ylabel('Treatment Effect')

sns.scatterplot(x=df_scoring_3['T'], y=treatment_effect_3, ax=axes[2], label='estimated')
sns.lineplot(x=df_scoring_3['T'], y=T_mm_3, ax=axes[2], color='green', label='actual')
axes[2].set_title('Product 3')
axes[2].set_xlabel('Discount')
axes[2].set_ylabel('Treatment Effect')

fig.suptitle('Discount Treatment Effect')
plt.savefig('D://opt_ex//plots//discount_treatment_effect.png')


# # Curve fit

popt_1, pcov_1 = response_curves(treatment_effect_1, df_scoring_1)
popt_2, pcov_2 = response_curves(treatment_effect_2, df_scoring_2)
popt_3, pcov_3 = response_curves(treatment_effect_3, df_scoring_3)

# Calculate response curves
treatment_effect_curve_1 = michaelis_menten(df_scoring_1['T'], popt_1[0], popt_1[1])
treatment_effect_curve_2 = michaelis_menten(df_scoring_2['T'], popt_2[0], popt_2[1])
treatment_effect_curve_3 = michaelis_menten(df_scoring_3['T'], popt_3[0], popt_3[1])

#
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot scatter plots on each subplot
sns.scatterplot(x=df_scoring_1['T'], y=treatment_effect_1, ax=axes[0], label='estimated')
sns.lineplot(x=df_scoring_1['T'], y=treatment_effect_curve_1, ax=axes[0], color='green', label='response curve')
axes[0].set_title('Product 1')
axes[0].set_xlabel('Discount')
axes[0].set_ylabel('Treatment Effect')

sns.scatterplot(x=df_scoring_2['T'], y=treatment_effect_2, ax=axes[1], label='estimated')
sns.lineplot(x=df_scoring_2['T'], y=treatment_effect_curve_2, ax=axes[1], color='green', label='response curve')
axes[1].set_title('Product 2')
axes[1].set_xlabel('Discount')
axes[1].set_ylabel('Treatment Effect')

sns.scatterplot(x=df_scoring_3['T'], y=treatment_effect_3, ax=axes[2], label='estimated')
sns.lineplot(x=df_scoring_3['T'], y=treatment_effect_curve_3, ax=axes[2], color='green', label='response curve')
axes[2].set_title('Product 3')
axes[2].set_xlabel('Discount')
axes[2].set_ylabel('Treatment Effect')


fig.suptitle('Discount Treatment Effect')
plt.savefig('D://opt_ex//plots//discount_treatment_effect_curve.png')

# # Optimisation


# List of products
products = ["product_1", "product_2", "product_3"]

# Set total budget to be the sum of the mean of each product reduced by 20%
total_budget = (df_scoring_1['T'].mean() + df_scoring_2['T'].mean() + df_scoring_3['T'].mean()) * 0.80

# Dictionary with min and max bounds for each product - set as +/-20% of max/min discount
budget_ranges = {"product_1": [df_scoring_1['T'].min() * 0.80, df_scoring_1['T'].max() * 1.2], 
                 "product_2": [df_scoring_2['T'].min() * 0.80, df_scoring_2['T'].max() * 1.2], 
                 "product_3": [df_scoring_3['T'].min() * 0.80, df_scoring_3['T'].max() * 1.2]}

# Dictionary with response curve parameters
parameters = {"product_1": [popt_1[0], popt_1[1]], 
              "product_2": [popt_2[0], popt_2[1]], 
              "product_3": [popt_3[0], popt_3[1]]}


# Set initial guess by equally sharing out the total budget
initial_guess = [total_budget // len(products)] * len(products)

# Set the lower and upper bounds for each product
bounds = [budget_ranges[product] for product in products]

# Set the equality constraint - constraining the total budget
constraints = {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}

# Run optimisation
result = minimize(
    lambda x: objective_function(x, products, parameters),
    initial_guess,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={'disp': True, 'maxiter': 1000, 'ftol': 1e-9},
)

# Extract results
optimal_treatment = {product: budget for product, budget in zip(products, result.x)}
print(f'Optimal promo budget allocations: {optimal_treatment}')
print(f'Optimal orders: {round(result.fun * -1, 2)}')



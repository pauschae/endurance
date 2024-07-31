import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model.decisions import simulate_dgp, calc_marginal_likelihood
import estimagic as em
 

# Define Parameters
params = {
    "value": [0, 1, 2, 1, 2, 1, 0.5, 5],
    "lower_bound": [-np.inf, 0, 0, 0, 0, 0, 0, 0]
}

index_names = ["alpha_mean", "alpha_variance", "sigma_shape", "sigma_scale", 
               "beta_shape", "beta_scale", "k_alpha", "k_beta"]

params = pd.DataFrame(data=params, index=index_names)

# Simulate the DGP
params_questions, params_individuals, simulated_data = simulate_dgp(params, 100, 500)

# Merge parameters with the simulated data
df_with_params = pd.merge(simulated_data, params_questions, on='question_id')
df_with_params = pd.merge(df_with_params, params_individuals, on='individual_id')
df_with_params["answered_correctly"] = df_with_params.answer == df_with_params.correct_answer

# Look at the Distributions of Parameters
# k
sns.histplot(params_individuals['k'], kde=False, bins=10)
plt.xlabel('k values')
plt.ylabel('Frequency')
plt.show()

# beta
sns.histplot(params_individuals['beta'], kde=False, bins=10)
plt.xlabel('beta values')
plt.ylabel('Frequency')
plt.show()

# alpha
sns.histplot(params_questions['alpha1'], kde=False, bins=10)
plt.xlabel('alpha values')
plt.ylabel('Frequency')
plt.show()

# First plot: Probability to answer correctly over time
sns.regplot(data=df_with_params, x="t", y="answered_correctly", lowess=True)
plt.xlabel('Time (t)')
plt.ylabel('Probability of Answering Correctly')
plt.title('Probability to Answer Correctly Over Time')
plt.show()

# Second plot: Remaining endurance over time
sns.regplot(data=df_with_params, x="t", y="remaining_endurance", x_bins=np.arange(0, 100, 10), order=2)
plt.xlabel('Time (t)')
plt.ylabel('Remaining Endurance')
plt.title('Remaining Endurance Over Time')
plt.show()

#Estimate the Model From Simulated Data
start_params = params.assign(value=8*[0.1])
my_kwargs = {"data_questions":params_questions,
                "data_individuals":simulated_data,
                "n_draws":1000}

# Sort individual level data by question id to facilitate question level analysis
data_individuals = simulated_data.sort_values(by=['question_id', 'individual_id'], ascending=True)
  
calc_marginal_likelihood(start_params,  **my_kwargs)

res = em.estimate_ml(
        loglike=calc_marginal_likelihood,
        params=start_params,
        optimize_options={"algorithm": "scipy_lbfgsb"},
        loglike_kwargs=my_kwargs,
        logging = "mylog.db"
    )

res.summary().round(3)
fig = em.criterion_plot("mylog.db")
fig.show()
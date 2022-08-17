---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region slideshow={"slide_type": "slide"} -->
# Lesson Introduction: GP From Fundamentals
Building a GP one piece at a time
<!-- #endregion -->

## Current holdups/questions
* `f_star = gp.conditional("f_star", np.array([.2])[:, None])` is causing an exception I'm not sure why
* In the second example below without pymc, if the data is generated from the kernel how is it observed?
  * Other than that its really neat to see the GP manually built from a mean function and covariance

```python slideshow={"slide_type": "skip"}
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc.sampling_jax
import scipy.stats as stats

plt.style.use('intuitivebayes.mplstyle')
figsize = (14,7)  
```

```python
# Start with simulated sine wave 
np.random.seed(1)
x_axis = np.linspace(-4,4, 20)


sigma = .1
# Copy the x_vals a couple of times to get multiple points per x_val
x_vals = np.tile(x_axis, 3)
noise = stats.norm(0, sigma).rvs(x_vals.shape)

# Repeat the data a couple of times
y_obs = np.sin(x_vals) + noise
```

# Fitting and predicting a point

```python
figsize = (14,7)  
fig, ax = plt.subplots(figsize=figsize)
ax.scatter(x_vals, y_obs)

index = 10
ax.axvline(x_axis[index], linestyle='--')
ax.scatter(x_axis[index], y_obs[index]);
```

In this lesson we're going to make a prediction at a specific x value of interest, x prime as its typically called, just like we did in the last lesson

But now were going to do two things differently


## Our GP

```python
X = x_vals[:,None]
x_prediction = .2

with pm.Model() as latent_gp_model:
    # Specify the covariance function.
    cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)

    # Specify the GP.  The default mean function is `Zero`.
    gp = pm.gp.Latent(cov_func=cov_func)
    
    # Place a GP prior over the function f.
    f = gp.prior("f", X=X)
    
    # TODO: This line is causing an aesara exception
    f_star = gp.conditional("f_star", np.array([.2])[:, None])

    obs = pm.Normal("lik", mu=f, sigma=sigma, observed=y_obs)
    
    # TODO: Can we change these var names to be more intuitive

    trace = pm.sample(1000, chains=2, return_inferencedata=True)
    
    pred_samples = pm.sample_posterior_predictive(trace.posterior, var_names=["f_star"])
```

One, We're going to do it using a GP. And 2 were going to show you exactly how we're going to do it in great detail. Even more detail than you're seeing here

```python
pred_samples.posterior_predictive.values
```

```python
vals = pred_samples.posterior_predictive.f_star.values.squeeze()
```

Question for Bill: Posterior Predictive looks very wide. Am I accidentally sampling from the prior?

```python
figsize = (14,7)  
fig, ax = plt.subplots(figsize=figsize)

ax.scatter(x_vals, y)
ax.scatter(np.full(vals.shape, x_prediction), vals, alpha=.4);
```
## Test Section
Temporary questions for bill
* If Y is randomly generated how is it observed/

```python
n = 200  # The number of data points
X = np.linspace(0, 10, n)[:, None]  # The inputs to the GP must be arranged as a column vector

# Define the true covariance function and its parameters
ℓ_true = 1.0
η_true = 3.0

## TODO IB: Construct this manually using numpy ad arrays
cov_func = η_true ** 2 * pm.gp.cov.Matern52(1, ℓ_true)

# A mean function that is zero everywhere
# TODO: Construct this manually using np.tile or something like that
mean_func = pm.gp.mean.Zero()

# The latent function values are one sample from a multivariate normal
# Note that we have to call `eval()` because PyMC3 built on top of Theano
f_true = np.random.multivariate_normal(
    mean_func(X).eval(), cov_func(X).eval() + 1e-8 * np.eye(n), 1
).flatten()

# The observed data is the latent function plus a small amount of T distributed noise
# The standard deviation of the noise is `sigma`, and the degrees of freedom is `nu`
σ_true = 2.0
ν_true = 3.0
y = f_true # + σ_true * np.random.standard_t(ν_true, size=n)

## Plot the data and the unobserved latent function
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
ax.plot(X, f_true, "dodgerblue", lw=3, label="True generating function 'f'")
ax.plot(X, y, "ok", ms=3, label="Observed data")
ax.set_xlabel("X")
ax.set_ylabel("y")
plt.legend();
```

```python
cov_func(X).eval()
```




<!-- #region slideshow={"slide_type": "slide"} -->
## Section 10: Multivariate normals
* The distribution that underlies it all
* How covariance
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Section 20: ?
<!-- #endregion -->

## Section 30: ?


## Section 40: 


# Section 10: Multivariate normals





## Section Recap
* Normal or Gaussian Distributions 
* Covariance is the relationship between the value on one dimension with another


# Section 20: 


## Section Recap


# Section 30:

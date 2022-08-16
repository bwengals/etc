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
x = np.linspace(-4,4, 20)
noise = stats.norm(0, .1).rvs(x.shape)
y = np.sin(x) + noise
```

# Fitting and predicting a point

```python
figsize = (14,7)  
fig, ax = plt.subplots(figsize=figsize)
ax.scatter(x,y)

index = 10
ax.axvline(x[index], linestyle='--')
ax.scatter(x[index],y[index]);
```

In this lesson we're going to make a prediction at a specific x value of interest, x prime as its typically called, just like we did in the last lesson

But now were going to do two things differently


## Our GP

```python
X = x[:,None]

x_prediction = .2
with pm.Model() as latent_gp_model:
    # Specify the covariance function.
    cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)

    # Specify the GP.  The default mean function is `Zero`.
    gp = pm.gp.Latent(cov_func=cov_func)
    
    # Place a GP prior over the function f.
    f = gp.prior("f", X=X)
    
    # TODO: Can we change these car names to be more intuitive
    f_star = gp.conditional("f_star", np.array([[.2]]))
    trace = pm.sample(1000, chains=2, return_inferencedata=True)
    
    pred_samples = pm.sample_posterior_predictive(trace.posterior, var_names=["f_star"])
```

One, We're going to do it using a GP. And 2 were going to show you exactly how we're going to do it in great detail. Even more detail than you're seeing here

```python
pred_samples.posterior_predictive.values
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

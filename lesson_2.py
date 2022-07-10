# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

# %%
import pymc.sampling_jax

# %% [markdown] slideshow={"slide_type": "slide"}
# # Lesson Introduction: Art Class
# Build intuition through creativity

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Section 10: How to a draw a line
# * Showcase our challenge: Modeling C02 data
# * Draw a line using linear regression
# * Draw a couplle more lines with Bayesian Linear Regression

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Section 20: How to draw a wiggly line
# * Discuss where plain linear regression misses
# * How we can improve our model with additional term

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Section 30: A different method for drawing

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Section 40: Placeholder

# %% [markdown] slideshow={"slide_type": "slide"}
# # Section 10: How to a draw a line
# The first model that everyone tries (probably)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Mauna Loa 
# <center>
#   <img src="img/Aa_channel_flow_from_Mauna_Loa.jpg"  />
# </center>

# %% [markdown] slideshow={"slide_type": "notes"}
# Lets first talk about Hawaii, specifically Mauna Loa, which is quite beautiful as you can see here
#
# On this island there is a research facility where they conduct experiments.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Mauna Loa CO2 Experiment
#
# <center>
#     <div>
#        <img src="img/9oyu_2ghf_141031.jpg" height="400" />
#     </div>
# </center>

# %% [markdown] slideshow={"slide_type": "notes"}
# - Been measuring C02 for the last X years
# - A key experiment in the study of climate change
# - Longest running continuous collection of atmospheric CO2 measurements
#

# %% [markdown] slideshow={"slide_type": "slide"}
# #  Mauna Loa CO2 dataset
#

# %% slideshow={"slide_type": "-"}
data = pd.read_csv("data/co2_mm_mlo.csv", header=51)

# %%
plt.figure(figsize=(14,6))
plt.plot(data["decimal date"], data["average"], "k.")
plt.title("C02 measures")
plt.xlabel("Year")
plt.ylabel("C02 Measurement");

# %% [markdown] slideshow={"slide_type": "notes"}
# Here's the time series, theres a number of things we can note immediately. Is the trend, at a gance it looks roughly linear but perhaps not quite, it also has a seasonal pattern. Specifically it has some sort of cycle

# %% [markdown] slideshow={"slide_type": "skip"}
# ## Zooming In

# %% hideCode=false slideshow={"slide_type": "skip"}
year = data[(data["decimal date"] > 2021) & (data["decimal date"] <= 2022)]

plt.figure(figsize=(14,6))
plt.plot(year["decimal date"], year["average"], "ko")
plt.title("Zoomed in Mauna Lao")
plt.xlabel("Year")
plt.ylabel("C02 Measurement");

# %% [markdown] slideshow={"slide_type": "notes"}
# If we zoom in we can see a couple more features. The data has an an annual cycle which peaks in the spring, lowest in the fall

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Lets create a model
# <center>
#     <div>
#        <img src="img/PyMC.png" height="400" />
#     </div>
# </center>

# %% [markdown] slideshow={"slide_type": "notes"}
# We're statisticians so buiding models is what we do. 

# %% [markdown] slideshow={"slide_type": "subslide"}
# ##  Lets create a model using a principled workflow

# %% [markdown]
# <center>
#     <div>
#        <img src="img/IB_Bayesian_workflow.jpg" height="400" />
#     </div>
# </center>

# %% [markdown] slideshow={"slide_type": "notes"}
#
# But we're also principled statisticians that follow the bayesian workflow
# Let's start simple, and try and increase complexity.
# Sometimes it's hard to resist the temptation to reach for the most "advanced" approach you know first.  It's always better to work up in this direction than the opposite.
#
# We'lll

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Modeling the trend with a function

# %% [markdown]
# $$\huge{CO2\_Level = f(x)}$$

# %% [markdown] slideshow={"slide_type": "notes"}
# Keeping it simple lets model the trend.
#
# For now lets ignore the small year to year variation First lets try fitting the whole thing with a straight line.  Let's ignore the annual up-and-down pattern (the technical term is called "seasonality", link to description of term) 
#
# Lets also start simple and redefine the basics
# Mathematical functions are the main tools we use to fit data.  Whether we are extrapolating or forecasting, or interpolating, we need something that given an `x`, outputs a `y`, or the range of plausible `y` values.  
#

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Linear Function

# %% [markdown]
# $$CO2\_Level = mx + b$$

# %% [markdown] slideshow={"slide_type": "notes"}
# Lets start by using familiar function you've seen before but before you skip this slide lets talk through it.
# We're making a bold statement here. We're saying that if we have model parameters m and b, we can estimate C02_level by plugging in a value of X.
#
#

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Estimating the  parameters

# %% [markdown]
# $$CO2\_Level = \textbf{m}x + \textbf{b}$$

# %% [markdown] slideshow={"slide_type": "notes"}
# Now we still need to figure out What should we guess for $m$ and $b$?
#
# - $\beta_0$ is the Y intercept, so what's the CO2 value here at year = 0?  Kind of hard.  It's usually good to normalize our data, so lets
# rescale things so that the data starts near year "zero".  We can do this by subtracting the smallest x date from all our x values. Doing so won't change the results of our model, so long as we make sure to add it back in later.

# %% [markdown] slideshow={"slide_type": "subslide"}
# # *A* way to pick an intercept
#

# %%
intercept = data.sort_values(["decimal date"])["average"].head(1)
intercept

# %%
plt.figure(figsize=(14,6))
plt.plot(data["decimal date"] - data["decimal date"].min(), data["average"], "k.");

plt.grid(True)
plt.plot(0, intercept.values, marker="o", markersize=10)
plt.ylim([275, 430])
plt.xlabel("Year Index")
plt.ylabel("C02 Measurement");

# %% [markdown] slideshow={"slide_type": "notes"}
# Let's just say we decide the intercept is where the first data point is. We're the modelers here so we can pick our parameters how we choose.
#
# Lets just assume the first point is the intercept

# %% [markdown] slideshow={"slide_type": "subslide"}
# # *A* way to pick a slope

# %%
xl = np.linspace(0, 70, 200)

possible_line1 = 315.7 + 1.5 * xl

# %%
plt.figure(figsize=(12,5))
plt.plot(data["decimal date"] - data["decimal date"].min(), data["average"], "k.")
plt.plot(0, intercept.values, marker="o", markersize=10)

plt.plot(xl, possible_line1, "dodgerblue")
plt.xlabel("Year Index")
plt.ylabel("C02 Measurement");

# %% [markdown] slideshow={"slide_type": "notes"}
# **TODO**: Add rise over run picture
#
# Now that we've picked an intercept all we need to do is pick a slope. Lets use rise over run
#
# The slope parameter `m` might be a little easier.  Slope is defined as "rise over run", so lets try calculating. We might as well use that as one of our guesses.  Eyeballing it with upward curve makes this a bit tricky, but lets also try 300, and 290.   
#
# The slope parameter `m` might be a little easier.  Slope is defined as "rise over run", so lets try calculating that at a few spots.  

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Multiple Estimations

# %% slideshow={"slide_type": "-"}
xl = np.linspace(0, 70, 200)

possible_line1 = 315.7 + 1.5 * xl
possible_line2 = 300.0 + 2.0 * xl
possible_line3 = 290.0 + 2.0 * xl

# %%
plt.figure(figsize=(12,5))
plt.plot(data["decimal date"] - data["decimal date"].min(), data["average"], "k.");
plt.plot(xl, possible_line1, "dodgerblue")
plt.plot(xl, possible_line2, "dodgerblue")
plt.plot(xl, possible_line3, "dodgerblue")

plt.grid(True)
plt.xlabel("Year Index")
plt.ylabel("C02 Measurement");

# %% [markdown] slideshow={"slide_type": "notes"}
# We can pick a couple more data points and repeat this process
# - Lets start from year 10 to year 30:  Looking at the chart, at year 10, CO2 is about 320.  At year 30 CO2 is about 350.  That gives $\frac{350 - 320}{30-10} = 1.5$. - Year 30 to year 60: $\frac{410 - 350}{60-30} = 2.0$
# - Year 40 to year 50: $\frac{385 - 370}{50-40} = 2.0$
#
# Lets plug in our guesses and plot a few lines.
#
# These are clearly pretty rudimentary guesses, but they at least look plausible?
#
# How can we be more rigorous mathematically though? 

# %% [markdown] slideshow={"slide_type": "subslide"}
# # OLS Regression
# Add statsmodel to dependencies later, or switch this to matrix inverse method

# %%
import statsmodels.formula.api as sm
result = sm.ols(formula="A ~ B + C", data=df).fit()

# %% [markdown]
# Matrix inverse method, least squares, don't necessarily need statsmodels, formula language, all that.  Have folks seen that before? Not sure how much explanation is needed here. 

# %%
x = data["decimal date"].values - data["decimal date"].min()
X = np.vstack((np.ones(len(x)), x)).T 

y = data["average"].values

((b_est, m_est), _, _, _) = np.linalg.lstsq(X, y, rcond=None)
b_est, m_est

# %% [markdown]
# The least squares estimates for the y-intercept is 305.6, and for the slope its 1.6, which are pretty close to our previous guesses.

# %%
plt.figure(figsize=(12,5))
# plot data
plt.plot(data["decimal date"] - data["decimal date"].min(), data["average"], "k.");

# plot guesses
plt.plot(xl, possible_line1, "dodgerblue")
plt.plot(xl, possible_line2, "dodgerblue")
plt.plot(xl, possible_line3, "dodgerblue", label="Guesses")

# plot least squares estimate
plt.plot(xl, b_est + m_est * xl, color="tomato", lw=3, label="Least squares estimate");


plt.legend();
plt.grid(True)
plt.xlabel("Year Index")
plt.ylabel("C02 Measurement");

# %% [markdown] slideshow={"slide_type": "notes"}
# Now some of you might be saying, well obviously a better way is to minimize the sum of squares between each observed data point and the line to estimate the optimal m and b.
#
# Some other people might even be a bit fancier and say, we're going to least squares regression or ordinary linear regression
#
# And most people dont say either, but instead just load up excel or their favorite ML library, press run, get a line, and think thats theres no better way

# %% [markdown] slideshow={"slide_type": "subslide"}
# # Bayesian Regression

# %%
with pm.Model() as model:
    b = pm.Normal("b", mu=300, sigma=100)
    m = pm.Normal("m", mu=0.0, sigma=100)
    sigma = pm.HalfNormal("sigma", sigma=10)
    
    x_ = pm.MutableData("x", data["decimal date"] - data["decimal date"].min())
    mu = pm.Deterministic("mu", m * x_ + b)
    
    pm.Normal("y", mu=mu, sigma=sigma, observed=data["average"])
    

# %% [markdown] slideshow={"slide_type": "notes"}
# But we're bayesian here, and we know how powerful this estimation method can be not to get one line, but all lines. Lets go ahead and use PyMC to get all possible lines **and** their relative plausability.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Fit the model in PyMC

# %%
with model:
    idata = pm.sample()

# %%
az.plot_posterior(idata.posterior, var_names=["b", "m", "sigma"]);

# %% [markdown] slideshow={"slide_type": "notes"}
# We can see that with a Bayesian approach, we can let the model produce guesses, and associate each guess with a "plausibility score", or a posterior probablity. 
#

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Making Predictions

# %%
with model:
    pm.set_data({'x': xl})
    ppc = pm.sample_posterior_predictive(idata, var_names=["mu", "y"])
    idata.extend(ppc)

# %% [markdown] slideshow={"slide_type": "notes"}
# We also can make predictions about future observations themselves

# %%
plt.figure(figsize=(12,5))

az.plot_hdi(x=xl + data["decimal date"].min(), hdi_data=az.hdi(idata.posterior_predictive.mu), color="slateblue")
az.plot_hdi(x=xl + data["decimal date"].min(), hdi_data=az.hdi(idata.posterior_predictive.y), color="slategray")

plt.plot(data["decimal date"], data["average"], "k.");

plt.grid(True);

# %% [markdown] slideshow={"slide_type": "notes"}
# - Our guesses were pretty good, but PyMC's guesses are better
# - Blue shaded is posterior predictive estimate of the line.  Since we have so much data, PyMC is pretty confident about this.
# - Gray shaded is posterior predictive estimate of the data generated by the line.  It's wide enough to cover most of the data.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Could we do better though?

# %% hideCode=true
plt.figure(figsize=(12,5))

az.plot_hdi(x=xl + data["decimal date"].min(), hdi_data=az.hdi(idata.posterior_predictive.mu), color="slateblue")
az.plot_hdi(x=xl + data["decimal date"].min(), hdi_data=az.hdi(idata.posterior_predictive.y), color="slategray")

plt.plot(data["decimal date"], data["average"], "k.");
plt.xlim(2010, 2023)
plt.ylim(370, 440)

plt.grid(True)
plt.xlabel("Year")
plt.ylabel("C02 Measurement");

# %% [markdown]
# If we zoom in again though we can more clearly see that there's extra patterns were missing. In the next sections we'll address those as well. For now let's talk through our section recap

# %% [markdown] hidePrompt=true slideshow={"slide_type": "subslide"}
# ## Section Recap
# * We want to estimate C02 Levels starting with a line
#
# ### Review
# * Drew a single line using 
#   * a elementary school method
#   * Ordinary Least Square Regression
#   * Drew many lines using Bayes Linear Regression
#
# ### Key takeaways
# * Functions take inputs and provide us estimates
#   * Needed to estimate the parameters
# * We *assumed* a functional form mx+b
#

# %% [markdown]
# # Section 20:  How to draw a wiggly line
# *Adding* more nuance

# %% [markdown]
# $$CO2\_Level = \textbf{m}x + \textbf{b}$$

# %%
plt.figure(figsize=(12,5))

az.plot_hdi(x=xl + data["decimal date"].min(), hdi_data=az.hdi(idata.posterior_predictive.mu), color="slateblue")
az.plot_hdi(x=xl + data["decimal date"].min(), hdi_data=az.hdi(idata.posterior_predictive.y), color="slategray")

plt.plot(data["decimal date"], data["average"], "k.");

plt.grid(True)

plt.xlabel("Year")
plt.ylabel("C02 Measurement");

# %% [markdown]
# In the last section we roughly we defined a linear model to capture the trend. We were able to estimate the parameters of that linear model using Bayesian method. However because our model only defined a line all, we got back straight lines

# %% [markdown]
# ## What about all periodicity?

# %%
plt.figure(figsize=(12,5))

az.plot_hdi(x=xl + data["decimal date"].min(), hdi_data=az.hdi(idata.posterior_predictive.mu), color="slateblue")
az.plot_hdi(x=xl + data["decimal date"].min(), hdi_data=az.hdi(idata.posterior_predictive.y), color="slategray")

plt.plot(data["decimal date"], data["average"], "k.");
plt.xlim(2010, 2023)
plt.ylim(370, 440)

plt.grid(True)
plt.xlabel("Year")
plt.ylabel("C02 Measurement");

# %% [markdown]
# We say that were missing the squigliness though, more specifically referred to are periodicity.

# %% [markdown]
# ## Zooming in on one period, or cycle

# %%
year = data[(data["decimal date"] > 2021) & (data["decimal date"] <= 2022)]

plt.figure(figsize=(14,6))
plt.plot(year["decimal date"], year["average"] - year["average"].mean(), "ko");

# %% [markdown]
# By inspecting these points up close we can see a familiar function. Try guessing what it is. We'll pause for a second to give a moment to think.
#
# Did you say sine? Or did you guess cosine? Either way you're right

# %% [markdown]
# ## Overlaying out random guess

# %% [markdown]
# $$CO2\_Level = sin(year* \pi)$$

# %%
year = data[(data["decimal date"] > 2021) & (data["decimal date"] <= 2022)]

plt.figure(figsize=(14,6))
plt.plot(year["decimal date"], year["average"] - year["average"].mean(), "ko");

xl = np.linspace(year["decimal date"].min(), year["decimal date"].max(), 100)
plt.plot(xl, 3*np.cos(2 * np.pi * xl * 1.0 - 0.5 * np.pi));

# %% [markdown]
# Like our trend we can use use the sine function parameterized by x, or year in this case, to model our data.
#
# We guessed some parameters like before but it's great, there's certainly room for improvement.  This does give us some OK starting points though, and we can use these to help us set priors for our PyMC model.
#
# - frequency: 1 (cycle per year)
# - amplitude: 3.  After normalizing the observed data by subtracting the mean, we can see the cycle goes up and down from about -3 to 3. 
# - phase: $-\frac{\pi}{2}$, this gets cosine to start at zero and increase first, which sort of matches the data, but should be a free parameter in the model.

# %% [markdown]
# ## Sine waves don't go up

# %%
data["decimal date"]
sin_component = 3*np.cos(2 * np.pi * xl)

# %%
year = data[(data["decimal date"] > 1960) & (data["decimal date"] <= 2022)]

plt.figure(figsize=(14,6))
plt.plot(year["decimal date"], year["average"] - year["average"].mean(), "ko");

xl = np.linspace(year["decimal date"].min(), year["decimal date"].max(), 10000)
plt.plot(xl, 3*np.cos(2 * np.pi * xl * 1.0 - 0.5 * np.pi));

# %% [markdown]
# We have another problem though, sine waves don't go up but we have a solution for that

# %% [markdown]
# ## Additive models

# %%
xl = np.linspace(year["decimal date"].min(), year["decimal date"].max(), 10000)

sin_component = 3*np.cos(2 * np.pi * xl * 1.0 - 0.5 * np.pi)

# We bot these from the Bayesian estimation
m, b =1.6, 306
linear_component = m*(xl-year["decimal date"].min()) + b

squiggly_line = linear_component + sin_component

# %%
plt.figure(figsize=(14,6))
plt.plot(year["decimal date"], year["average"], "ko", alpha=.1);

plt.plot(xl, squiggly_line);

# %% [markdown]
# What we could do is add our linear term and our sin wave together to create a new model. Now we get the effects of both together. Better, now instead of guessing parameter let's use Bayesian methods to estimate our parameters again.

# %% [markdown]
# ## Bayesian Additive model

# %%
with pm.Model() as model:
    b = pm.Normal("b", mu=300, sigma=100)
    m = pm.Normal("m", mu=0.0, sigma=100)
    sigma = pm.HalfNormal("sigma", sigma=10)
    
    x_ = pm.MutableData("x", data["decimal date"] - data["decimal date"].min())
    mu_linear = pm.Deterministic("mu_linear", m * x_ + b)
    
    
    A = pm.HalfNormal("A", sigma=5)
    phi = pm.Normal("phi", mu=-np.pi / 2, sigma=0.1)
    freq = pm.Normal("freq", mu=1.0, sigma=0.1)
    mu_periodic = pm.Deterministic("mu_periodic", A * pm.math.cos(2 * np.pi * x_ * freq + phi))
    
    mu = pm.Deterministic("mu", mu_linear + mu_periodic)
    
    pm.Normal("y", mu=mu, sigma=sigma, observed=data["average"])
    
    
with model:
    idata = pm.sample(tune=2000, chains=2, target_accept=0.98)

# %% [markdown]
# You'll notice a couple of extra terms now.

# %% [markdown]
# ## Our parameter estimates

# %%
az.plot_posterior(idata.posterior, var_names=["b", "m", "A", "freq", "phi", "sigma"]);

# %% [markdown]
# Now using ArviZ let's plot the posterior 
#
# - It was a bit tricky to get the sampler to converge, but now it looks like the results are sensible.  The parameter estimates are also somewhat in line with our guesses.  
# - Now lets look at the posterior predictive and draw some samples

# %%
xl = np.linspace(0, 70, 1000)

with model:
    pm.set_data({'x': xl})
    ppc = pm.sample_posterior_predictive(idata, var_names=["mu", "y"])
    idata.extend(ppc)

# %% [markdown]
# ## Making prections

# %%
plt.figure(figsize=(12,5))

az.plot_hdi(x=xl + data["decimal date"].min(), hdi_data=az.hdi(idata.posterior_predictive.mu), color="slateblue")
az.plot_hdi(x=xl + data["decimal date"].min(), hdi_data=az.hdi(idata.posterior_predictive.y), color="slategray")

mu_samples = idata.posterior_predictive.mu.stack(sample=["chain", "draw"]).isel(sample=[1,2,3]).values

plt.plot(data["decimal date"], data["average"], "k.");
plt.plot(xl + data["decimal date"].min(), mu_samples, color='dodgerblue');

plt.grid(True);

# %% [markdown]
# Usnig 
# - the results look similar to the straight line model, except when we look at individual samples of `mu` we can see the seasonal pattern reflected. 
# - Lets zoom in also and look at the small scale structure

# %% [markdown]
# ## Zooming into our posterior predictive

# %%
plt.figure(figsize=(12,5))

az.plot_hdi(x=xl + data["decimal date"].min(), hdi_data=az.hdi(idata.posterior_predictive.y), color="slategray")

mu_samples = idata.posterior_predictive.mu.stack(sample=["chain", "draw"]).isel(sample=[1,2,3]).values

plt.plot(data["decimal date"], data["average"], "k.");
plt.plot(xl + data["decimal date"].min(), mu_samples, color='dodgerblue');

plt.grid(True);
plt.xlim([2005, 2010]);
plt.ylim([360, 400]);

# %% [markdown]
# Even at the best looking part of the curve, there are still pretty obvious issues still here.
#
# What is wrong:
# - Linear term seems insufficient.  Upward increasing trend
# - "Cosine" seems insufficient.  Is it a perfect sine wave?  Or is something that's almost a sine wave, but still has a repeating pattern.
# - How much more fine structure are we missing in the data?
#
# Now you might say well lets go add another function, but where does it stop?
#
# - With this functional approach, nothing stops us from forecasting infinitely far into the future.  Is that reasonable?

# %% [markdown]
# ## The steps to our current approach
#
# 1. Guess at a functional form of the series
# 2. Estimate parameters for that function
# 3. See what's wrong
# 4. Go back to step 1

# %% [markdown]
# Continuing with this approach, what might we try?  Coming up with functional forms is difficult and it doesnt always work!  Also, how much do we care about these parameters? We might just to know what the CO2 levels might be, not necssarily care about the components of the model or their parameters
#
# Now don't get us wrong, additive models are a valid approach in many situations, but that's not our case. Nor is this is the right course! The good news is there is another approach, but before we get ahead of ourselves lets talk through our section recap

# %% [markdown]
# ## Section Recap
# * Functions can be added together to create new models
#   * Knowing as additive models
# * The modeler still needs to decide what the functions are
#   * And estimate the parameters even if they're not if interest
# * With the C02 estimation even with an extended additive model there's issues

# %% [markdown]
# # Section 30
#
# Let's take a different approach, lets try drawing, and lets think carefully about *how* we are drawing.  To start, lets focus on the most recent years worth of data.

# %%
year = data.sort_values("decimal date").tail(12)

plt.figure(figsize=(14,6))
plt.plot(year["decimal date"], year["average"], "ko");

# %% [markdown]
# - Connect the dots?   No, we shouldn't expect to predict CO2 at every point.  The process by which CO2 gets into the atmosphere is too complicated.  The line we draw shouldn't pass through every point perfectly, but there's clearly some kind of overall pattern happening.
# - If we hit every point, how would we have any uncertainty?
# - Lets start by drawing somewhere in the middle.  

# %%
### HIDE THIS CODE
# Going to pretend to draw the MAP estimate of this model as what we "draw", but suprise, it's actually a GP. 

subset = data.sort_values("decimal date").tail(12)
x = subset["decimal date"].values - subset["decimal date"].min()

y_mu = subset["average"].mean()
y_sd = subset["average"].std()
y = (subset["average"].values - y_mu) / y_sd

coords = {'time': subset['decimal date'].values}

with pm.Model(coords=coords) as model:
    
    eta = pm.HalfNormal('eta', sigma=1)
    ell = pm.Gamma('ell', alpha=2, beta=3)
    cov_eq = eta**2 * pm.gp.cov.Exponential(1, ls=ell)
    
    eta_per = pm.HalfNormal('eta_per', sigma=1)
    ell_per = pm.Gamma('ell_per', alpha=2, beta=3)
    cov_per = eta_per**2 * pm.gp.cov.Periodic(1, period=1.0, ls=ell_per)
        
    gp_eq = pm.gp.Marginal(cov_func=cov_eq)
    gp_per = pm.gp.Marginal(cov_func=cov_per)
    
    gp = gp_eq + gp_per
    
    sigma = pm.HalfNormal("sigma", sigma=1)
    lik = gp.marginal_likelihood("lik", X=x[:, None], y=y, noise=sigma, dims='time')
    
with model:
    idata = pm.sampling_jax.sample_blackjax_nuts()

# %%
az.plot_trace(idata, var_names=['ell', 'ell_per'], transform=lambda x: x);

# %%
xnew = np.linspace(np.min(x) - 1, np.max(x) + 1, 500)

with model:
    f = gp.conditional('f', Xnew=xnew[:, None], jitter=1e-4)
    ppc = pm.sample_posterior_predictive(idata, var_names=['f'])
    
idata.extend(ppc)


# %%
def plot_range(xmin, xmax):
    fig = plt.figure(figsize=(14, 6))
    ax = fig.gca()

    f = idata.posterior_predictive.f.stack(sample=['chain', 'draw']).mean(dim='sample')
    gp_posterior = y_sd * f + y_mu 

    x = xnew + subset["decimal date"].min()
    keep_ix = [ix for ix, v in enumerate(x) if xmin < v < xmax]
    
    ax.plot(x[keep_ix], gp_posterior[keep_ix], lw=5); 
    ax.plot(subset["decimal date"], subset["average"], "ok");
    ax.set_xticks(np.arange(2021.4, 2022.5, 0.1))
    return ax


# %% [markdown]
# ### Start drawing
#
# Let's start drawing at Jan 1, 2022, and try and think carefully about *why* we are drawing the line the way we are.  It seems like a completely natural thing, to just trace these lines out, but while drawing let's pay really close attention to HOW are we taking the information in the data, and translating that into our lines that we are tracing.

# %%
ax = plot_range(2022.0, 2022.03)

# %% [markdown]
# Starting in early January 2022, lets drawing the line both to the left and the right.  For planning out where we'll move the pencil too, clearly the first date points to the left and right of where we are starting are the most important to consider.  We are definitely going to draw nearest (in the "y" sense) to the point nearest in the "x" sense, and the direction and slope of our line is determined be the point past it, and the point before we start drawing.  These are the three most important points to consider, with the point before 2021.1 is the most important.

# %%
ax = plot_range(2021.97, 2022.08)

tmp = data[(data["decimal date"] > 2022.0) & (data["decimal date"] < 2022.1)]
ax.plot(tmp["decimal date"], tmp["average"], "oy", markersize=20, alpha=0.7);

tmp = data[(data["decimal date"] > 2021.9) & (data["decimal date"] < 2022.2)]
ax.plot(tmp["decimal date"], tmp["average"], "oy", markersize=15, alpha=0.7);
ax.plot(tmp["decimal date"], tmp["average"], "ok");

# %% [markdown]
# Let's think if there is a rule or algorithm we can express mathematically for what we are doing here.  In a rough sense, we 
# 1. Pick a starting point
# 2. Find the closest three data points (closest in terms of "x")
# 3. This one is fuzzier, we sort of draw a line using the "y" values as guideposts.
#
# This isn't quite enough.  To make an algorithm or mathematical model, we need to be specific.  Point 3 is not specific enough, but lets roll with it anyway and see where we end up.
#
# Is this enough?  To help visualize, lets just look at these three data points and ask, if we just see these three and not the rest, would we still draw the same line?

# %%
ax = plot_range(2021.97, 2022.08)

tmp = data[(data["decimal date"] > 2022.0) & (data["decimal date"] < 2022.1)]
ax.plot(tmp["decimal date"], tmp["average"], "oy", markersize=20, alpha=0.7);

tmp = data[(data["decimal date"] > 2021.9) & (data["decimal date"] < 2022.2)]
ax.plot(tmp["decimal date"], tmp["average"], "oy", markersize=15, alpha=0.7);
ax.plot(tmp["decimal date"], tmp["average"], "ok");

ax.set_xlim([2021.9, 2022.2]);

# %% [markdown]
# Looking at these, the answer is "Maybe?".  But what if then we saw another point here (red):

# %%
ax = plot_range(2021.97, 2022.08)

tmp = data[(data["decimal date"] > 2022.0) & (data["decimal date"] < 2022.1)]
ax.plot(tmp["decimal date"], tmp["average"], "oy", markersize=20, alpha=0.7);

tmp = data[(data["decimal date"] > 2021.9) & (data["decimal date"] < 2022.2)]
ax.plot(tmp["decimal date"], tmp["average"], "oy", markersize=15, alpha=0.7);
ax.plot(tmp["decimal date"], tmp["average"], "ok");

ax.plot([2021.91], [419], "or", markersize=10)

ax.set_xlim([2021.9, 2022.2]);

# %% [markdown]
# The blue line heading to the left should probably start bending a little sooner than it is to the new red point, and probably shouldn't pass through the datapoint that it's about to run into.  
#
# ### Digression, expanding on step 2
#
# How do we think about this then?  Maybe our rule of 3 isn't good enough.  Maybe we need to consider more data points.  But clearly, the data points closer to where we are drawing are more important than ones further away, so let's *weight them based on how far away they are from the pencil*.  From our starting point on Jan 1, 2022, let's give every other data point a weight, ranging from 1, most important, to 0, not important at all.  For a datapoint to get a 1, it would have to be exactly at Jan 1, 2022.  For a data point to get a zero, it would have to be infinitely far away. 
#
# **It's easier to design a mathematical formula to do this for us, instead of just coming up with weight values that look right based on the chart**  What if we had a thousand data points to do this for!
#
# We'll need a few peices to construct our formula:
# - The location of our pencil, $x'$
# - To make this general to any dataset, let's define distance from our pencil to any point on the x axis (not just the data points we have above at their discrete locations), $x - x' $.  We need the distance to always be positive, so let's square it, $d = (x - x')^2$.
# - Right now the distance goes from 0 (super close) to $\infty$ (super far).  We want to scale it to go from 1 (super close) to zero (super far).  All we have to do is transform it using the exponential function, and make the exponent negative,
# $$
# k(x, x') = \exp(-d) = \exp( -(x - x')^2 )
# $$
# - If you've seen GPs before, you might be starting to recognize this function!  If not, don't worry, we'll experiment with functions like this much more.  

# %% [markdown]
# It might feel like there's a lot to unpack here, but let's remember our goal:  **We want a way to "weight" different data points for when we draw a line, based on where our pencil currently is**.  We intuitively know how to draw a line through some dots, to the point where it feels so easy that you don't even think about *how* you're actually doing it.  In order to be precise, and to make this intuition "implementable" by a computer for any kind of data, we need to represent this intuition into a bit of math.  
#
#
# Looking at the above figure, our pencil is at zero, and this gives us the "similarity" of its "y" value, using the distance of other points on "x".  We'll weight each of the $y$ values according to our similarity function, and then sum them to get a weighted average of y values. 
#
# Try this yourself!  Make a small table of (x, y) coordinates (you pick), and see how we can use our similarity function to automatically let us automatically draw in reasonable values.  Here's an example:

# %%
coordinates = [
    (0.0, 2.0),
    (1.0, 3.0),
    (3.0, 2.0),
    (4.0, 2.5),
]
x, y = zip(*coordinates)

x = np.asarray(x)
y = np.asarray(y)

plt.plot(x, y, 'ok');

# %% [markdown]
# theres our example data, and here is the similarity function k, and we use it to get our weights, one for each data point in the data set.

# %%
k = lambda x, x_prime: np.exp(-(x - x_prime)**2)

x_prime = 2.0 # we want to fill in where our pencil is, at x = 2
weights = k(x, x_prime)
weights

# %%
np.sum(weights)

# %%
# notice that our weights don't sum to 1!  So lets just normalize them so that they do
weights_n = weights / np.sum(weights)

# now we can weight our "y" values so they sum to 100%

# %%
pencil_y = np.sum(y * weights_n)

# %%
plt.plot(x, y, 'ok');
plt.plot([x_prime], [pencil_y], "bo");

# %% [markdown]
# Looks reasonable!  Let's do a bunch in for loop, as if our pencil is drawing:

# %%
fig = plt.figure(figsize=(8,5))
ax = fig.gca();

ax.plot(x, y, 'ok');

x_primes = np.linspace(0.5, 4.8, 30)

for x_prime in x_primes:  ## animate this?
    weights = k(x, x_prime)
    weights_n = weights / np.sum(weights)
    pencil_y = np.sum(y * weights_n)
    
    ax.plot([x_prime], [pencil_y], "o", color="dodgerblue")

# %% [markdown]
# We almost got a real GP going here!  We are about here:
#
# ![image.png](attachment:8c0a249f-0ef9-4c8f-9ebf-1c572dd75f18.png)
#
# There's still a ways to go until we get to a full fledged Gaussian process, with uncertainty estimates and everything.  But this is really the essence of *how* GPs are work to model data under the hood.  Let's review our algorithm we had earlier, we can make it much more specific now:
#
# #### Old version:
#
# 1. Pick a starting point
# 2. Find the closest three data points (closest in terms of "x")
# 3. This one is fuzzier, we sort of draw a line using the "y" values as guideposts.
#
#
# #### Updated:
#
# 1. Pick a starting point, call it $x'$.
# 2. Calculate the "similarity" weights between $x'$ and all the other $x$ locations in the data.
# 3. Normalize so they sum to 100\%, then apply weights to the corresponding $y$ values.  Sum to get an estimate of $y' = f(x')$.
#
# Then repeat for any other $x'$ you need!
#
# ##### Heads up:
# - Although we used the metaphor of drawing, we don't need to actually put a marker down, and then "move" from that marker left to right.  We can just pick any arbitrary $x'$ points and get estimates for them this way.  

# %% [markdown]
# ### Back to Mauna Loa
#
# Let's finish drawing in our line using our new proto-GP method!  Should be easy now right?

# %%
subset = data.sort_values("decimal date").tail(12)

x = subset["decimal date"].values
y = subset["average"].values

fig = plt.figure(figsize=(14, 6))
ax = fig.gca()

ax.plot(x, y, "ok")
ax.set_xticks(np.arange(2021.4, 2022.5, 0.1))


x_primes = np.linspace(2021.4, 2022.5, 100)

for x_prime in x_primes:  ## animate this?
    weights = k(x, x_prime)
    weights_n = weights / np.sum(weights)
    
    pencil_y = np.sum(y * weights_n)
    
    ax.plot([x_prime], [pencil_y], "o", color="dodgerblue")

# %% [markdown]
# What went wrong?!  We need one last finishing touch to make this something usable.  

# %% [markdown]
# Let's add another parameter that "standardizes" the distance between $x$ and $x'$.  What if our data is between x = 0 and x = 0.1?  Maybe we still want the distance from x = 0 and x = 0.1 to feel "far".  Or what if our data is between 500 and 50,000, and the distance between 700 and 800 is "close"?  Let's call this parameter $\ell$ for "lengthscale", since we use it to rescale the lengths or distances between our data.  
# - We redefine our distance $d$ to be $d = \left(\frac{(x - x')}{\ell}\right)^2 = \frac{(x - x')^2}{\ell^2}$
#
# Our new "similarity" function (since it gives 1 is similar and 0 as not) is then
# $$
# k(x, x') = \exp(-d) = \exp \left( \frac{-(x - x')^2}{\ell^2} \right)
# $$
#
# #### Heads up
#
# You might recognize this as *almost* a normal, or Gaussian distribution.  This is NOT where the name "Gaussian" in Gaussian process comes from.  We'll see where it comes from in the next lesson.  
#
# Lets make a plot of our new function to see how it works, and verify that it behaves how we want it to.  Not considering our data above, lets put our "pencil" at x' = 0, and plot this from x = 0 to x = 5.  Let's try different lengthscales too.

# %%
x = np.linspace(0, 5, 100)

x_prime = 0.0

k = lambda x, x_prime, lengthscale: np.exp(-(x - x_prime)**2 / lengthscale**2 )

lengthscale = 0.25
plt.plot(x, k(x, x_prime, lengthscale), label=f"lengthscale = {lengthscale}", lw=2);

lengthscale = 1.0
plt.plot(x, k(x, x_prime, lengthscale), label=f"lengthscale = {lengthscale}", color="k", lw=3);

lengthscale = 2.0
plt.plot(x, k(x, x_prime, lengthscale), label=f"lengthscale = {lengthscale}", lw=2);

lengthscale = 3.0
plt.plot(x, k(x, x_prime, lengthscale), label=f"lengthscale = {lengthscale}", lw=2);

plt.ylabel("Similarity weight")
plt.xlabel("x")

plt.legend();

# %%

# %%

# %%

# %%

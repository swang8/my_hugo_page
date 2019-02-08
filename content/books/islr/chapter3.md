---
title: "Chapter3"
date: 2019-02-07T14:46:09-06:00
tags: ['islr', 'statistics learning', 'R']
---
# Linear Regression

*Linear regression* is a very simple supervised learning methods, though still very useful. 

## Simple Linear Regression

*Simple linear regression* is a straightforward approach for predicting a quantitative response $Y$ on the basis of a single predictor variable $X$. It assumes that there is approximately a linear relationship between $X$ and $Y$.

<div>
$$
Y \approx \beta_{0} + \beta_{1}X
$$
</div>

In the equation, $\beta_0$ and $\beta_1$ are two unknown constants that represetn the *intercept* and *slope* termes in the linear model. Together, $\beta_0$ and $\beta_1$ are know as the model `coefficients` or `parameters`. Once we have used our training data to produce estimates $\hat{\beta}_0$ and $\hat{\beta}_1$ for the model coefficients, we can predict the response $\hat{y}$.

<div>
$$
\hat{y} = \hat{\beta}_0 + \hat{\beta}_{1} x
$$
</div>

### Estimate the Coefficients
In practic, $\beta_0$ and $\beta_1$ are unknown. So before we can use the equation to make predictions, we must use data to estimate the coefficients. Let 

<div>
$$
(x_1,y_1), (x_2, y_2), \ldots, (x_n, y_n)
$$
</div>

represent $n$ observation pairs, each of which consists of a measurement of $X$ and a measurement of $Y$. Our goal is to obtain coefficient estimates $\hat{\beta}_0$ and $\hat{\beta}_1$ such that the linear model fits the avilable data well, that is, so that $y_i \approx \hat{\beta}_0 + \hat{\beta}_1 x_i$ for $i=1,2,\ldots,n$. In other words, we want to find an intercept $\hat{\beta}_0$ and a slope $\hat{\beta}_1$ such that the resulting line is as close a possible to the data points.

There are number of ways of measuing `closeness`. However, by far the most common approach involes minimiizng the `least squares` criterion.

Let $\hat{y} = \hat{\beta}_1 + \hat{\beta}_1 x_i$ be the prediction for $Y$ based on the *i*th values of $X$. The $e_i = y_i - \hat{y}_i$ represents the *ith* `residual` -- this is the difference between the *i*th response value that is predicted by our linear model. We define the `residual sum of squares (RSS)` as

<div>
$$
RSS=e_1^2 + e_2^2 + \ldots + e_n^2
$$
</div>

or equivalently as 
<div>
$$
RSS = (y_1 - \hat{\beta}_0 - \hat{\beta}_1 x_1)^2 + (y_2 - \hat{\beta}_0 - \hat{\beta}_1 x_2)^2 + \ldots + (y_n - \hat{\beta}_0 - \hat{\beta}_1 x_n)^2 
$$
</div>

The least squares approach choose $\hat{\beta}_0$ and $\hat{\beta}_1$ to minimize the RSS. Using some calculus, one can show that the minimizers are 

<div>
$$
\begin{aligned}
&\hat{\beta}_1 = \frac{\sum_{i=1}^n(x_i-\bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2},
\\
&\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x},\\
&where, \\
&\bar{y} \equiv \frac{1}{n} \sum_{i=1}^n y_i, \\
&\bar{x} \equiv \frac{1}{n} \sum_{i=1}^n x_i
\end{aligned}

$$
</div>

The aforementioned formula defines the `least squares coefficient estimates` for simple linear regression.

### Assessing the accuracy of the Coefficient Estimates

We assume that the *true* relationship between $X$ and $Y$ takes for thorm $Y=f(X)+\epsilon$ for some unknown function $f$, where $\epsilon$ is a mean-zero reandom error term. If $f$ is to be approximated by a liner function, then we can write this relationship as 
<div>
$$
Y = \beta_0 + \beta_1X + \epsilon
$$
</div>

The model given in the formula defines the *population regression line*, which is the best linear approximation to the true relationship between $X$ and $Y$. The least squares regression coefficient estimates characterize the `least square line`.

Using information from a sample to estimate characteristics of a large population. For example, suppose htat we are interested in knowing the population mean $\mu$ of some random variable $Y$. Unfortunately, $\mu$ is unknown, but we do have access to $n$ observations from $Y$, which we can use to estimate $\mu$. A reasonable estimate is $\hat{\mu} = \bar{y}$, where $\bar{y} = \frac{1}{n}\sum_{i=1}^ny_i$ is the sample mean. The sample mean and the population mean are different, but in general the sample mean will provide a good estimate of the population mean.

The analogy between linear regression and estimation of the mean of a random variable is an apt one based on the concept of `bias`. If we use the sample mean $\hat{\mu}$ to estimate $\mu$, this estimate is `unbiased`, in the sense that on average, we expect $\hat{\mu}$ to equal to $\mu$. It means, if we could average a huge numbr of estimates of $\mu$ obtained from a huge number of sets of observations, then this average would *exactly* equal to $\mu$. Hence, an unbiased estimator does not *systematically* over- or under-estimate the true parameter. 

How accurate is hte sample mean $\hat{\mu}$ as an estimate of $\mu$?

We have established that the average of $\hat{\mu}$s over many data sets will be very close to $\mu$, but that a single estimate of $\hat{\mu}$ may be sunstantial underestimate or overestimate of $\mu$. How far off will that single estimate of $\hat{\mu}$ be? In general, we answer this question by computing the `standard error` of $\hat{\mu}$, written as $SE(\hat{\mu})$. We have the well-known formula

<div>
$$
Var(\hat{\mu}) = SE(\hat{\mu})^2 = \frac{\sigma^2}{n}
$$
</div>

Where $\sigma$ is the standard deviatioin of each of the realizations $y_i$ of $Y$. The equation wells us how this deviation shrinks with $n$ -- the more observations we have, the smaller the standard error of $\hat{\mu}$. 

The standard errors associated with $\hat{\beta}_0$ and $\hat{\beta}_1$ can be calculated as:

<div>
$$
\begin{aligned}
&SE(\hat{\beta}_0)^2 = \sigma^2\Big[\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\Big] \\
&SE(\hat{\beta}_1)^2 = \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
\end{aligned}
$$
</div>

Where $\sigma^2 = Var(\epsilon)$. For these formalas to be strictly valid, we need to assume that the error $\epsilon_i$ for each observation are uncorrelated with common variance $\sigma^2$. Though this might not be true for many cases, the formula still turs out to be a good approximation. 

Notice in the formula, $SE(\hat{\beta}_1)$ is smaller when the $x_i$ are more spread out; intuitively we have more leverage to estimate a slope when this is the case. We can also see that $SE(\hat{\beta}_0)$ would be the same as $SE(\hat{\mu})$ if $\bar{x}$  were zero (in which case $\hat{\beta}_0$ would be equal to $\bar{y}$). 

In gereral, $\sigma^2$ is not known, but can be estimated from the data. This estimation is known as the the `residual standard error`, and is given by the formula $RSE = \sqrt{RSS/(n-2)}$. Strictly speaking, when $\sigma^2$ is estimated from the data we should write $\widehat{SE}(\hat{\beta}_1)$ to indicate that an estimate has been made, but from simplicity of notation, we will drop the extra "hat".

Standard errors can be used to compute *confidence intervals*. A 95% confidence interval is defined as a range of values such that with 95% probability, the range will contain the true unknown value of the parameter. The range is defined in terms of lower and upper limits computed from the sample of data. For linear regression, the 95% confidence interval for $\beta_1$ approximately takes the form

<div>
$$
\hat{\beta}_1 \pm 2 \cdot SE(\hat{\beta}_1)
$$
</div>

That is, there is approximately a 95% chance that the interval 

<div>
$$
\Big[ \hat{\beta}_1 - 2 \cdot SE(\hat{\beta}_1), \hat{\beta}_1 + 2 \cdot SE(\hat{\beta}_1) \Big]
$$
</div>

will contain the true value of $\beta_1$. Similaryly, a confidence interval for $\beta_0$ approximately takes the form

<div>
$$
\hat{\beta}_0 \pm 2 \cdot SE(\hat{\beta}_0)
$$
</div>

Standard errors can also be used to perform *hypothesis tests* on the coefficients. The most common hypothesis test involves testing the *null hypothesis* of 

<div>
$$
H_0: There\ is\ no\ relationship\ between\ X\ and\ Y
$$
</div>

versus the *alternative hypothesis*

<div>
$$
H_a: There\ is\ some\ relationship\ between\ X\ and\ Y
$$
</div>

Mathematically, this corresponds to testing

<div>
$$
H_0: \beta_1 = 0
$$
</div>

versus

<div>
$$
H_a: \beta_1 \neq 0
$$
</div>

since if $\beta_1 = 0$ then the model reduces to $Y = \beta_0 + \epsilon$, and $X$ is not associated with $Y$. To test the null hypothesis, we need to determine where $\hat{\beta}_1$, our estimate for $\beta_1$, is sufficiently far from zero that we can be confident that $\beta_1$ is non-zero.

How far is far enough? This of course depends on the accuracy of $\hat{\beta}_1$--that is, it depends on $SE(\hat{\beta}_1)$. If $SE(\hat{\beta}_1)$ is small, then even relatively small values of $\hat{\beta}_1$ may provide strong evidence that $\beta_1 \neq 0$, and hence that there is a relationship between $X$ and $Y$. In contrast, if $SE(\hat{\beta}_1)$ is large, then $\hat{\beta}_1$ must be large in absolute value in order for us to reject the null hypothesis. In practice, we compute a `t-statistic`, give by

<div>
$$
t = \frac{\hat{\beta}_1 - 0}{SE(\hat{\beta}_1)}
$$
</div>

which measures the number of standard deviations that $\hat{\beta}_1$ is away from 0. If there really is no relationship between $X$ and $Y$, then we expect that the *t* will have a *t*-distribution with $n-2$ degrees ofgreedom. The *t*-distribution has a bell shape and for values of $n$ greater than approximately 30 it is quite similar to normal distribution. 

Consequently, it is a simple matter to compute the probability of observing any value equal to $|t|$ or larger, assuming $\beta_1 = 0$. We call this probability the `p-value`. Roughly speaking, we interpret the *p-value* as follows: **a small p-value indicates that it is unlikely to ovserve such a substantial association between the predictor and the resonse due to chance, in the absence of any real association between the predictor and the resonse**. Hence, if we see a small p-value, then we can infer that there is an association between the predictor and the response. We *reject* the null hypothesis -- that is, we declare a relationship to exist between $X$ and $Y$ -- if the p-value is small enough. Typical p-value cutoffs for rejecting the null hypothesis are 5 or 1 %. When $n = 30$, these correspond to t-statistics of around 2 and 2.75, respectively.  

### Assessing the accuracy of the model

Once we have rejected the null hypothesis in favor of the alternative hypothesis, it is natural to want to quantify the *extent to which the model fits the data*. The quality of linear regression fit is typically assessed using two related quantities: the `residual standard error` (RSE) and the $R^2$ statistic.

<u>*Residual Standard Error*</u>

Recall from the model that associated with each observation is an error term $\epsilon$. Due to the presence of these error terms, even if we knew the true regression line, we would not be able to perfectly predict $Y$ from $X$. The `RSE` is an estimate of the standard deviation of $\epsilon$. Roughly speaking, it is the average amount that the response will deviate from the true regression line. It is computed using the formula

<div>
$$
RSE = \sqrt{\frac{1}{n-2}RSS} = \sqrt{\frac{1}{n-2}\sum_{i=1}^n(y_i-\hat{y}_i)^2}
$$
</div>

Note that `RSS` was defined as 

<div>
$$
RSS=\sum_{i=1}^n (y_i - \hat{y}_i)^2
$$
</div>

The *RSE* is considered a measure of the *lack of fit* of the model to the data. If the predictions obtained using the model are very close to the true outcome values -- that is, if $\hat{y}_i \approx y_i$ for $i=1,2,\ldots,n$--then RSE will be small, and we can conclude that the model fits the data very well. On the other hand, if $\hat{y}_i$ is very far from $y_i$ for one or more observatins, then the RSE may be quite large, indicating that the model doesn't fit the data well.

<u>*The $\underline{R^2}$ Statistic*</u>

The RSE provides an absolute measure of lack of fit of the model to the data. But since it is measured in the units of $Y$, it is not always clear what constitutes a good RSE.

The $R^2$ statistic provides an alternative measure of fit. It takes the form of a proportion--the proportion of variance explained--and so it always takes on a value between 0 and 1, and is independent of the scale of $Y$.

To calculate $R^2$, we use the formula

<div>
$$
R^2 = \frac{TSS-RSS}{TSS} = 1 - \frac{RSS}{TSS}
$$
</div>

where $TSS=\sum(y_i - \bar{y})^2$ is the total sum of squares, which measures the total variance in the response $Y$ that can be thought of as the amount of variability inherent in the response before the regression is performed. In contrast, *RSS* measures the amount of variability that is left unexplained after performing the regression. Hence, $TSS-RSS$ measures the amount of variability in the response that is explained (or removed) by performing the regression, and $R^2$ measures the *proportion of variability in Y that can be explained using X*. An $R^2$ statistic that is close to 1 indicates that a large proportion of the variability in the response has been explained by the regression. A number near 0 indicates that the regression did not explain mch of the variability in the response; this might occur because the linear model is wrong, or the inherent error $sigma^2$ is high, or both.

The $R^2$ statistic has an interpretational advantage over the *RSE*, since unlike the *RSE*, it always lies between 0 and 1. However, it can still be challenging to determine what is a *good* $R^2$, and in general this will depend on the application. 

The $R^2$ statistic is a measure of the linear relationship between $X$ and $Y$. Recall that *correlation*, defined as 

<div>
$$
Cor(X, Y) = \frac{\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^n(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^n(y_i-\bar{y})^2} }
$$
</div>

is also a measure of the linear relationship between $X$ and $Y$. This suggests that we might be able to use $r = Cor(X,Y)$ instead of $R^2$ in order to assess the fit of the linear model. In fact, it can be shown that in the simple linear regression setting, $R^2=r^2$. In other words, the squared correlation and the $R^2$ are identical. However, for the multiple linear regression problem, in which we use several predictors simultaneously to predict the response. The concept of correlation between the predictors and the response does not extend automatically to this setting, since correlation quantifies the association between a single pair of variables rather than between a larger number of variables. We will see that $R^2$ fills this role.

## Multiple Linear Regression
A simple linear model can be extended to accommodate multiple predictors. We can do this by giving each predictor a separate slope coefficient in a single model. In general, suppose that we have $p$ distanct predictors. Then the multiple linear regression model takes the form

<div>
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon
$$
</div>

where $X_j$ represents the *jth* predictor and $\beta_j$ quantifies the association between that variable and the response. We interpret $\beta_j$ as the *average* effect on $Y$ of a one unit increase in $X_j$, *holding all other predictors fixed*.

### Estimating the Regression Coefficients



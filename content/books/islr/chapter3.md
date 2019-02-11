---
title: "Chapter3: Linear Regression"
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

The paraeters are estimated using the same least squares approach that we saw in the context of simple linear regression. We choose $\beta_0, \beta_1\ldots,\beta_p$ to minimize the sum of squared residuals.

<div>
$$
\begin{aligned}
RSS &= \sum_{i=1}^n(y_i - \hat{y}_i)^2 \\
&= \sum_{i=1}^n(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_{i1} - \hat{\beta}_2 x_{i2} - \cdots - \hat{\beta}_p x_{ip})^2
\end{aligned}
$$
</div>

### Some Important questions

When we perform multiple linear regression, we usually are interested in answering a few important questions:

1. *Is at least one of the predictors $X_1, X_2, \ldots, X_p$ useful predicting the response?*

2. *Do all the predictors help to explain $Y$, or is only a subset of the predictors useful?*

3. *How well does the model fit the data?*

4. *Given a set of predictor values, what response value should we predict, and how accuracy is our prediciton?*

<u>One: Is There a Relationship Between the Response and Predictors?</u>

Recall that in the simple linear regression setting, in order to determine whether there is a relationship between the response and the predictor we can simply check whether $\beta_1 = 0$. In the multiple regression setting with $p$ predictors, we need to ask whether all of the regression coefficients are zero, i.e. whether $\beta_1 = \beta-2 = \cdots = \beta_p = 0$. As in the simple linear regression setting, we use a hypothesis test to answer this questions We test the null hypothesis, 

<div>
$$
H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0
$$
</div>

versus the alternative

<div>
$$
H_a: at\ least\ one\ \beta_j\ is\ non-zero
$$
</div>

This hypothesis test is performed by computing the `F-statistic`,

<div>
$$
F = \frac{(TSS-RSS)/p}{RSS/(n-p-1)},
$$
</div>

where , as with simple linear regression, $TSS=\sum(y_i - \bar{y})^2$ and $RSS=\sum(y_i - \hat{y}_i)^2$.

If the linear model assumptions are correct, one can show that 

<div>
$$
E\{ RSS/(n-p-1) \} = \sigma^2
$$
</div>

and that, provided $H_0$ is true,

<div>
$$
E\{ (TSS-RSS)/p \} = \sigma^2
$$
</div>

Hence, when there is **no** relationship between the response and predictors, the `F-statistic` to take on value close to 1; on the other hand, if $H_a$ is true, then $E\{(TSS-RSS)/p\} > \sigma^2$, so we expect `F-statistic` to take on value greater than 1.

How large does the `F-statistic` need to be before we can reject $H_0$ and conclude that there is a relationship?  It turns out the answer depends on the values of $n$ and $p$. 

When $n$ is large, and `F-statistic` that is just a little larger than 1 might still provide evidence against $H_0$. In contrast, a larger `F-statistic` is needed to reject $H_0$ if $n$ is small.

When $H_0$ is  true and the error term $\epsilon_i$ have a normal distribution, the `F-statistic` follows an F-distribution. P-value can be calculated based on the distribution. 

Sometimes, we want to test that a particular subset of $q$ of the coefficients are zero. This corresponding to a null hypothesis

<div>
$$
H_0: \beta_{p-q+1} = \beta_{p-q+2} = \ldots = \beta_p = 0
$$
</div>

where for convinience we have put the variables chose for omission at the end of the list. In this case, we fit a second model that uses all the variables *except* those last $q$. Suppose that the resudual sum of squares for that model is $RSS_0$. Then the appropriate F-statistic is 

<div>
$$
F = \frac{(RSS_0 - RSS)/q}{RSS/(n-p-1)}
$$
</div>


Normally, for each individual predictor a *t-statistic* and a *p-value* were reported. These provide information about whether each individual predictor is related to the response, after adjusting for the other predictors. It turns out that each of these are exactly equivalent to the F-test that omits that single variable from the model, leaving all the others in. So it reports the `partial effect` of  adding that variable to the model.

Given these individual p-values for each variable, why do we need to look at the overall F-statistic? After all, it seems likely that if any one of the p-values for the individual variables is very small, then *at least one of the predictors is related to the response*. However, this logic is flawed, especially when the number of predictors $p$ is large.

For instace, consider an example in which $p = 100$ and $H_0: \beta_1=\beta_2=\ldots=\beta_p=0$ is true, so no variable is truly associated with the response. In this situation, about 5% of the p-values associated with each variable will be below 0.05 by chance. In other words, we expect to see approximately five *small* p-values even in the absence of any true association between the predictors and hte response. In fact, we are almost guaranteed that we will observe at least one p-value below 0.05 by change! Hence if we use the individual t-statistics and associated p-values in order to decide whether or not there is any association between the variables and the response, there is a very high chance that we will incorrectrly concluded that there is a relationship. However, the F-statistic does not suffer from this problem because it adjusts for the number of predictors. Hence, if $H_0$ is true, there only a 5% chance that the F-statistic will result in a p-value below 0.05 regarless of the number of predictors or the number of observations.

The approach of using an F-statisc to test for any association between the predictors and the response works when $p$ is relative *small*, and certainly small compared to $n$. However, sometimes we have a very large number of variables. If $p > n$ then there are more coefficients $\beta_j$ to estimate than overvations from which to estimate them. In this case we cannot even fit the multiple linear regressio model using least squares, so the F-statistic cannot be used, and neither can most the other concepts that we have seen so far in this chapter. When $p$ is large, some of the approaches discussed in the next section, such as `forward selection`, can be used. This *high-dimiensional* setting is discussed in greater detail in Chapter 6.

<u>Two: Deciding on Important Variables</u>

As discussed in the previous section, the first step in a multiple regression analysis is to compute the F-statistic and to determine the associated p-values. If we conclude on the basis of that p-values that at least one of the predictors is related to the response, then it is natural to wonder *which* are the guilty ones! We could look at the individual p-values, but as discussed, if $p$ is large we are likely to make some false discoveries.

It is possible that all of the predictors are associated with the response, but it is more often the case that the response is only related to a subset of the predictors. The task of determining which predictors are associated with the response, in order to fit a single model involving only those predictors, is refered to as `variable selection`.

Ideally, we would like to perform variable selectoiin by trying out a lot of different models, each containing a different subset of the predictors. For instance, if $p=2$, then we can consider four models: (1) a model containing no variables, (2) a model containing $X_1$ only, (3) a model containing $X_2$ only, (4) a model containing both $X_1$ and $X_2$. We can then select the *best* model out of all of the models that we have considered. How do we determine which model is best? Various statistics can be used to judge the quality of a model. These include $Mallow's C_p$, $Akaike\ information\ criterion\ (AIC)$, $Bauesian\ information\ criterion\ (BIC)$, and $adjusted\ R^2$. We can also determine which model is best by plotting various model outputs, such as the residuals, in order to search for patterns.

Unfortunately, there are a total of $2^p$ models that contain subsets of $p$ variables. This means that even for moderate $p$, trying out every possible subset of the predictors is infeasible. For instance, we saw that if $p=2$, then there are $2^2 = 4$ models to consider. But if $p=30$, then we must consider $2^30=1,073,741,824$ models! This is not practical.

We need an automated and efficient approach to choose a smaller set of models to consider. There are three classical approaches for this tasks:

* `Forward selection`. We begin with the *null model* -- a model that contains an intercept but no predictors. We then fit $p$ simple linear regressions and add to the null model the variable that results in the lowes RSS. We then add to that model the variable that results in the lowes RSS for the new tow-variable model. This appraoch is continued until some stopping rule is statisfied.

* `Backward selection`. We start with all variables in the model, and remove the variable with the largest p-value--that is, the variable that is the least statistically significant. The new $(p-1)$-variable model is fit, and the variable with the largest p-value is removed. This procedure continues until a stopping rule is reached. For instance, we may stop when all remaining variables have a p-value below some threshold.

* `Mixed selection`. This is a combination of forward and backward selection. We start with no variables in the model, and as with forward selection, we add the variable that provides the best fit. We continue to add variables one-by-one. However, if at any point the p-value for one of the variables in the model rises above a certain threshold, then we remove that variable from the model. We continue to perform these foreard and backward steps until all variables in the model have a sufficiently low p-value, and all variables outside the model would have a large p-value if added to the model.

Backward selection cannot be used if $p > n$, while forward selection can always be used. Forward selection is a greedy approach, and might include variables early that later become redundant. Mixed selection can remedy this.

<u>Three: Model Fit</u>

Two of the most common numerial measures of model fit are the RSE and $R^2$, the fraction of variance explained. 

Recall that in simple regression, $R^2$ is the square of the correlation of the response and the variable. In multiple linear regression, it turns out that it equals $Cor(Y, \hat{Y})^2$, the square of the correlation between the response and fitted linear model; in fact one property of the fitted linear model is that it maximizes this correlation among all possible linear models.

In general RSE is defined as 

<div>
$$
RSE = \sqrt{\frac{1}{n-p-1}RSS}
$$
</div>

In addition to looking at the RSE and $R^2$ statistics, it can be useful to plot the data. 

<u>Four: Predictoins</u>

Once we have fit the multiple regression model, it is straightforward to apply in order to predict the response $Y$ on the basis of a set of values for the predictors $X_1,X_2,\ldots,X_p$. However, there are three sorts of uncertainty associated with this prediction.

1. The coefficient estimates $\hat{\beta}_0, \hat{\beta}_1, \ldots,\hat{\beta}_p$ is only an estimate for the *true* population regression plane. The inacccuracy in the coefficient estimates is related to the *reducible error*. We can compute a *confidence interval* in order to determine how close $\hat{Y}$ will be to $f(X)$.

2. Of course, in practice assuming a linear modle for $f(X)$ is almost always an approximation of reality, so there is an additional source of potentially reducible error which we call *model bias*. So when we use a linear model, we are in fact estimating the best linear approximation to the true surface. However, here we will ignore this discrepancy, and operate as if the linear model were correct. 

3. Even if we know $f(X)$ -- that is, even if we knew the true values for $\beta_0,
beta_1,\ldots,\beta_p$ -- the response value cannot be predicted perfectly because of the random error $\epsilon$ in the model, which is refered as *irreducible error*. How much will $Y$ vary from $\hat{Y}$? We use `prediction intervals` to answer this question. Prediction invervals are always wider than confidence intevals, because they incorporate both the error in the estimate for $f(X)$ and the uncertainty as to how much an individual point will differ from the population regression plane (the irreducible error).

## Other considerations in the regression Model

### Qualitative Predictors

In practice, often many predictors are qualitative.

<u>Predictors with Only Two Levels</u>
If a qualitative predictor only has two levels, or possible values, then incorporating it into a regression model is simple. We simply create an indicator or *dummy variable* that takes on two possible numerical values. 

<div>
$$
x_i=
\begin{cases}
1 &\text{if ith person is female} \\
0 & \text{if ith person is male}
\end{cases}
$$
</div>

<u>Qualitative Predictors with More than Two Levels</u>

When a qualitative predictor has more than two levels, a single dummy variable cannot represent all possible values. In this situation, we can create additional dummy variables. Multiple dummy variables can be used.

### Extentions of the Linear Model

The standard linear regression model provides interpretable results and works quite well on many real-world problems. However, it makes several highly restrictive assumptions that are often violated in practive. Two of the most important assumptions state that the relationship between the predictors and response are `additive` and `linear`. The additive assumption means that the effect of changes in a predictor $X_j$ on the response $Y$ is independent of the values of the other predictors. The linear assumption states that the change in the response $Y$ due to a one-unit change in $X_i$ is constant, regardless of the value of $X_j$. 

Some common classical approaches for extending the linear model.

<u>Removing the Additive Assumption</u>
 
Consider the standard linear regression model with two variables,

<div>
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \epsilon
$$
</div>

According to this model, if we increase $X_1$ by one unit, then $Y$ will increase by an average of $\beta_1$ units. Notice that the presence of $X_2$ does not alter this statement -- that is, regardless of the value of $X_2$, a one-unit increase in $X_1$ will lead to a $\beta_1$-unit increase in $Y$.

One way of extending this model to allow for interaction effects is to include a third predictor, called an `interaction term`, which is constructed by computing the product of $X_1$ and $X_2$. This results in the model

<div>
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2 + \epsilon
$$
</div>

How does inclusion of this interaction term relax the additive assumption?

Notice the formula can be written as 

<div>
$$
\begin{aligned}
Y &= \beta_0 + (\beta_1 + \beta_3 X_2)X_1 + \beta_2 X_2 + \epsilon \\
&=\beta_0 + \tilde{\beta}X_1 + \beta_2 X_2 + \epsilon
\end{aligned}
$$
</div>

where $\tilde{\beta} = \beta_1 + \beta_3 X_2$. Since $\tilde{\beta}$ changes with $X_2$, the effect of $X_1$ on $Y$ is no longer constant: adjusting $X_2$ will change the impact of $X_1$ on $Y$.

The `hierarchical principle`: if we include an interaction in a model, we should also include the main effects, even if the p-values associated with their coefficients are not significant.

<u>Non-linear Relationship</u>

As discussed previously, the linear regression model assumes a linear relationship between the response and predictors. But in some cases, the true relationship between the response and the predictors may be non-linear. A simple way to directly extend the linear model to accommodate non-linear relationships, using `polynomial regression`. More complex approaches for performing non-linear fits in more general settings will be introduced later.

### Potential Problems

When we fit a linear regression model to a particular data set, many problems may occur. 

1. Non-linearity of the response-predictor relationships

2. Correlation of error terms

3. Non-constant variance of error terms

4. Ourliers

5. High-leverage points

6. Collinearity

In practice, identifying and overcoming these problems is as much an art as a science. 

<u>1. Non-linearity of the Data</u>

The linear regression model assumes that there is a straight-line relationship between the predictors and the response. If the true relationship is far from linear, then virtually all of the conclusions that we draw from the fit are suspect. In addition, the prediction accuracy of the model can be significantly reduced.

`Residual plots` are a useful graphical tool for identifying non-linearity. Given a simple linear regression model, we can plot the residuals, $e_i = y_i - \hat{y}_i$ versus the predictor $x_i$.

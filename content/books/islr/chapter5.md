---
title: "Chapter5 Resampling Methods"
date: 2019-03-01T16:17:55-06:00
draft: 
---
Resampling methods are an indispensable tool in modern statistics. They involve repeatedly drawing samples from a training set and refitting a model of interest on each sample in order to obtain additional information abou the fitted model. For example, in order to estimate the variability of a linear regression fit, we can repeatedly draw different samples from the training data, fit a linear regression to each new sample, and then examine the extent to which the resulting fits differ. 

We will disccuss two of the most commonly used resampling methods, `cross-validatoin` and the `bootstrap`. Both methods are important tools in the practical application of many statistical learning procedures. For example, cross-validation can be used to estimate the test error associated with a given statistical learning method in order to evaluate its performance, or to select the appropriate level of flexibility. The process of evaluating a model's performance is known as *model assessment*, whereas the process of selecting the proper level of flexibility for model is known as *model selection*. The bootstrap is used in several contexts, most cmomonly to provide a measure of accuracy of a parameter estimate or of a given statistical learning method.

## Cross-Validation

Test error rate: Average error that results from using a statistical lerning method to predict the response on a new observation--that is , a measurement that was not used in the training the method. Given a data set, the use of particular statistical learning method is warranted if it results in a low test error. The test error can be easily calculated if a designated test set is available. Unfortunately, this is usually not hte case. In contrast, the training error can be easily calculated by applying the statistical learning method to the obseveations used in its training. As we discussed before, the training error rate often is quite different from the test error rate, and in particular the former can dramatically underestimate the latter.

In the absence of a very large designated test set that can be used to directly estimate the test erro rate, a number of techniques can be used to estimate this quantity using hte available traininig data. Some methods make a mathematical adjustment to the training error rate in order to estimate the test error rate. 

In this section, we instead consider a class of methods that estimate the test error rate by *holding out* a subset of the training observations from the fitting process, and then applying the statistical learning method to those held out observations.

### The Validaton Set Approach

Suppose that we would like to estimate the test error associated with fitting a particular statistical learning method on a set of observations. The `Validation set` approach, is a very simple strategy for this task. It involes randomly dividing the availble set of obserations into two parts, a `training set` and a `validation set` or `hold-out set`. The model is fit on the training set, and the fitted model us used to predict the response for the observations in the validation set. The resulting validation set error rate -- typically, assessed using MSE in the case of quantitative response -- provides an estiamte of the test error rate.

### Leave-One-Out Cross-Validation

*Leave-one-out cross-validation* (LOOCV) is closely related to the validation set approach, but it attempts to address that method's drawbacks.

Like the validation set appraoch, LOOCV involves splitting the set of observations into two parts. However, instead of creating two subsets of comparable size, a single observation $(x_1, y_1)$ is used for validation set, and hte remaining observations  ${(x_2, y_2), \ldots,(x_n, y_n)}$ make up the training set. The statistical learning method is fit on the $n-1$ traiing observations, and a prediction $\hat{y}_1$ is mad for the excluded observation, using its values $x_1$. Since $(x_1, y_1)$ was not used in the fitting process, $MSE_1 = (y1 - \hat{_1})^2$ provides an approximately unbiased estimate for the test error. But even though $MSE_1$ is unbiased for the test error, it is a poor estimate because it is highly variable, since it is based on a single observation $(x_1, y_1)$. 

We can repeat the procedure by selecting $(x_2, y_2)$ for the valdiation data, training the statistical learning procedure on the $n-1$ observations $(x_1, y_1), (x_3, y_3), \ldots,(x_n, y_n)$, and computing $MSE_2=(y_2 - \hat{y}_2)^2$. Repeating this approach $n$ times produces $n$ squared errors, $MSE_1, \ldots, MSE_n$. The LOOCV estimate for the test MSE is the average of these $n$ test error estimates:

<div>
$$
CV_{(n)} = \frac{1}{n}\sum_{i=1}^n MSE_i
$$
</div>

LOOCV has a couple of major advantages over the validation set approach.

1). It has far less bias. In LOOCV, we repeatedly fit the statistical learning method using training sets that contains $n-1$ observations, almost as many as are in the entire data set. This is in contrast to the validation set approach, in which the training set is typically around half the size of the original data set. Consequently, the LOOCV approach tends not to overestiamte the test error as much as the validation approach does.

2). In contrast to the validation approach which will yield different results when applied repeatedly due to randomness in the training/validation set splits, performing LOOCV multiple times always yields the same results: there is no randomness in the training/validation set splits.

LOOCV has the potential to be expensive to implement, since the model has be fit $n$ times. This ca nbe very time consuming if $n$ is large, and if each individual model is slow to fit. With least squares linear or plynomial regression, an amazing shortcut makes the cost of LOOCV the same as that of a single modle fit! 

<div>
$$
CV_{(n)} = \frac{1}{n}\sum_{i=1}^n \Big(\frac{y_i - \hat{y}_i}{1-h_i}\Big)^2
$$
</div>

where $\hat{y}_i$ is the $i$th fitted value from the original least square fit, and $h_i$ is the leverage defined as the following,

<div>
$$
h_i = \frac{1}{n} + \frac{(x_i-\bar{x})^2}{\sum_{i'=1}^n (x_{i'} - \bar{x})^2}
$$
</div>

This is like the ordinary MSE, except the $i$th residual is divided by $1-h_i$. The leverage lies between $1/n$ and 1, and reflects the amount that an observation influences its own fit. Hence the residuals for high-leverage points are inflated in this formula by exactly the rigth amount for this equality to hold.

LOOCV is a very general method, and can be used with any kind of predictive modeling. For example we could use it with logistic regression or linear discriminant analysis. The magic formula does not hold in general, in which case the model has to be regit $n$ times.

### k-Fold Cross-Validation

An alternative to LOOCV is k-fold CV. This approach involves randomly dividing the set of observations into $k$ groups, or folds, of approximately equal size. The first fold is treated as a validation set, and the method is fit on the remaining $k-1$ folds. The mean squared error, $MSE_1$, is then computed on the observations in the held-out fold. This procedure is repeated $k$ times; each time, a different group of obervations is treated as a validation set. This process results in $k$ estimates of the test error, $MSE_1, MSE_2, \ldots, MSE_k$. The k-fold CV estimate is computed by averaging these values, 

<div>
$$
CV_{(k)} = \frac{1}{k}\sum_{i=1}^k MSE_i
$$
</div>

It is not hard to see that LOOCV is a special case of k-fold CV in which $k$ is set to equal $n$. In practice, one typical performs k-fold CV using $k=5$ or $k=10$. The most obvious advantage of k-fold is computational.

### Bias-Variance Trade-Off for k-Fold Cross-Validation

In addition to computaional advantage, a less obvious but potentially more important advantage of k-fold CV is that if often gives more accurate estimates of the test error rate than does LOOCV. 

LOOCV will give approximately unbiased estimates of the test error, since each training set contains $n-1$ observations, which is almost as many as the number of observations in the full data set. And performing k-fold CV, say, $k=5$ or $k=10$ will lead to an intermediate level of bias, since each training set contains $(k-1)n/k$ observations--fewer than in the LOOCV approach, but substantially more than in the validation set approach. Therefore, from the perspective of bias reduction, it is clear that LOOCV is to be preferred to k-fold CV.

However, LOOCV has higher variance than does k-fold CV with $k<n$. Why is this the case? When we perform LOOCV, weare in effect averaging the outputs of $n$ fitted models, each of which is trained on an almost identical set of observations; therefore, these outputs are highly (positively) correlated with each other. In contrast, when we perform k-fold CV with $k<n$, we are averaging the outputs of $k$ fitted models that are somewhat less correlated with each other, since the overlap between the training sets in each model is smaller. Since the mean of many highly correlated quantities has higher variance than does the mean of many quantities that are not as highly correlated, the test error estimate resulting from LOOCV tends to have higher variance than does the test error estimate resulting from k-fold CV.

To summarize, there is a bias-variance trade-off associated with the choice of $k$ in k-fold CV. Typically given these considerations, one perform k-fold CV using $k=5$ or $k=10$, as these values have been shown empirically to yield test error rate estimates that suffer neigher from excessively high bias nor from very high variance.

### Cross-Validation on Classification Problems

Cross-validation can work just as described earlier for the qualitative responses. In stead of using MSE to quantify test error, we will use the number of misclassified observations. For instance, in the classification setting, the LOOCV error rate takes the form

<div>
$$
CV_{(n)} = \frac{1}{n}\sum_{i=1}^n Err_{i}
$$
</div>

where $Err_i = I(y_i \stackrel{!}{=} \hat{y}_i)$. 

## The Bootstrap

The `bootstrap` is a widely applicable and extremely powerful statistical tool that can be used to quantify the uncertainty associated with a given estimator or statistical learning method. As a simple example, the bootstrap can be used to estimate the standard errors of the coefficients from a linear regression fit. The power of the bootstrap lies in hte fact that it can be easily applied to a wide range of statistical learning mthods, including some for which a measure of variability is otherwise difficult to obtain and is not automatically output by statistical software.

Suppose that we wish to invest a fixed sum of money in two financial assets that yield returns of $X$ and $Y$, respectively, where $X$ and $Y$ are random quantities. We will invest a fraction of $\alpha$ of our money in $X$, and will invest the remaining $1-\alpha$ in $Y$. Since there is variability associated with the returns on these two assets, we wish to choose $\alpha$ to minimize the total risk, or variance, of our investment. In other words, we want to minimize $Var(\alpha X + (1-\alpha)Y)$. One can show that the value that minimize the risk is given by 

<div>
$$
\begin{aligned}
\alpha &= \frac{\sigma_{Y}^2 - \sigma_{XY}}{\sigma_{X}^2+\sigma_{Y}^2-2\sigma_{XY}}
\\
where,\\
\sigma_{X}^2 &= Var(X) \\
\sigma_{Y}^2 &= Var(Y) \\
\sigma_{XY} &= Cov(X, Y)
\end{aligned}
$$
</div> 

In reality, the quantities $\sigma_{X}^2$, $\sigma_{Y}^2$ and $\sigma_{XY}$ are unknown. We can compute estimtes for these quantities, $\hat{\sigma}_{X}^2$, $\hat{\sigma}_{Y}^2$ and $\hat{\sigma}_{XY}$, using a data set that contains past measurements for $X$ and $Y$. We can then estimate the value of $\alpha$ that minimizes the variance of our investment.

It is natural to wish to quantify the accuracy of our estimate of $\alpha$. To estimate the standard deviation of $\hat{\alpha}$, we repeated the process of simulating 100 paired observations of $X$ and $Y$, and estiamting $\alpha$ 1000 times. We thereby obtained 1000 estimates for $\alpha$, which we can call $\hat{\alpha}_1, \hat{\alpha}_2,\ldots,\hat{\alpha}_1000$. Based on these values, we can then estimate the $SE(\hat{\alpha})$.

In practice, however, the procedure for estimating $SE(\hat{\alpha})$ outlined above cannot be applied, because for real data we cannot generate new samples from the original population. However, the bootstrap approach allows us to us a computer to emulate the process of obtaining new smaple sets, so that we can estimate the variability of $\hat{\alpha}$ without generating additional samples. Rather than repeatedly obtaining independent data sets from the population, we instead obtain distince data sets by repeatedly sampling observations *from the original data set*.

----
Lab: <a href="/jupyter/Chapter5_Resampling_Lab.html">Chapter5_Resampling_Lab.html</a>

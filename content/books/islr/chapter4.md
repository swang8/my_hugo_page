---
title: "Chapter4 Classification"
date: 2019-02-18T09:19:59-06:00
---
In this chapter, we study approaches for predicting qualitative responses, a process that is known as `classification`. Predicting a qualitative response for an obervation can be refered to as *classifying* that observation, since it involves assigning the observation to a category, or class. On the other hand, often the methods used for classification first predict the probability of each of the categories of a qualitative variable, as the basis for making the classification. In this sense they also behave like regression methods.

We will first discuss three of the most widely used classifiers: `logistic regression`, `linear discriminant anlaysis`, and `K-nearest neighbors`. 

## An overview of Classification

Just as in the regression setting, in the classification setting we have a set of training observations $(x_1,y_1), \ldots,(x_n, y_n)$ that we can use to build a classifier. We want our classifier to perform well not only on the training data, but also on the test observations that were not used to train the classifier.

## Why not Linear Regression

We have stated that linear regression is not appropriate in the case of qualitative response. Why not?

Suppose that we are trying to predict the medical condition of a patient in the emergency room on the basis of her symptoms. In this simplified example, there are three possible diagnoses: stroke, drug overdose, and epileptic seizure. We could consider encoding these values as a quantitative response variable, $Y$, as followes:

<div>
$$
y = 
\begin{cases}
1 \text{ if stroke}; \\
2 \text{ if drug overdoses};\\
3 \text{ if epileptic seizure}.
\end{cases}
$$
</div>

Using this coding, least squares could be used to fit a linear regression model to predict $Y$ on the basis of a set of predictions $X_1,\ldots,X_p$. Unfortuantely, this coding implies an ordering on the outcomes, putting *durg overdose* in between *stroke* and *epileptic seizure*, and insisting that the difference among them are the same. In practice, there is no particular reason that this needs to be the case. For instance, one could choose an equally reasonable coding, 

<div>
$$
y = 
\begin{cases}
1 \text{ if drug overdoses};\\
2 \text{ if epileptic seizure};\\
3 \text{ if stroke}. 
\end{cases}
$$
</div>

which would imply a totally different relationship among the three conditions. Each of these coding would produce fundamentally different linear models that would ultimately lead to different sets of predicitons on test observations.

If the response variable's values did take on a natural ordering, such as $mild, moderate, severe$, and we felt the gap between mild and moderate  was similar to the gap between moderate and severe, then a $1,2,3$ coding would be reasonable. Unfortunately, in general there is no natural way to convert a qualitative response variable with more than two levels into a quantitative response that is ready for linear regression.

For a binary (two level) qualitative response, the situation is better. For instance, perhaps there are only two possiblities for the patient's medical condition: stroke and drug overdose. We then potentially use the dummy variable approach to code the response as follows:

<div>
$$
y = 
\begin{cases}
1 \text{ if stroke}; \\
2 \text{ if  drug overdoses}
\end{cases}
$$
</div>

We could then fit a linear regression to this binary response, and predict drug overdose if $\hat{Y} > 0.5$ and stroke otherwise. In this case, even if we flip the above coding, linear regerssion will produce the same final predictions.

For a binary response with a $0/1$ coding as above, regression by least square does make sense; it can be shown that the $X\hat{\beta}$ ovbtained using linear regression is in fact an estiamte of $Pr(\text{drug overdose}|X)$ in this special case. However, if we use linear regression, some of our estimates might be outside the $[0,1]$ interval, making them hard to interpret as probabilities!

Curiously, it turns out that the classifictions that we get if we use linear regression to predict a binary response will be the same as for the linear discriminant analysis (LDA) procedure.

## Logistic Regression

### The Logistic Model

How should we model the relationship between $p(X)=Pr(Y=1|X)$ and X?

In previous section, we talked about using a linear regression model to represent these probabilities:

<div>
$$
p(X) = \beta_0 + \beta_1 X
$$
</div>

If we use this approach to predict defaul=yes using balance data set, we will have predictions are much bigger than 1 when deal with big balance. These predictions are not sensible, since of course the true probability of default regardless of the credit card balance, must fall between 0 and 1. 

To avoid this problem, we must model $p(X)$ using a function that gives outputs between 0 and 1 for all values of $X$. Many functions meet this description. In logistic regression, we use the `logistic function`,

<div>
$$
p(X)=\frac{e^{\beta_0+\beta_1 X}}{1+e^{\beta_0+\beta_1 X}}
$$
</div>

To fit the model, we use a method called `maximum likelihood`, which we discuss in the next section.

After a bit of manipulation, we find that

<div>
$$
\frac{p(X)}{1-p(X)} = e^{\beta_0+\beta_1 X}
$$
</div>

The quantify $p(x)/[1-p(X)]$ is caled the `odds`, and can take on any value between 0 and $\infty$. Values of the odds close to 0 and $\infty$ indicate very low and very high probabilities, respactively.

By taking the logarithm of both side of the formula, we arrive at

<div>
$$
log\Bigg(\frac{p(X)}{1-p(X)}\Bigg) = \beta_0+\beta_1 X
$$
</div>

The left-hand side is called the `log-odds` or `logit`. We see that the logistic regression model has a logit that is linear in $X$.

### Estimating the Regression Coefficients

The coefficients $\beta_0$ and $\beta_1$ are unknown, and must be estimated based on the available training data. Although we could use non-linear least squares to fit the model, the more general method of `maximum likelihood` is preferred, since it has better statistical properties. 

The basid intuition behind using maximum likelihood to fit a logistic regression model is a s follows: we seek estimates for $\beta_0$ and $\beta_1$ such that the predicted probability $\hat{p}(x_i)$ of default for each individual corresponds as closely as possible to the individual's observed default status. In other words, we try to find $\hat{\beta}_0$ and $\hat{\beta}_1$ such that plugging these estimates into the model for $p(X)$ yields a number close to one for those who defaulted, and a number close to zero for all who did not. This can be formalized using a mathematical equation called a `likelihood funciton`:

<div>
$$
\ell(\beta_0, \beta_1) = \prod_{i:y_i=1} p(x_i) \prod_{i':y_{i'}} (1 - p(x_{i'}))
$$
</div>

The estimate $\hat{\beta}_0$ and $\hat{\beta}_1$ are chosen to *maximize* this likelihood function.

Maximum likelihood is a very general approach that is used to fit many of the non-linear regression models. 

### Making predictions

Once the coefficients have been estimated, it is a simple matter to compute the probability for any given input. 

### Multiple Logistic Regression

We now consider the problem of predicting a binary response using multiple predictors. By analogy with the extension from simple to multiple linear regression, we can generalize as follows:

<div>
$$
log\Big(\frac{p(X)}{1-p(X)}\Big) = \beta_0 + \beta_1 X_1 + \cdots + \beta+p X_p
$$
</div>

Where $X=(X1,\ldots,X_p)$ are $p$ predictors. The equation can be rewritten as 

<div>
$$
p(X) = \frac{e^{\beta_0+\beta_1 X_1 + \cdots + \beta_p X_p}}{1+e^{\beta_0+\beta_1 X_1 + \cdots + \beta_p X_p}}
$$
</div>

### Logistic Regression for >2 Response Classes

The two-class logistic regression models discussed in the previous sections have multiple-calss extensions, but in practice they tend not to be used all that often. Instead, `discriminant anlaysis`, is popular for multiple-class classification.

## Linear Discriminant Analysis, LDA

Logistic regression involves directly modeling $Pr(Y=k|X=x)$ using the logistic function. We now consider an alternative and less direct approach to estimating these probabilities. In this alternative approach, we model the distributino of the predictors $X$ separately in each of the response class (i.e., given $Y$), and then use Bayes' theorem to flid these around into estiamtes for $Pr(Y=k|X=x)$. When these distributions are assumed to be normal, it turns out that the model is very similar in form to logistic regression.

Why do we need another method, when we have logistic regression?

* When the classes are well-separated, the parameter estimates for the logistic regression model are surprisingly unstable. Linear discriminant anlaysis does not suffer from this problem.

* if $n$ is small and the distribution of the predictors $X$ is approximately normal in each of the classes, the linear discriminant model is again more stable then the logistic regression model.

* As mentioned earlier, linear discriminant analysis is popular when we have more than two response classes.

### Using Bayes' Theorem for Classification

Suppose that we wish to classify an observation into one of $K$ classes, where $K \geq 2$. In other words, the qualitative response variable $Y$ can take on $K$ possible distinct and unordered values. Let $\pi_k$ represent the overall or `prior` probability that a randomly chosen ovservation comes from the $k$th class; this is the probability that a given observation is associated with the $k$th category of the response varible $Y$. Let $f_k(X) \equiv Pr(X=x|Y=k)$ denote the `density function` of $X$ for an obervation that comes from the $k$th class. In other words, $f_k(x)$ is relatively large if there is a high probability that an observation in the $k$th class $X \approx x$, and $f_k(x)$ is small if it is very unlikely that an observation in the $k$th class has $X \approx x$. Then Bayes' theorem states that

<div>
$$
Pr(Y=k|X=x) = \frac{\pi_k f_k(x)}{\sum_{l=1}^K \pi_l f_l(x)}
$$
</div>

In accordance with our earlier notation, we will use the abbreviation $p_k (X) = Pr(Y=k|X)$. This suggests that instead of directly computing $p_k(X)$, we can imply plug in estiamtes of $\pi_k$ and $f_k (X)$ into the formula. In general, estimating $\pi_k$ is easy if we have a random sample of $Y$s from the population: we simply compute the fraction of the training observations that belog to the $k$th class. However, estimating $f_k(X)$ tends to be more challenging, unless we assume some simple forms for these density. We refer to $p_k(x)$ as the `posterior` probability that an observation $X=x$ belongs to the $k$th class. That is, it is the probability that the observation belongs to the $k$th class, given the predictor value for that observation.

We know that the Bayes classifier, which classifies an observation to the class for which $p_k(X)$ is largest, has the lowest possible error rate out of all classifiers. Therefore, if we can find a way to estimate $f_k(X)$, then we can develop a classifier that approximates the Bayes classfier. 

### Linear Discriminant Analysis for p = 1

For now, assume that $p = 1$--that is, we have only one predictor. We would like to obtain an estimate for $f_k (x)$ that we can plug into the formula in order to estimate $p_k(x)$. We will then classify an observation to the class for which $p_k(x)$ is greatest. In order to estiamte $f_k(x)$, we will first make some assumptions about its form.

Suppose we assume that $f_k(x)$ is *normal* or *Gaussian*. In the one-dimensional setting, the normal density takes the form 

<div>
$$
f_k(x) = \frac{1}{\sqrt{2\pi}\sigma_k}\exp\Big(-\frac{1}{2\sigma_k^2}(x - \mu_k)^2\Big),
$$
</div>

where $\mu_k$ and $\sigma_k^2$ are the mean and variance parameters for the $k$th class.

For now, lwet us further assume that $\sigma_1^2=\ldots=\sigma_k^2$: that is, there is a shared variance term across all $K$ classes, which for simplicity we can denote by $\sigma^2$. Plugging the $f_k(x)$ into the formula, we find that

<div>
$$
p_k(x) = \frac{\pi_k \frac{1}{\sqrt{2\pi}\sigma}\exp\Big(-\frac{1}{2\sigma^2}(x - \mu_k)^2\Big)}{\sum_{l=1}^K \pi_l\frac{1}{\sqrt{2\pi}\sigma}\exp\Big(-\frac{1}{2\sigma^2}(x - \mu_l)^2\Big)}
$$
</div>

Taking the log of the formula and rearranging the terms, we will see that this is equivaletn to assigning the observations to the class for which

<div>
$$
\delta_k(x) = x \cdot \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + log(\pi_k)
$$
</div>

is largest. 

For instance, if $K = 2$ and $\pi_1 = \pi_2$, then the Bayes classifier assigns an observation to class 1 if $2x(\mu_1 - \mu_2) \gt \mu_1^2 - \mu_2^2$, and to class 2 otherwise. In this case, the Bayes decision boundary corresponds to the point where 
<div> 
$$
x = \frac{\mu_1^2 - \mu_2^2}{2(\mu_1 - \mu_2)}=\frac{\mu1 + \mu2}{2}
$$
<div> 

In practice, the `linear discriminant analysis` method approximates the Bayes classifier by plugging estimates for $\pi_k,\mu_k$, and $\sigma^2$. The following estimates are used in particular:

<div>
$$
\begin{aligned}
\hat{\mu}_k &= \frac{1}{n_k}\sum_{i:y_i=k} x_i\\

\hat{\sigma}^2 &= \frac{1}{n-k}\sum_{k=1}^K\sum_{i:y_i=k} (x_i - \hat{\mu}_k)^2
\end{aligned}
$$
</div>

where $n$ is th total number of training observations, and $n_k$ is the number of training ovservations in the $k$th class. The estimate for $\mu_k$ is simply the average of all the training observations from the $k$th class, which $\hat{\sigma}^2$ can be seen as a weighted average of the sample variances for each of the $K$ classes. Sometimes we have knowledge of the class membership probabilities $\pi_1,\ldots,\pi_k$, which can be used directly. In the absence of any additional information, LDA estimates $\pi_k$ using the proportion of the training observations that belong to the $k$th class. In other words,

<div>
$$
\hat{\pi}_k = n_k / n
$$
</div>

The LDA classifier plugs the estimates and assigns an observation $X = x$ to the class for which

<div>
$$
\hat{\delta}_k(x) = x \cdot \frac{\hat{\mu}_k}{\hat{\sigma}^2} - \frac{\hat{\mu}_k^2}{2\hat{\sigma}^2} + log(\hat{\pi}_k)
$$
</div>

is largest. The word *linear* is the classifier's name stems from the fact that the *discriminant function* $hat{\delta_k(x)}$ are linear functions of $x$ (as opposed to a more complex function of $x$).

To reiterate, the LDA classifier results from assuming that the observations within each class come from a normal distribution with a class-specific mean vaector and a common variance $\sigma^2$, and pluggin estiamtes for these parameters into the Bayes classifier. In the following section, we will consider a less stringent set of assumptions, by allowing the observations in the $k$th class to have a class-specific variance, $\sigma_k^2$.

### Linear Discriminatnt Analysis for p > 1

We now extend the LDA classifier to the case of multiple predictors. To do this, we will assume that $X = (X_1,X_2,\ldots,X_p)$ is drawn from a *multiple Gaussian* or multivariate normal distribution, with a class-specific mean vector and a common covarince matrix.

The multivariate Gaussian distribution assemes that each individual predictor follows a one-dimensional normal distribution, with some correlation between each pair of predictors.

To indicate that a p-dimensional randome variable $X$ has a multivariate Gaussian distribution, we write $X \sim N(\mu, \Sigma)$. Here $E(X) = \mu$ is the mean of $X$ (a vector with $p$ components), and $Cov(X)=\Sigma$ is the $p \times p$ covariance matrix of $X$. Formally, the multivariate Gaussian density is defined as

<div>
$$
f(x)=\frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}\exp\Big(-\frac{1}{2}(x-\mu)^T\Sigma^{-1} (x-u)\Big)
$$
</div>

In the ase of $p > 1$ predictors, the LDA classifier assumes that the observations in the $k$th class are drawn from a multivariate Gaussian distribution $N(\mu_k, \Sigma)$, where $\mu_k$ is a class-specific mean vector, and $\Sigma$ is a covariance matrix that is common to all $K$ classes. Plugging the density function for the $k$th class, $f_k(X = x)$, and performing little of algebra reveals the Bayes classifier assigns an observation $X=x$ to the class for which 

<div>
$$
\delta_k(x) = X^T\Sigma^{01} \mu_k - \frac{1}{2}\mu_k^T \Sigma^{-1}\mu_k + log\pi_k
$$
</div>

is the largest.

### Quadratic Discriminant Analysis

As we have discussed, LDA assumes that the observations within wach class are drawn from a multivariate Gaussian distribution with a class-specific mean vector and a covariance matrix that is cmommon to all $K$ classes. `Quadratic discriminant analysis` (QDA) provides an alternative approach. Like LDA, the QDA classifier results from assuming that the observations from each class are drawn a Gaussian distribution, and plugging estiamtes for the parameters into Bayes' theorem in order to perform prediction. However, unlike LDA, QDA assumes that each class has its own covariance matrix. That is, it assumes that an observation from the $k$th class is of the form $X \sim N(\mu_k, \Sigma_j)$, where $\Sigma_k$ is a covariance matrix for the $k$th class. Under this assumption, the Bayes classifier assigns an observation $X = x$ to class for which 

<div>
$$
\begin{aligned}
\delta_k(x) &= -\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1}(x - \mu_k) - \frac{1}{2}log|\Sigma_k|+log\pi_k \\
&=-\frac{1}{2}x^T\Sigma_k^{-1}x + x^T\Sigma_k^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma_k^{-1}-\frac{1}{2}log|\Sigma_k| + log\pi_k
\end{aligned}
$$
</div>

is largest. So the QDA classifier involves plugging estimates for $\Sigma_k, \mu_k$ and $\pi_k$. The quantity $x$ appears as a quadratic function. This is where QDA gets its name.

Why does it matter whether or not we assume that the $K$ classes share a common covariance matrix? In other words, why would one prefer LDA to QDA or vice-versa? The answer lies in the bias-variance trade-off. When there are $p$ predictors, then estiamting a covariance matrix requires estimating $p(p+1)/2$ parameters. QDA estimates a separate covariance matrix for each class, for a total of $Kp(p+1)/2$ parameters. With 50 predictors this is some multiple of 1225, which is a lot of parameters. By instead assuming that the $K$ classes share a common covariance matrix, the LDA model becomes linear in $x$, which means there are $Kp$ linear coefficients to estimate. Consequently, LDA is a much less flexible classifier than QDA, and so has substantially lower variance. This can potentially lead to improved prediction performance. But there is a trade-off, if LDA's assumption that the $K$ class share a common covariance matrix is badly off, then LDA can suffer from high bias. ROughly speaking, LDA tends to be a better bet than QDA if there are relatively few training obsrvations and so reducing variance is crucial. In contrast, QDA is recommended if the training set is very large, so that the variance of the classifier is not a major concern, or if the assumption of a common variance matrix for the $K$ classes is clearly untenable.

## A Comparison of Classification Methods

We have considered three different classification approaches: logistic regression, LDA, and QDA. In chapter 2, we also discussed the K-nearest neighbors (KNN) method. We now consider the types of scenarios in which on approach might dominate the others.

Though their motivations differ, the logistic regression and LDA methods are closely connected. Consider the two-class setting which $p=1$ predictor, and let $p_1(x)$ and $p_2(x)=1-p_1(x)$ be the probability that the observation $X=x$ belongs to class 1 and class 2, respectively. In the LDA framework, we can see that the log odds is given by 

<div>
$$
log\Big(\frac{p_1(x)}{1-p_1(x)}\Big) = log\Big(\frac{p_x(x)}{p_2(x)}\Big) = c_0 + c_1 x
$$
</div>

where $c_0$ and $c_1$ are functions of $\mu_1$, $\mu_2$ and $\sigma^2$. We also know that in logistic regression,

<div>
$$
log\Big(\frac{p_1}{1-p_1}\Big) = \beta_0 + \beta_1 x
$$
</div>

Both are linear functions of $x$. Hence, both logistic regression and LDA produce linear decision boundaries. The only difference between the two approaches lies in the fact that $\beta_0$ and $\beta_1$ are estimated using maximum likelihood, where $c_0$ and $c_1$ are computed using the estimated mean and variance from a normal distribution. This same connection between LDA and logistic regression also holds for multidimensinal data with $p > 1$.

Since logistic regression and LDA differ only in their fitting procedures, one might expect the two approaches to give similar results. This is often, but not always, the case. LDA assumes that the observations are drawn from a Gaussian distribution with a common covariance matrix in each class, and so can provide some improvements over logistic regression when this assumption approximately holds. Conversely, logistic regression can outperform LDA if these Gaussian assumptions are not met.

Recall that KNN takes a completely different approach from the classifier seen this chapter. In order to make a prediction for an observation $X = x$, the $K$ training ovservations that are closest to $x$ are identified. Then $X$ is assigned to the class to which the plurality of these observations belong. Hence KNN is a completely non-parametric approach: no assumptions are made about the shape of the decision boundary. Therefore, we can expect this approach to dominate LDA and logistic regression when the decision boundary is highly non-linear. On the other hand, KNN does not tell us which predictors are important; we don't get a table of coefficients.

Finally, QDA serves as a compromise between the non-parametric KNN method and the linear LDA and logistic regression approaches. Since QDA assumes a quadratic decision boundary, it can accurately model a wider range of problems than can the linear methods. Though not as flexible as KNN, QDA can perform better in the presence of a limited number of training observations because it does make some assumptions abou the form of the decision boundary.

No one method will dominate the others in every situatioin. When the true decision boundaries are linear, then the LDA and logistic regression appraoches will tend to perform well. When the boundaries are moderately non-linear, QDA may give better results. Finally for much more complicated decision boundaries, a non-parametric approach such as KNN can be superio. But the level of smoothness for a non-parametric approach must be chosen carefully.

Finally, we cann that in the regression setting we can accommodate a non-linear relationship between the predictors and the response by performing regression using transformations of the predictors. A similar approach could be taken in the classification setting. For instance, we could create a more flexible version of logistic regression by including $X^2, X^3$ and even $X^4$ as predictors. This may or may not improve logistic regression performance, depending on whether the increase in variance due to the added flexibility is offset by a sufficiently large reduction in bias. We could do hte same for LDA. If we add all possible quadratic terms and cross-products to LDA, the form of the model would be the same as the QDA model, although the parameter estimates would be different. This device allows us to move somewhere between an LDA and a QDA model.



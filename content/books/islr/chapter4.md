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

The coefficients $\beta_0$ and $\beta_1$ are unknown, and must be estimated based on the available training data. Althought\ we could use non-linear least squares to fit the model, the more general method of `maximum likelihood` is preferred, since it has better statistical properties. 

The basid intuition behind using maximum likelihood to fit a logistic regression model is a s follows: we seek estimates for $\beta_0$ and $\beta_1$ such that the predicted probability $\hat{p}(x_i)$ of 

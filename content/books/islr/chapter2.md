---
title: "Chapter2"
date: 2019-02-01T14:46:09-06:00
tags: ['islr', 'statistics learning', 'R']
---
# Statistical Learning
## What is statistial learning
Suppose we observe a quantitative response $Y$ and $p$ different predictors, $X_1, X_2, \ldots,X_p$ . We assume that there is a relationship between $Y$ and $X=(X_1, X_2,\ldots,X_p)$, which can be written as

<div>
$$
Y=f(X)+\epsilon
$$
</div>

Here $f$ is some fixed but unknown fucntion of $X_1,X_2,\ldots,X_p$, and $\epsilon$ is a random `error term`, which is independent of $X$ and has mean `zero`. In this formulation, $f$ represents the `systematic` informationa $X$ provides about $Y$.

### Why estimate $f$ ?
Two reasons: **prediction** and **inference**.

#### <u>Prediction</u>
In many cases, a set of input $X$ are readily available, but the output $Y$ can not be easily obtained. In this setting, since the error term averages to zero, we can predict $Y$ using

<div>
$$
\hat{Y} = \hat{f}(X)
$$
</div>

Where $\hat{f}$ represents our estimate for $f$, and $\hat{Y}$ represents the resulting predictio for $Y$. In this setting, $\hat{f}$ is oftern treated as a *black box*, in the sense that one is not typically concerned with the exact form of $\hat{f}$, provided that it yields accurate predictions for $Y$.

`reducible error` and `irreducible error`

The accuray of $\hat{Y}$ as prediction for $Y$ depends on two quantities, which we will call them *reducible error* and *irreducible error*. In general, $\hat{f}$ will not be a perfect estimate for $f$, and this inaccuracy will introduce some error. This error is *reducible* because we can potentially improve the accuracy of $\hat{f}$ by using the most appropriate statistical learning techs to estimate $f$. However, even if it were possible to form a perfect etimate for $f$, so that our estimated response took the form $\hat{Y}=f(X)$, our prediction would still have some error in it!! **Why?** This is because $Y$ is also a function of $\epsilon$, which, by definition, cannot be predicted using $X$. Therefore, variables associated with $\epsilon$ also affects the accuracy of our predictions. This is known as the *irreducible* error.

Why is the irreducible error larger than zero? 
The quantify $\epsilon$ may contain unmeasured variables that are useful in predicting $Y$: since we don't measure them, $f$ cannot use them for its prediction. The euatity $\epsilon$ may also contain unmeasurable variation. For example, the risk of an adverse reaction might vary for a given patient on a given ay, depending on manufacturing variation in the drug itself or the patient's general feeling of well-being on that day.

Consider a given estimate $\hat{f}$ and a set of predictors $X$, which yields the prediction $\hat{Y}=\hat{f}(X)$. Assume for a moment that both $\hat{f}$ and $X$ are fixed. Then, it is easy to show that

<div>
$$
\begin{aligned}
E(Y-\hat{Y})^2 &= E[f(X) + \epsilon - \hat{f}(X)]^2  \\
&=\underbrace{[f(X) - \hat{f}(X)]^2}_{reducible} + \underbrace{Var(\epsilon)}_{irreducible} 
\end{aligned}
$$
</div>

Where $E(Y - \hat{Y})^2$ represents the average, or `expected value`, of the squared difference between the predicted and actual value of $Y$, and Var($\epsilon$) represents the **variance** associated with the error term $\epsilon$.

Keep in mind that the **irreducible error** will always provide an upper bound on hte accuracy of our prediction for $Y$. The focus would be estimating of $f$ with the aim of minimizing the **reducible error**.


#### <u>Inference</u>
We are often interested in understanding the way that $Y$ is affected as $X_1,X_2,\ldots,X_p$ change. In this situation we wish to estimate $f$, but our goal is not necessarily to make predictions for $Y$. We instead want to understand the relationship between $X$ and $Y$. More specifically, to understand how $Y$ changes as a function of $X_1,X_2,\ldots,X_p$.

Now $f$ cannot be treated as a *black box*, because we need to know its exact form. In this setting, one may be interested in answering the following questions:

* *<u>Which predictors are associated with the response?</u>* It's often the case that only a small fraction of the repdictors are substantially associated with $Y$. Identifying a few *important* predictors among a large set of possible variables can be very useful.

* *<u>What is the relationship between the response and each predictor?</u>* Some predictors may have a positive relationship with $Y$, while others may have opposite relationships. Depending on the complexity of $f$, the relationship between the response and a given predictor may also depend on the valurs of the other predictors.

* *<u>Can the relationship between $Y$ and each predictor be adequately summarized using a linear equation, or is the relatioship more complicated?</u>* Historically, most mthods for estimating $f$ have taken a linear form. In some situations, such an assumption is resonable or even desirable. But often the true relationship is more complicated, in which case a linear model may not provide an accurate representation of the relationship between the input and output variables.

### How do we estimate $f$ ?
We will explore many linear and non-linear approaches for estimating $f$. However, these methods generally share certain characteristics. 

**Training data**, data that will be used to train, or teach, our method how to estimate $f$.

The goal is to apply a statistical learning method to the *training data* in order to estimate the unknown function $f$. In other words, we want to find a funtion $\hat{f}$ that $Y\approx\hat{f}(X)$ for any given observation $(X, Y)$. 

Broadly speaking, most statitical learning methods for this task can be characterized as either `parametric` or `non-parametric`.

#### <u>Parametric Methods</u>
Parametric methods involve a two-step model-based approach.

1. First, we make an assumption about the functional form, or shape, of $f$. For example, one very simple assumption is that $f$ is linear in $X$:

<div>
$$
f(X)=\beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} + \cdots+\beta_{p}X_{p}
$$
</div>

This is *linear model*. Once we have assumed that $f$ is linear, the problem of estimation $f$ is greatly simplified. Instead of having to estimate an entirely arbitrary p-dimentional function $f(X)$, one only needs to estimate the *p+1* coefficients $\beta$.

2. After a model has been selected, we need a procedure that uses the training data to *fit* or *train* the model. In the case of the linear model, we need to estimate the parameters $\beta_{0}$ , $\beta_{1}$ , $\ldots$ , $\beta_{p}$. That is, we want to find values of these parameters such that 

<div>
$$
Y \approx \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} + \cdots + \beta_{p}X_{p}
$$
</div>












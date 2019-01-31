---
title: "Introduction"
date: 2019-01-30

---

# An overview of statistial learning
*Statistical learning* refers to a vast set of tools for understanding
data.

Two categories: *supervised* and *unsupervised*.

<code>Supervised</code>: Build models based on **known** input and output data, then use the model
for prediction or estimation.

<code>Unsupervised</code>: There are inputs but no supervised outputs. We can learn
relationships and structures from such data.

## Notation and simple algebra

Let the $X$ denotes a matrix. $X_{ij}$ represents the value of column
$j$ and row $i$.

<div>
$$
X = 
\left(\begin{array}{cc}
x_{11} &x_{12} &\cdots &x_{1p}\\ 
x_{21} &x_{22} &\cdots &x_{2p}\\
\ldots &\ldots &\ldots &\ldots\\
x_{n1} &x_{n2} &\cdots &x_{np}
\end{array}\right)
$$
</div>

## get the R package
```r
install.packages("ISLR")
```



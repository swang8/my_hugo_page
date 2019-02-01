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

Let the $X$ denotes a matrix. $X_{ij}$ represents the value of row $i$ and column $j$.

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


For the rows of $X$, wich we write as $x_1, x_2, ..., x_n$ . 

<div>
$$
x_i = \begin{pmatrix}
x_{i1} \\
x_{i2} \\
\vdots
x_{ip}
\end{pmatrix}
$$
</div>

Vectors are by default represented as columns. We use $X_1$, $X_2$, $\ldots$, to represent the columns of $X$.

<div>
$$

X_j = \begin{pmatrix}
x_{1j} \\
x_{2j} \\
\vdots \\
x_{nj}
\end{pmatrix}

$$
</div>

Using this notation, the matirx $X$ can be written as:

<div>
$$
X = \left(X_1 \space X_2 \space  \cdots \space  X_p\right)
$$
</div>

or

<div>
$$

X = \begin{pmatrix}
x_{1}^T \\
x_{2}^T \\
\vdots \\
x_{n}^T
\end{pmatrix}

$$
</div>

The $^T$ notation denotes the `transpose` of a matrix.

We use $y_i$ to denote the $i$ th observation of the variable on which we wish to make predictions. Hence we wirte the set of all $n$ observations in vector format as 

<div>
$$
y = \begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{pmatrix}
$$
</div>

The out observed data consits of {$ (x_1,y_1),(x_2,y_2),\ldots ,(x_n,y_n)$}, where each $x_i$ is a `vector` of length $p$.

Occationally we will want to indicate the dimension of a particular object. 

To indicate that an object is a scalar: $a \in \mathbb{R}$. 

To indicate that it is avector of length $k$: $a \in \mathbb{R}^k$. 

To indicate that an object is a $r \times s$  matrix: $ A \in \mathbb{R}^{r \times s}$.

The product of matrix $A$ and matrixt $B$ is denoted $AB$.

<div>
$$
A = \begin{pmatrix}
1 &2 \\
3 &4
\end{pmatrix} 

and \space 

B=\begin{pmatrix}
5 &6\\
7 &8
\end{pmatrix}
$$
</div>

Then

<div>
$$
AB = \begin{pmatrix} 1 &2 \\ 3 &4 \end{pmatrix} \begin{pmatrix} 5 &6 \\ 7 &8 \end{pmatrix}
=\begin{pmatrix}
1 \times 5 + 2 \times 7  & 1 \times 6 + 2 \times 8 \\
3 \times 5 + 4 \times 7  & 3 \times 6 + 3 \times 8 \\
\end{pmatrix}
$$
</div>

## get the R package
```r
install.packages("ISLR")
```



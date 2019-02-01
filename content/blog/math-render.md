---
title: "Math render problem"
date: 2019-01-31T12:19:59-06:00
tags: ['katex', 'tips']
---

I have encountered a cople of problems when trying to show math formula in markdown files using $\KaTeX$.

### Issues:

#### Inline formula rendering is not working

Solution: add the following lines to the file `themes/beautifulhugo/layouts/partials/head_custom.html`.

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/katex.min.css" integrity="sha384-9eLZqc9ds8eNjO3TmqPeYcDj8n+Qfa4nuSiGYa6DjLNcv9BtN69ZIulL9+8CqC9Y" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/katex.min.js" integrity="sha384-K3vbOmF2BtaVai+Qk37uypf7VrgBubhQreNQe9aGsz9lB63dIFiQVlJbr92dw2Lx" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/contrib/auto-render.min.js" integrity="sha384-kmZOZB5ObwgQnS/DuDg6TScgOiWWBiVt0plIRkZCmE6rDZGrEOQeHM5PcHi+nyqe" crossorigin="anonymous"></script>
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
               delimiters: [
                   {left: "$$", right: "$$", display: true},
                   {left: "\\[", right: "\\]", display: true},
                   {left: "$", right: "$", display: false},
                   {left: "\\(", right: "\\)", display: false}
               ]
        });
    });
</script>

```

#### Somehow, the code block `$$...$$` was not recognized by $\KaTeX$

Solution: adding a `div` tag for the code black seems to solve this issue.

```html
<div>
$$
Y =
\left(\begin{array}{cc}
x_{11} &x_{12} &\cdots &x_{1p}\\
x_{21} &x_{22} &\cdots &x_{2p}\\
\ldots &\ldots &\ldots &\ldots\\
x_{n1} &x_{n2} &\cdots &x_{np}
\end{array}\right)
$$
</div>

<div>
$$
Z = \begin{pmatrix}
a &b\\
c &d
\end{pmatrix}
$$
</div>

```

<div>
$$
Y =
\left(\begin{array}{cc}
x_{11} &x_{12} &\cdots &x_{1p}\\
x_{21} &x_{22} &\cdots &x_{2p}\\
\ldots &\ldots &\ldots &\ldots\\
x_{n1} &x_{n2} &\cdots &x_{np}
\end{array}\right)
$$
</div>

<div>
$$
Z=\begin{pmatrix}
a &b\\
c &d
\end{pmatrix}
$$
</div>






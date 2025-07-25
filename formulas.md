# Overlap Integral Calculation for Normal Modes

## Some General Definitions

**3D Gaussian Function**

Given by 

$$f_i(r) = a_i \exp\left(-\frac{(r-b_i)^T(r-b_i)}{2c_i^2}\right)$$ 

where:

- $a_i$ is the amplitude,
- $b_i$ is the center,
- $c_i$ is the width.

**Gaussian Product Theorem**

The Gaussian Product Theorem states that the product of two Gaussian functions is also a Gaussian function. We compute

$$f_1(r)f_2(r) = a_1 a_2 \exp\left(-\frac{(r-b_1)^T(r-b_1)}{2c_1^2}\right) \exp\left(-\frac{(r-b_2)^T(r-b_2)}{2c_2^2}\right)$$

we can focus first on the exponent inside our Gaussian function:

$$ - \frac{(r-b_1)^T(r-b_1)}{2c_1^2} - \frac{(r-b_2)^T(r-b_2)}{2c_2^2} = - \frac{c_2^2(r-b_1)^T(r-b_1) + c_1^2(r-b_2)^T(r-b_2)}{2c_1^2c_2^2}$$



Expanding the terms we get

$$ = - \frac{c_2^2(r^Tr - 2b_1^Tr + b_1^Tb_1) + c_1^2(r^Tr - 2b_2^Tr + b_2^Tb_2)}{2c_1^2c_2^2}$$

and combine the same terms

$$ = - \frac{(c_1^2 + c_2^2)r^Tr - 2c_1^2b_1^Tr - 2c_2^2b_2^Tr + c_1^2b_1^Tb_1 + c_2^2b_2^Tb_2}{2c_1^2c_2^2}$$


**Overlap Integral**

As $r=(x,y,z)$ is a position vector and $b_i = (b_{ix}, b_{iy}, b_{iz})$ is the center of the Gaussian, the overlap integral is defined as:


$$S= \int_{\mathbb{R}^3} f_i(r) f_j(r) dr = \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} f_i(x,y,z) f_j(x,y,z) dx dy dz$$

We can use the Gaussian Product Theorem and know that the integral of a general Gaussian function is given by $\int_{-\infty}^{\infty} \exp(-\alpha u^2) du = \sqrt{\frac{\pi}{\alpha}}$.

This leads us in the end to the expression:

$$S = a_1 a_2 \left(\frac{2\pi c_1^2 c_2^2}{c_1^2 + c_2^2}\right)^{3/2} \exp\left(-\frac{(b_1 - b_2)^T(b_1 - b_2)}{2(c_1^2 + c_2^2)}\right)$$

**Normalization to the Volume of the VDW Sphere**

In order to determine the parameters $a_i$ for our Gaussian functions we have to normalize the Gaussian function to the volume of the VDW sphere. Mathematically this means:

$$\int_{\mathbb{R}^3} f_i(r) dr = V_{vdw} = \frac{4}{3} \pi r_{vdw}^3$$

This leads us after some tedious calculations to the expression:

$$a_i = \frac{r^3_{vdw}}{3\sqrt{2\pi} c_i^3}$$

As we model the Gaussian Sphere at the moment we take $c_i = r_{vdw}$, which leads to:

$$a_i = \frac{r^3_{vdw}}{3\sqrt{2\pi} r_{vdw}^3} = \frac{1}{3\sqrt{2\pi}}$$


## Determining the Volume Change 

The following figure shows two atoms with centers $b_1$ and $b_2$ which are distanced by $d = ||b_1 - b_2||$. We now want to calculate the volume change of the VDW spheres if these atoms are displaced by some vectors $u_1$ and $u_2$. In reality this vectors will the be the normal modes of the molecule.

![alt text](image.png)

After the displacements we get new centers $b_1' = b_1 + u_1$ and $b_2' = b_2 + u_2$ we can insert this into the overlap integral:

$$S(b_1',b_2') = a_1 a_2 \left(\frac{2\pi c_1^2 c_2^2}{c_1^2 + c_2^2}\right)^{3/2} \exp\left(-\frac{(b_1' - b_2')^T(b_1' - b_2')}{2(c_1^2 + c_2^2)}\right)$$

$$S(b_1 + u_1, b_2 + u_2) = a_1 a_2 \left(\frac{2\pi r_{vdw}^4}{2r_{vdw}^2}\right)^{3/2} \exp\left(-\frac{(b_1 - b_2 + u_1 - u_2)^T(b_1 - b_2 + u_1 - u_2)}{2(c_1^2 + c_2^2)}\right)$$

To simplify this we denote $S_0 = a_1a_2 \left(\frac{2\pi r_{vdw}^4}{2r_{vdw}^2}\right)^{3/2}$

$$S(b_1 + u_1, b_2 + u_2) = S_0 \exp\left(-\frac{(b_1 - b_2 + u_1 - u_2)^T(b_1 - b_2 + u_1 - u_2)}{2(c_1^2 +c_2^2)}\right)$$

### Directional Derivative

We now want a formula to estimate the rate at which our overlap integral changes in a particular direction. This is given by the directional derivative, Because our function is differentiable anyways, we can write the directional derivative as:

$$\nabla_v f(x) = \nabla f(x) \cdot v$$

where $v$ is the direction in which we want to calculate the derivative

### Calculation of the Gradient

Define $r = b_1 - b_2 + u_1 - u_2$ then our overlap integral can be written as:

$$S(b_1 + u_1, b_2 + u_2) = S_0 \exp\left(-\frac{r^Tr}{2(c_1^2+c_2^2)}\right)$$

**Gradient $\nabla_{u_1}S$**

$$\nabla_{u_1}S = S_0 \exp\left(-\frac{r^Tr}{2(c_1^2+c_2^2)}\right) \cdot \nabla_{u_1}\left(-\frac{r^Tr}{2(c_1^2+c_2^2)}\right)$$

Where we can now use $\nabla_{u_1}r = \nabla_{u_1}(b_1 - b_2 + u_1 - u_2) = e_1$ (the unit vector in the direction of $u_1$), which implies $\nabla r = 2r$

$$\nabla_{u_1}S = S\cdot(-\frac{2r}{2(c_1^2+c_2^2)}) = -\frac{Sr}{c_1^2 + c_2^2}$$

**Gradient $\nabla_{u_2}S$**

Here we get exactly the same result but with a different sign:

$$\nabla_{u_2}S = S\cdot(-\frac{2r}{2(c_1^2+c_2^2)}) = \frac{Sr}{c_1^2 + c_2^2}$$

Already for this simple model this gives us a approximation of the change in the overlap we have to calculate, for small displacments

$$\Delta S \approx \nabla_{u_1}S + \nabla_{u_2}S$$


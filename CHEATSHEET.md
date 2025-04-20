# CHEATSHEET: Your ML Handbook & Q&A

> A logically structured, topic-wise reference for ML and Data Science concepts, crafted from your questions and answers.

---

## Table of Contents

1. [Linear Algebra Essentials](#linear-algebra-essentials)
2. [NumPy Core Operations](#numpy-core-operations)
3. [Pandas Quick Reference](#pandas-quick-reference)
4. [Key Machine Learning Concepts](#key-machine-learning-concepts)

---

## 1. Linear Algebra Essentials

### Scalars vs Vectors

- **Scalar**: Single number (e.g., learning rate `η = 0.01`).
- **Vector**: List of numbers indicating magnitude and direction (e.g., `[3, 4]`).
- **Numeric Vector**: Vector with numeric elements (int/float) suitable for linear algebra.

### Vector Magnitude (Norms)

- **L₂ (Euclidean)**:
  \[
  \|v\|_2 = \sqrt{\sum_i v_i^2}
  \]
  - Distance from the origin to vector’s tip.
  - Preferred norm in Euclidean contexts (like K-means).

- **L₁ (Manhattan)**:
  \[
  \|v\|_1 = \sum_i |v_i|
  \]
  - Sum of absolute vector components.

- **L₃ Norm**:
  \[
  \|v\|_3 = \left(\sum_i |v_i|^3
ight)^{1/3}
  \]

### Unit Vectors

- Vector of length 1 in the direction of original vector \( v \):
  ```python
  v_hat = v / np.linalg.norm(v)
  ```

### Dot Product & Alignment

- **Dot Product Formula**:
  \[
  a \cdot b = \|a\|\|b\|\cos	heta = \sum_i a_i b_i
  \]

- **Geometric meaning**:
  - Positive dot → acute angle, vectors aligned similarly.
  - Zero dot → orthogonal vectors (90°).
  - Negative dot → obtuse angle, vectors pointing opposite.

- **Cosine Similarity**:
  ```python
  cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
  ```

- **Q & A**:
  - **Alignment**: Measure of how closely two vectors point in the same direction, indicated by the sign of the dot product.
  - **Dot product of different shapes**: Not possible, shapes must match.
  - **Cosine similarity of -1**: Vectors are perfectly opposite (180° alignment).

### Vector Projection

- **Projection Formula**:
  \[
  	ext{proj}_v(u) = rac{u \cdot v}{v \cdot v} v
  \]

- **Scalar projection** (component of \( u \) along \( v \)):
  \[
  	ext{comp}_v(u) = rac{u \cdot v}{\|v\|} = \|u\|\cos	heta
  \]

- **Residual vector** (component perpendicular to \( v \)):
  \[
  r = u - 	ext{proj}_v(u)
  \]

- **Distances in projection**:
  - Along-vector distance (scalar proj length): `||proj_v(u)||`.
  - Perpendicular distance (residual length): `||u - proj_v(u)||`.

### Vector Addition

- **Formula**:
  \[
  u + v = [u_i + v_i]
  \]
  - Geometrically: Move along \( u \), then \( v \).

---

## 2. NumPy Core Operations

### Array Creation

- `np.array([...])`: Vectorize lists.
- `np.zeros(shape), np.ones(shape), np.arange(start, stop, step)`: Create numeric arrays.

### Norm Computations in NumPy

```python
np.linalg.norm(x, ord=None, axis=None)
```

- Default: L₂ norm for vectors.

### Dot Product in NumPy

```python
np.dot(a, b) # sum(a[i] * b[i])
```

---

## 3. Pandas Quick Reference

### Row-wise Normalization

```python
df.apply(lambda row: row / np.linalg.norm(row), axis=1)
```

- Normalizes each row vector to unit length.
- Use `apply(axis=1)` for row-wise custom operations.

---

## 4. Key Machine Learning Concepts

### Cosine Similarity

- Measures vector similarity independent of magnitude (length).
- Range: [-1, 1]

```python
cos_sim(a, b) = (a·b) / (||a|| ||b||)
```

### Regularization (L1 vs L2)

- **L1 (Lasso)**: Penalty encourages sparse solutions.
  \[
  	ext{Penalty}_{L1} = \lambda \|w\|_1 = \lambda \sum_i |w_i|
  \]

- **L2 (Ridge)**: Penalty shrinks weights without forcing sparsity.
  \[
  	ext{Penalty}_{L2} = \lambda \|w\|_2^2 = \lambda \sum_i w_i^2
  \]

- Controls model complexity, prevents overfitting.

---

## Derivation & Understanding Tips

- **Projection derivation**: Use orthogonality condition (residual · v = 0).
- **Dot product derivation**: Align vectors head to tail; dot is projection length times magnitude.
- **Norm understanding**: Derived from geometry; distance definitions (Euclidean, Manhattan).

---

*End of cheatsheet — your structured ML reference.*

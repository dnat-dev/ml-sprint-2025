```markdown
# CHEATSHEET: Your ML Handbook & Q&A

> Living reference covering NumPy, Pandas, Linear Algebra and core ML concepts—annotated with your own questions and answers.

---

## Table of Contents

1. [NumPy](#numpy)  
2. [Pandas](#pandas)  
3. [Linear Algebra](#linear-algebra)  
4. [ML Concepts](#ml-concepts)  

---

## NumPy

### Array Creation & Basics

- **Signatures**  
  ```python
  np.array([...])
  np.zeros(shape)
  np.ones(shape)
  np.arange(start, stop, step)
  ```
- **Summary**  
  Build _n_-dimensional arrays for fast numeric operations.
- **Q & A**  
  - **Q:** How do you vectorise a Python list?  
    **A:** Wrap it with `np.array(my_list)`.

### Norms (Lp)

- **Signature**  
  ```python
  np.linalg.norm(x, ord=None, axis=None)
  ```
- **Default**  
  - `ord=None` ⇒ L₂ norm for 1‑D arrays, Frobenius norm for matrices.
- **Variants**  
  - **L1 (`ord=1`)** → `sum(abs(v))`  
  - **L2 (`ord=2`)** → `sqrt(sum(v**2))`  
  - **L∞ (`ord=np.inf`)** → `max(abs(v))`  
  - **Lₚ (`ord=p`)** → `(sum(abs(v)**p))**(1/p)`
- **Q & A**  
  - **Q:** What’s the default `ord`?  
    **A:** L₂ for 1‑D arrays.  
  - **Q:** Can you use `ord=3`?  
    **A:** Yes—it computes the L₃ norm: `(∑|vᵢ|³)**(1/3)`.  
  - **Q:** Why is L1 ≥ L2?  
    **A:** Squaring then root‑taking shrinks sums; equality only when only one component is non‑zero.

### Dot Product & Alignment

- **Formula**  
  ```python
  np.dot(a, b)  # sum(a[i] * b[i])
  ```
- **Geometric**  
  \[
    a \cdot b = \|a\|\;\|b\|\;\cos\theta
  \]
- **Q & A**  
  - **Q:** What does “alignment” mean?  
    **A:** How closely two vectors point in the same direction (positive = same, zero = orthogonal, negative = opposite).  
  - **Q:** Is the dot product commutative (i.e. does xᵀy = yᵀx)?  
    **A:** Yes—because ∑ aᵢbᵢ = ∑ bᵢaᵢ.  
  - **Q:** Are dot and cross products the same?  
    **A:** No—the dot product yields a scalar; the cross product (3D only) yields a vector perpendicular to both inputs.  
- **Cosine Similarity**  
  ```python
  cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
  ```

---

## Pandas

### Row‑wise Normalisation

- **Pattern**  
  ```python
  df.apply(lambda row: row / np.linalg.norm(row), axis=1)
  ```
- **Summary**  
  Scale each DataFrame row to unit length for distance‑based models.
- **Q & A**  
  - **Q:** Do you use `apply` or `transform`?  
    **A:** Use `apply(..., axis=1)` for custom lambdas; `transform` for built‑in functions.

---

## Linear Algebra

### Scalar vs Vector

- **Scalar**  
  Single number (e.g., learning rate `η = 0.01`).
- **Vector**  
  Ordered list of scalars encoding magnitude + direction (e.g., `[3, 4]`).
- **Q & A**  
  - **Q:** What is a “numeric vector”?  
    **A:** A vector whose components are real numbers (ints/floats), ready for linear‑algebra ops.

### Magnitude (L₂ Norm)

- **Formula**  
  \[
    \|v\|_2 = \sqrt{\sum_i v_i^2}
  \]
- **Summary**  
  Euclidean length of a vector.
- **Q & A**  
  - **Q:** Why prefer L₂ for k‑means?  
    **A:** k‑means minimises squared Euclidean distances—L₂ matches that objective.

### Unit Vector

- **Formula**  
  ```python
  v_hat = v / np.linalg.norm(v)
  ```
- **Summary**  
  Direction‑only vector of length 1.
- **Q & A**  
  - **Q:** Is a unit vector equal to tan θ?  
    **A:** No—its components are `[cos θ, sin θ]`; tan θ = sin θ/cos θ.

### Vector Addition

- **Formula**  
  \[
    u + v = [u_i + v_i]
  \]
- **Summary**  
  Combine displacements head‑to‑tail.
- **Q & A**  
  - **Q:** What does the sum represent?  
    **A:** Total displacement after moving by u then v.

### Projection

- **Formula**  
  ```python
  proj_v_u = (u · v / (v · v)) * v
  ```
- **Summary**  
  Component of vector u along vector v’s direction.
- **Q & A**  
  - **Q:** How far is u from its projection onto v?  
    **A:** It’s the norm of the residual:  
      `||u - proj_v(u)|| = ||u|| * |sin θ|`.

---

## ML Concepts

### Cosine Similarity

- **Reuse dot & norms**  
  ```python
  cos_sim(a, b) = (a·b) / (‖a‖ ‖b‖)
  ```
- **Role**  
  Measure similarity of embeddings.

### Regularisation

- **L1 (Lasso)**  
  \[
    \|w\|_1 = \sum |w_i|
  \]
- **L2 (Ridge)**  
  \[
    \|w\|_2^2 = \sum w_i^2
  \]

---

> **Keep appending** new Q&As under relevant sub‑headings.  
> **Tip:** Use GitHub’s markdown preview and **Ctrl + F** to navigate quickly.

*End of cheatsheet—your living ML reference.*  
```
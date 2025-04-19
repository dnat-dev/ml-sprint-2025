# CHEATSHEET: Your ML Handbook & Q&A

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
- **Signatures:**  
  ```python
  np.array([...])  
  np.zeros(shape)  
  np.ones(shape)  
  np.arange(start, stop, step)
Summary: Build n‑dimensional arrays for fast numeric ops.

Q: How do you vectorise a Python list?
A: Wrap it with np.array(my_list).

Norms (Lp)
Signature: np.linalg.norm(x, ord=None, axis=None)

Default: L₂ norm for vectors, Frobenius for matrices (when ord=None).

Variants:

L1 (ord=1) → ∑|vᵢ|

L2 (ord=2) → √∑vᵢ²

L∞ (ord=np.inf) → max|vᵢ|

Lₚ (ord=p) → (∑|vᵢ|ᵖ)¹ᐟᵖ

Q: What’s the default ord?
A: L₂ for 1‑D arrays.

Q: Can you use ord=3?
A: Yes—computes the L₃ norm (∑|vᵢ|³)¹ᐟ³.

Q: Why is L1 ≥ L2?
A: Squaring then rooting shrinks compared to summing absolutes; equality only when one component is non‑zero.

Dot Product & Alignment
Formula: a·b = ∑ aᵢ bᵢ or np.dot(a, b)

Geometric: = ‖a‖ ‖b‖ cos θ

Q: What does “alignment” mean?
A: How closely two vectors point in the same direction:

positive → same direction

zero → orthogonal

negative → opposite

Cosine Similarity:

python
Copy
Edit
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
Pandas
Row‑wise Operations & Normalisation
Pattern:

python
Copy
Edit
df.apply(lambda row: row / np.linalg.norm(row), axis=1)
Summary: Scale each row to unit length for distance‑based methods.

Q: Do you use apply or transform?
A: apply(..., axis=1) for arbitrary row lambdas; use transform for built‑in vectorised functions.

Linear Algebra
Scalar vs Vector
Scalar: single number (e.g., learning rate η = 0.01).

Vector: ordered list of scalars → magnitude + direction, e.g. [3, 4].

Q: What is a “numeric vector”?
A: A vector whose components are real numbers (ints or floats), ready for linear‑algebra ops.

Magnitude (L₂ norm)
Formula: ‖v‖₂ = sqrt(sum(vᵢ²))

Summary: Euclidean length of a vector.

Q: Why prefer L₂ for k‑means?
A: k‑means minimises squared Euclidean distances, matching the L₂ objective.

Unit Vector
Formula: v̂ = v / ‖v‖₂

Summary: Direction‑only vector of length 1.

Q: Is unit vector = tan θ?
A: No—the components are [cos θ, sin θ]; tan θ = sin θ / cos θ.

Vector Addition
Formula: u + v = [uᵢ + vᵢ]

Summary: Combine displacements head‑to‑tail.

Q: What does the sum represent?
A: Total displacement when moving by u then v.

Projection
Formula: proj_v(u) = (u·v̂) * v̂

Summary: Component of u along v’s direction.

ML Concepts
Cosine Similarity
Reuse dot product & norms:
cos_sim(a,b) = (a·b) / (‖a‖ ‖b‖)

Role: Measure similarity of embeddings (text, users/items).

Regularisation
L1 (Lasso): ‖w‖₁ = ∑|wᵢ| → produces sparse weights.

L2 (Ridge): ‖w‖₂² → smooth shrinkage of all weights.
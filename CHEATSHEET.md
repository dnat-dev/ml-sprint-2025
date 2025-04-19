# CHEATSHEET: ML Core Concepts & Code Patterns

## Table of Contents
1. [NumPy](#numpy)  
2. [Pandas](#pandas)  
3. [Linear Algebra](#linear-algebra)  
4. [ML Concepts](#ml-concepts)  
5. [FAQ & Q&A](#faq--qa)  

---

## NumPy

### Array Creation
- **Signature:** `np.array([...])`, `np.zeros(shape)`, `np.ones(shape)`, `np.arange(start, stop, step)`
- **Summary:** Build nd‑arrays for efficient numerical ops.
- **Q:** How do you vectorise a Python list?  
  **A:** Wrap it with `np.array(list)`.  

### `np.linalg.norm`
- **Signature:** `np.linalg.norm(x, ord=None, axis=None)`
- **Summary:** Returns vector “size” (default L₂ norm for vectors).
- **Q:** What's default `ord` if unspecified?  
  **A:** L₂ norm (Euclidean).  
- **Q:** Can you use `ord=3`?  
  **A:** Yes, computes L₃ norm: `(sum |v_i|³)^(1/3)`.

---

## Pandas

### `apply` row‑wise normalization
- **Pattern:** `df.apply(lambda row: row/np.linalg.norm(row), axis=1)`
- **Summary:** Normalises each row to unit length.
- **Q:** Do you use `apply` or `transform`?  
  **A:** `apply(..., axis=1)` for row‑wise lambdas.

---

## Linear Algebra

### Scalar vs Vector
- **Scalar:** single number (e.g., learning rate η = 0.01).  
- **Vector:** ordered list of scalars with magnitude + direction (e.g., `[3,4]`).

### Dot Product
- **Formula:** `a·b = sum(a_i*b_i)` or `np.dot(a, b)`.
- **Geometric:** `= ||a|| ||b|| cos θ`.
- **Q:** What does “alignment” mean?  
  **A:** How closely two vectors point in same direction; sign/magnitude of dot.

### Magnitude (L₂ norm)
- **Formula:** `||v|| = sqrt(sum(v_i²))`.
- **Summary:** Euclidean length of vector.
- **Q:** Why prefer L₂ for k‑means?  
  **A:** k‑means minimises squared Euclidean distances.

### Unit Vector
- **Formula:** `v_hat = v / ||v||`.
- **Summary:** Direction-only vector of length 1.
- **Q:** Is unit vector = tan θ?  
  **A:** No; its components are [cos θ, sin θ], ratio = tan θ.

### Vector Addition
- **Formula:** `u + v = [u_i + v_i]`.
- **Summary:** Head-to-tail arrow addition.
- **Q:** What does the resulting vector represent?  
  **A:** Combined displacement of u then v.

### Projection
- **Formula:** `proj_v(u) = (u·v_hat) * v_hat`.
- **Summary:** Component of u along v’s direction.

---

## ML Concepts

### Cosine Similarity
- **Formula:** `(a·b) / (||a|| ||b||)`.
- **Summary:** Signed alignment between embeddings.

### Regularisation
- **L1 norm:** `||w||_1 = sum(|w_i|)` → sparsity (Lasso).  
- **L2 norm:** `||w||_2 = sqrt(sum(w_i²))` → smooth shrinkage (Ridge).

---

## FAQ & Q&A
- **Q:** Is each DataFrame row a vector?  
  **A:** After numeric encoding, yes—it's a feature vector.
- **Q:** What is a numeric vector?  
  **A:** A vector whose components are real numbers, ready for lin‑alg.
- **Q:** How to normalize DataFrame rows?  
  **A:** `df.apply(lambda row: row/np.linalg.norm(row), axis=1)`

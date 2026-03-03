# Routing Math and Load Balancing

## What This Is

Without load balancing, some experts get overloaded while others are idle. Load balancing loss encourages uniform expert utilization.

## The Math

### Router Probability

**For token $x$:**
$$g(x) = \text{softmax}(W_r x)$$

**Expert selection (top-k):**
$$S_x = \text{top-}k(g(x))$$

### Load Imbalance Problem

**Without aux loss:** Router may collapse to single expert.

**Example (8 experts, k=2):**
- Uniform: Each expert gets 25% of tokens
- Collapsed: Expert 0 gets 80%, others share 20%

### Load Balancing Loss

**Per-expert load:**
$$L_i = \sum_{x \in \text{batch}} \mathbb{1}(i \in S_x)$$

**Ideal load:** $L_{\text{ideal}} = \frac{\text{batch\_size} \cdot k}{E}$

**Auxiliary loss:**
$$\mathcal{L}_{\text{aux}} = \sum_{i=1}^{E} \left(\frac{L_i}{\sum_j L_j} - \frac{1}{E}\right)^2$$

**Total loss:**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{aux}}$$

Typical: $\lambda = 0.01$.

## The Kernel Implication

**Token dispatch (all-to-all):**
```cuda
// Route tokens to experts
for each token x:
    experts = top_k(router(x), k)
    for expert_id in experts:
        dispatch(x, expert_id)

// All-to-all communication
all_to_all(tokens_per_expert);
```

**Expert parallelism:**
- Each GPU holds subset of experts
- All-to-all sends tokens to correct GPU
- Compute expert FFN
- All-to-all return results

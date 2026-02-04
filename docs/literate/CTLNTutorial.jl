#  # CTLNs and Fixed Point Supports
#  
#  _Combinatorial Threshold–Linear Networks (CTLNs)_ are a class of recurrent neural networks whose dynamics are determined entirely by an underlying directed graph. CTLNs were introduced by Curto et al., and their subsequent theoretical development is surveyed in *Graph rules for recurrent neural network dynamics* [Curto & Morrison 2023](https://arxiv.org/abs/2301.12638). This line of work provides a compositional framework in which many long-term behaviors of a network can be understood by studying certain subgraphs.
#  
#  This page explains:
#  
#  - what a CTLN is,  
#  - what fixed point supports are and why they matter,  
#  - how we test whether a subset of neurons is a valid fixed point support,  
#  - why brute-force enumeration is hard, and  
#  - how a cover-based (sheaf-theoretic) dynamic programming workflow improves on brute-force enumeration.
#  
#  For details of our sheaf-theoretic formalization (fixed point support assignment seen as a presheaf, simply-embedded covers as Grothendieck pretopologies, and sheaf cases) and full proofs, we refer to the associated paper: 
# : Leal, W.; Cuffaro, M.; Hanks, T.; Bou Barceló, J.; Fairbanks, J. *Computing fixed points of CTLNs with separated presheaves and dynamic programming.* (In preparation), 2025.
#  
#  Leal, W.; Cuffaro, M.; Hanks, T.; Bou Barceló, J.; Fairbanks, J. *Computing fixed points of CTLNs with separated presheaves and dynamic programming.* (In preparation), 2025.
# 
# ## What is a CTLN?

# Consider a directed graph ``G = (V, E)`` with ``n = |V|`` vertices. Each vertex represents a neuron, and edges encode inhibitory interactions. A CTLN associated to ``G`` is a dynamical system of the form

# ```math
# \frac{dx_i}{dt}
# =
# - x_i(t)
# +
# \Big[ \sum_{j} W_{ij} x_j(t) + \theta \Big]_+,
# \quad i = 1,\dots,n,
# ```
# where:

# - ``x_i(t) \in \mathbb{R}_{\ge 0}`` is the activity (firing rate) of neuron ``i``,
# - ``W_{ij}`` is the synaptic weight from neuron ``j`` to neuron ``i``,
# - ``\theta \in \mathbb{R}`` is a constant external input,
# - ``[y]_+ = \max\{0,y\}`` is the threshold nonlinearity.

# In a standard CTLN parametrization, the weights come directly from the graph:

# ```math
# W_{ij} =
# \begin{cases}
# 0                 & \text{if } i = j, \\
# -1 + \varepsilon  & \text{if there is an edge } j \to i, \\
# -1 - \delta       & \text{if there is no edge } j \to i,
# \end{cases}
# ```
# with ``0 < \varepsilon \ll 1`` and ``\delta > 0``. Thus the **graph alone** determines the qualitative behavior of the CTLN.


# ---

# ## Fixed Points and Their Supports

# A **fixed point** of the CTLN is a vector ``x^* \in \mathbb{R}_{\ge 0}^n`` such that

# ```math
# x^*_i = \Big[\sum_j W_{ij} x^*_j + \theta\Big]_+ \quad \text{for all } i.
# ```

# The **support** of a fixed point is

# ```math
# \operatorname{supp}(x^*) := \{ i \mid x^*_i > 0 \}.
# ```

# A fundamental structural fact ([Curto & Morrison 2023](https://arxiv.org/abs/2301.12638)) is:

# > For each subset ``\sigma \subseteq V``, there is **at most one** fixed point with support ``\sigma``.

# Thus the main combinatorial object of interest is

# ```math
# \widehat{\mathrm{FP}}(G)
# = 
# \{ \sigma \subseteq V(G) \mid \sigma = \operatorname{supp}(x^*) \text{ for some fixed point } x^* \}
# \cup \{\emptyset\}.
# ```

# ### Why supports matter

# - Active neurons encode a memory or pattern.
# - Silent neurons ensure selectivity.
# - Different supports correspond to different attractors or dynamical regimes.

# Finding all supports gives a picture of the steady-state behaviors of the CTLN.

# ---

# ## How to Test a Candidate Support

# To check whether a subset ``\sigma \subseteq V(G)`` is a valid fixed point support:

# 1. **Solve the fixed point equations restricted to ``\sigma``** (assuming neurons in ``\sigma`` are active).
# 2. **Verify** the resulting solution is consistent with being silent off–support.

# To carry this out, one can follow these steps:

# **Step 1:** Restrict the Equations

# Assume

# - ``x^*_i > 0`` for ``i \in \sigma``,
# - ``x^*_j = 0`` for ``j \notin \sigma``.

# Then for each ``i \in \sigma``:

# ```math
# x^*_i
# =
# \sum_{j \in \sigma} W_{ij} x^*_j + \theta.
# ```

# This is a linear system in the variables ``x^*_\sigma``.

# **Step 2:** Matrix Form

# Define ``W_\sigma`` as the ``|\sigma| \times |\sigma|`` matrix

# ```math
# (W_\sigma)_{ij} =
# \begin{cases}
# W_{ij} & i \neq j, \\
# 0 & i = j.
# \end{cases}
# ```

# Then the restricted fixed point equation becomes

# ```math
# (I - W_\sigma)\, x^*_\sigma = \theta \cdot \mathbf{1}.
# ```

# If ``I - W_\sigma`` is invertible,

# ```math
# x^*_\sigma = (I - W_\sigma)^{-1} (\theta\mathbf{1}).
# ```

# **Step 3:** Validity Checks

# The support ``\sigma`` is valid iff:

# 1. **Positivity:**  
#    ``x^*_i > 0`` for all ``i \in \sigma``.

# 2. **Silence:**  
#    For each ``j \notin \sigma``,

#    ```math
#    \sum_{i \in \sigma} W_{ji} x^*_i + \theta \le 0.
#    ```

# Only if both conditions hold do we add ``\sigma`` to ``\widehat{\mathrm{FP}}(G)``.

# ---

# ## Brute-Force Enumeration and Its Cost

# The brute-force algorithm considers **all** nonempty subsets ``\sigma \subseteq V``. For each subset:

# 1. Form ``W_\sigma``,
# 2. Solve ``(I - W_\sigma)x^*_\sigma = \theta\mathbf{1}``,
# 3. Check positivity and silence conditions.

# Since there are ``2^n`` subsets, and each linear solve costs up to ``O(n^3)``,

# ```math
# T_{\text{brute}} = O(2^n \cdot n^3).
# ```


# This rapidly becomes infeasible as ``n`` grows.

# ---

# ## Compositional aspects

# There are three key results in Curto et.al. that invite a sheaf-theoretic reading:

# #### (CM1) Graphs can be decomposed into well-behaved pieces called symply embedded subgraphs

# **Definition (simply-embedded subgraphs)**
# We say that a subgraph `` G|_{\tau} `` is *simply-embedded in* `` G `` if for each  
# `` k \notin \tau ``, either:

# 1. `` k \to i `` for all `` i \in \tau ``, or  
# 2. `` k \not\to i `` for all `` i \in \tau ``.

# #### (CM2) Any global fixed point support restricts to a simply-embedded subgraph

# **Lemma (Support restriction)**  
# Let ``G|_\tau`` be simply-embedded in ``G``. Then for any ``\sigma\subseteq V(G)``,

# ```math
# \sigma \in \widehat{\mathrm{FP}}(G)\Rightarrow \sigma\cap \tau\in \widehat{\mathrm{FP}}(G|_\tau)\cup \{\emptyset\}.
# ```

# #### (CM3) Local solutions glue into global ones only for certain network architectures

# **Lemma**
# Let ``G|_{\tau_i}`` and ``G|_{\tau_j}`` be simply embedded in ``G``. If ``\sigma_i \in \widehat{\mathrm{FP}}(G|_{\tau_i})`` and ``\sigma_j \in \widehat{\mathrm{FP}}(G|_{\tau_j})`` satisfy

# ```math
# \sigma_i \cap \tau_j = \sigma_j \cap \tau_i,
# ```

# (i.e., they restrict to the same value on the overlap ``\tau_i \cap \tau_j``), then

# ```math
# \sigma_i \cup \sigma_j \in \widehat{\mathrm{FP}}(G|_{\tau_i \cup \tau_j})
# ```

# if and only if one of the following conditions holds:

# 1. ``\tau_i \cap \tau_j = \emptyset`` and  
#    ``\sigma_i, \sigma_j \in \widehat{\mathrm{FP}}(G|_{\tau_i \cup \tau_j})``, or

# 2. ``\tau_i \cap \tau_j = \emptyset`` and  
#    ``\sigma_i, \sigma_j \notin \widehat{\mathrm{FP}}(G|_{\tau_i \cup \tau_j})``, or

# 3. ``\tau_i \cap \tau_j \neq \emptyset``.

# ---

# ## The Curto-Morrison Conjecture (informally)

# The Curto-Morrison Conjecture can only be stated informally at this point:

# > The assignment of fixed point supports admits a genuine sheaf interpretation.

# ##  Our sheaf-theoretic framework

# Building on the restriction and gluing lemmas, we wrote a sheaf theoretical framework:

# #### (ST1) Constructed a *category of graphs and simple embeddings*
# **Lemma.** The following data defines a category, which we denote by  ``\mathsf{Grph_{se}}``:

# - objects are directed graphs,
# - morphisms are simply embedded inclusions  
#   ``G|_{\tau} \hookrightarrow G,``
# - composition is given by composition of graph inclusions.

# #### (ST2) Proved that the fixed point support assignment is functorial

# **Lemma.**  
# There is a functor  
# ```math
# \widehat{\mathrm{FP}} : \mathsf{Grph_{se}}^{op} \to \mathrm{Set}
# ```
# defined as follows:

# - **On objects:** for a graph $G$,

# ```math
#   \widehat{\mathrm{FP}}(G) \coloneqq \widehat{\mathrm{FP}}(G) \cup \{\emptyset\}.
# ```

# - **On morphisms:**  given a morphism ``\phi : G|_\tau \hookrightarrow G,``  witnessing that $G|_\tau$ is simply embedded in $G$, define

# ```math
#   \begin{aligned}
#   \widehat{\mathrm{FP}}(\phi) &: \widehat{\mathrm{FP}}(G) \to \widehat{\mathrm{FP}}(G|_\tau) \\
#                &\sigma \mapsto \sigma \cap \tau.
#   \end{aligned}
# ```
# Functoriality follows because the intersection is computed as a pullback.

# Having established the sheaf-theoretic framework for fixed point supports, we can now express the Curto–Morrison conjecture in precise categorical terms:

# > **Conjecture (Curto–Morrison)**  
# > The fixed point support presheaf  
# > ``\widehat{\mathrm{FP}}``  
# > is a **sheaf** with respect to the covering families  
# > ``\mathcal{J}^{\mathrm{dis}}``, ``\mathcal{J}^{\mathrm{clique}}``, and ``\mathcal{J}^{\mathrm{con}}``.


# #### (ST3) Showed that coverings by simple embeddings form *Grothendieck pretopologies*
# **Corollary**  
# The mapping  

# ```math
# \mathcal{J}(-) : \mathsf{Grph_{se}} \to \mathbf{Set}
# ```

# where

# ```math
#     \mathcal{J}(G) := \left\{ \Lambda \,\colon \Lambda \text{ is a simply-embedded cover of } G \right\}.
# ```
# defines a Grothendieck pretopology on $\mathsf{Grph_{se}}$.

# #### (ST4) Proved that the fixed point support functor ``\widehat{\mathrm{FP}}`` is “just” a *separated presheaf*
# **Theorem.**  The functor  

# ```math
# \widehat{\mathrm{FP}} : \mathsf{Grph_{se}}^{op} \to \mathbf{Set}
# ```

# is a separated presheaf with respect to $\mathcal{J}$.

# #### (ST5) Interpreted network architectures as restrictions of ``\mathcal{J}(-)`` and proved that for *disjoint unions* and *clique unions* the ``\widehat{\mathrm{FP}}`` functor is a *sheaf*

# **Theorem.** These functors are sheaves:

# - ``\widehat{\mathrm{FP}} : \big(\mathsf{Grph_{se}}^{op},\, \mathcal{J}^{\mathrm{dis}}\big) \to \mathbf{Set}`` (*disjoint unions*).

# - ``\widehat{\mathrm{FP}} : \big(\mathsf{Grph_{se}}^{op},\, \mathcal{J}^{\mathrm{clique}}\big) \to \mathbf{Set}``  (*clique unions*).

# We also showed that ``\mathcal{J}^{\mathrm{con}}`` is neither a Grothendieck pretopology nor a coverage, and therefore restricting to connected unions does not turn the ``\widehat{\mathrm{FP}}`` functor into a sheaf.

# Nevertheless, this remains a particularly interesting case: even though ``\widehat{\mathrm{FP}}`` is not a sheaf on ``\mathcal{J}^{\mathrm{con}}``, there is a **one-to-one correspondence** between global sections and matching families. This bijection allows us to compute global fixed-point supports efficiently using a dynamic programming algorithm (Algorithm 3 in Leal *et al.*) that is asymptotically faster than the general separated-presheaf algorithm.

# This is encouraging, as many graphs arising in applications are likely to admit connected-cover architectures.  We devote Subsection 4.3 in Leal *et al.* to these formal results and their computational implications.

# We can now state the solution to the conjecture:

# > **Result (Resolution of the Curto–Morrison Conjecture)**  
# > In response to the Curto–Morrison conjecture, we have shown that two of the proposed covering families  
# > ``\mathcal{J}^{\mathrm{dis}}`` and ``\mathcal{J}^{\mathrm{clique}}`` **do** turn the fixed point support functor  
# > ``\widehat{\mathrm{FP}}`` into a sheaf (Theorems 4.9 and 4.11 in Leal *et al.*),  
# > while the connected-cover family ``\mathcal{J}^{\mathrm{con}}`` does **not** (Example 4.14),  
# > as it fails to define a coverage or Grothendieck pretopology and therefore cannot yield a sheaf for ``\widehat{\mathrm{FP}}``.

# #### (ST6) Developed *dynamic programming algorithms* from these results and showed they achieve better asymptotic running time than brute force.

# Because ``\widehat{\mathrm{FP}}`` is a separated presheaf, local fixed-point data can be consistently glued.  This leads to the following dynamic-programming algorithm:

# **Algorithm 1:** Fixed Point Enumeration from a Simply-Embedded Cover

# 1. **Initialize:** Let ``\widehat{\mathrm{FP}}(G) \gets \emptyset.``
   
# 2. **Local brute-force enumeration:** For each subgraph in the cover ``\{\, G|_{\tau_i} \to G \,\}_{i \in I} \in \mathcal{J}(G),``  compute ``\widehat{\mathrm{FP}}(G|_{\tau_i})`` using brute force on the smaller induced subgraph.

# 3. **Iterate over local combinations**  
#    Consider each tuple  
#    ```math
#    (\sigma_i) \in \prod_{i \in I} \widehat{\mathrm{FP}}(G|_{\tau_i}).
#    ```
#    Check whether the family is **matching** on overlaps (i.e., agrees on intersections).

# 4. **Form global candidates**  
#    For each matching family, define  
#    ```math
#    \sigma \;=\; \bigcup_{i \in I} \sigma_i.
#    ```
#    Then validate globally:
#    - perform the global fixed-point test;
#    - if valid, add it to ``\widehat{\mathrm{FP}}(G)``.

# 5. **Return:** The final collection ``\widehat{\mathrm{FP}}(G)`` contains all global fixed-point supports of ``G``.

# Algorithm 1 has the following asymptotic time complexity:
# ```math
# T_{\text{se-cover}}
# =
# O\!\left(
# \underbrace{
# \sum_{i \in I} 2^{|\tau_i|} \cdot |\tau_i|^3
# }_{\text{Local enumeration}}
# \;+\;
# \underbrace{
# \left( \prod_{i \in I} m_i \right)
# \cdot
# \left( |I|^2 \cdot n + n^3 \right)
# }_{\text{Contributed supports} \times (\text{gluing + validation})}
# \right).
# ```
# Its complexity is governed by three structural features of the cover:  
# 1. the sizes ``|\tau_i|`` of the subgraphs ``G|_{\tau_i}``,  
# 2. the number of fixed point supports ``m_i`` contributed by each block, and  
# 3. the number of cover elements ``|I|`` relative to the size of ``G``.

# We showed that even in situations where the cover consists of many, very large subgraphs, each potentially contributing a large number of local supports, the separated-presheaf algorithm still achieves an asymptotic speedup over brute force.  To illustrate this, we introduce the notion of polynomial covers:

# **Definition (Polynomial Cover).** Let ``G`` be a graph with ``n`` vertices.  A covering family ``\{\, G|_{\tau_i} \hookrightarrow G \,\}_{i \in I}`` is called a *polynomial cover* if:

# 1. **Sublinear piece size:** each induced subgraph satisfies  
#    ``|\tau_i| \le k = o(n)``.

# 2. **Polynomial cover growth:** the number of pieces satisfies  
#    ``|I| = \mathrm{poly}(n)``.

# 3. **Polynomial support bound:** the total number of local supports satisfies  
#    ```math
#    \prod_{i \in I} m_i = \mathrm{poly}(n).
#    ```

# **Proposition (Polynomial covers yield asymptotic speedup).** If ``\{\, G|_{\tau_i} \hookrightarrow G \,\}_{i \in I}`` is a polynomial cover of a graph ``G,`` then the separated-presheaf algorithm satisfies

# ```math
# T_{\mathrm{se\text{-}cover}} = o\!\big(T_{\mathrm{brute}}\big).
# ```

# ---

# ## Connected Unions

# Algorithm 3 in Leal *et al.* computes the global fixed point supports of a graph ``G`` using a connected simply-embedded cover  
# ``\{\, G|_{\tau_i} \hookrightarrow G \,\}_{i \in I} \in \mathcal{J}^{\mathrm{con}}(G)``.  
# Since matching families always glue in this setting (Proposition 4.13 in Leal *et al.*), the algorithm avoids the validation step required in the general separated-presheaf case. With time complexity:

# ```math
# T_{\text{connected}}
# =
# O\!\left(
# \sum_{i \in I} 2^{|\tau_i|} \cdot |\tau_i|^3
# \;+\;
# \left( \prod_{i \in I} m_i \right)
# \cdot (|I|^2 \cdot n)
# \right).
# ```

# ---

# ## Special Architectures and Sheaf Behavior

# For *disjoint* and *clique* unions, the covering families consist of subgraphs with pairwise disjoint vertex sets. This gives a significant advantage: any tuple ``(\sigma_i) \in \prod_{i} \widehat{\mathrm{FP}}(G|_{\tau_i})`` automatically forms a matching family, so no compatibility checks are required. Algorithm 4 and Algorithm 5 in Leal *et al.*   show how to enumerate fixed points for disjoint and clique unions, respectively, and their time complexity is given by:

# ```math
# T_{\text{sheaf}}
# =
# O\!\left(
# \sum_{i \in I} 2^{|\tau_i|} \cdot |\tau_i|^3
# \;+\;
# \left( \prod_{i \in I} m_i \right) \cdot n
# \right)
# ```

# ---

# ## Asymptotic Comparison

# - **Sheaf cases** (disjoint and clique unions) are the fastest.
# - **Connected unions** still beat the general symply-embedded cover case and the brute force case.
# - **General simply-embedded covers** although it requires some global checks, it still beats brute force.
# - **Brute force** base line.

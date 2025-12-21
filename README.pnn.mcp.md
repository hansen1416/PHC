**PNN** here means **Progressive Neural Networks**: a *continual / progressive* training scheme that **grows a stack of policy subnetworks over time** to mitigate catastrophic forgetting. In PHC, it is used to **progressively add new “primitives”** to handle increasingly difficult motion subsets.

### What “PNN” does in PHC

PHC’s description matches the classic PNN recipe:

1. Train an initial **primitive policy** (P(1)) on the full motion set (\hat{Q}).
2. Evaluate (P(1)); collect the subset of sequences it fails on as a “hard set” (\hat{Q}(1)).
3. **Freeze** (P(1)), initialize a new primitive (P(2)), and train it on harder data; repeat to build (P(1),\dots,P(K)).

PHC also discusses a practical **variant**: instead of explicit lateral connections, new primitives may be **initialized from the previous primitive’s weights** (a weight-sharing / warm-start form), while still freezing older primitives to preserve earlier capabilities. 

Because motion imitation lacks clean “task labels,” PHC notes that *manual* subnetwork selection (typical in vanilla PNN) is not ideal, and therefore introduces a **composer** to combine primitives automatically.

### What a “primitive” is (and how it differs)

A **primitive** in PHC is simply **one policy network** (P(k)) (one “expert”) that maps the current state/goal to an action distribution and can imitate **some subset** of motions. After progressive training, PHC explicitly treats the result as a set ({P(1)\dots P(K), P(F)}), where each primitive covers part of (\hat{Q}). 

### PNN vs. a single primitive (your “train a primitive” command)

* **Model structure**

  * *Single primitive:* one policy (P).
  * *PNN:* multiple frozen policies (P(1),P(2),\dots) added progressively (optionally with lateral connections or warm-start).

* **Forgetting**

  * *Single primitive:* susceptible to **catastrophic forgetting** when learning harder motions later.
  * *PNN:* mitigates forgetting by **freezing** earlier primitives and allocating new capacity for harder subsets.

* **Specialization**

  * *Single primitive:* a generalist that must cover all motion diversity within fixed capacity.
  * *PNN:* produces a set of **specialists** (each primitive responsible for a harder subset discovered by evaluation).

* **Inference / selection**

  * *Single primitive:* direct forward pass.
  * *PNN (PHC setting):* needs a mechanism to decide “which primitive to use”; PHC addresses this by learning a **composer** that combines primitives (to avoid manual task switching).

So, in your folder names like `..._pnn_...`, **PNN typically signals the model was trained with this progressive multi-primitive pipeline**, whereas your “primitive” training command is the **single-policy baseline** (no progressive expansion, no composer).



-------------------



“**Grows a stack of policy subnetworks over time**” is PNN jargon for **adding new network “columns” sequentially**, while **freezing** previously trained columns. In PHC, each column corresponds to a **primitive policy** (P(k)).

## How PNN adds new primitives (PHC’s usage)

PHC follows the standard PNN pattern:

1. **Train the first primitive** (P(1)) on the full motion set (\hat{Q}) until convergence.
2. **Evaluate** (P(1)) on (\hat{Q}); collect the sequences it fails as the “hard set” (\hat{Q}(1)).
3. **Freeze** (P(1)) (no more gradient updates).
4. **Create a new primitive** (P(2)) (PHC says randomly initialized) and **add lateral connections** from each layer of (P(1)) into the corresponding layer of (P(2)).
5. Train (P(2)) on the harder subset; repeat to obtain (P(1),P(2),\dots,P(K)).

So the “stack” is literally the set ({P(1),\dots,P(K)}) accumulated over training, with earlier ones frozen and later ones having extra capacity and (optionally) lateral inputs from earlier ones.

## Do PNN primitives get “put together”?

**Not by PNN itself.** In *vanilla* PNN, you typically **manually choose one column** for the current task (task switching). PHC explicitly notes this manual switching is problematic for motion imitation (no clean task labels).

## Why it “sounds like MCP” (and the key difference)

You’re picking up the right connection: **PHC uses MCP to combine primitives**, but **PNN and MCP play different roles**:

* **PNN = training-time capacity growth / anti-forgetting mechanism**
  It explains *how you obtain* multiple primitives by progressive expansion and freezing.

* **MCP = inference-time composition mechanism**
  After primitives are trained, PHC trains a **composer** (C) that outputs weights (w) and **multiplicatively combines** the primitives’ action distributions (a product-of-experts style combination), rather than selecting only one expert.

PHC’s final policy is explicitly a **composer + pretrained primitives** setup:
[
\pi_{\text{PHC}}(a_t\mid s_t) \propto \prod_i P(i)(a_t\mid s_t)^{,C_i(s_t)}
]
i.e., **all actors are “active” in the product** (contrast: top-1 MoE activates one).

PHC even states **“Unlike MCP, we progressively train our primitives …”**—meaning MCP alone is the *combiner*, while PHC’s novelty is combining it with **progressive training (PNN-style)** to get better-specialized primitives before composition.

**Bottom line:**

* PNN answers: *“How do we add new primitives without forgetting?”* (freeze old, add new capacity, optional lateral connections).
* MCP answers: *“How do we use multiple primitives together at runtime?”* (learn a composer and multiply distributions).

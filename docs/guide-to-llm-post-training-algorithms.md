# A Guide to Reinforcement Learning Post-Training for LLMs: PPO, DPO, GRPO, and Beyond

> **Sumber:** https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms  
> **Penulis:** Karina Zadorozhny  
> **Dipublikasikan:** 19 Januari 2026  
> **Catatan:** Artikel ini merupakan Community Article di Hugging Face. Formula matematika dirender ulang ke format LaTeX standar.

---

## Definitions

Let's define standard reinforcement learning terms with an LLM setup in mind.

- **State $s_t$**: The current context — the original user prompt and all tokens generated so far.  
  *Example:* Prompt: "The sky is..." → State: `["The", "sky", "is"]` in the token-space

- **Action $a_t$**: The next token being generated.  
  *Example:* `"blue"`

- **Policy $\pi_\theta$**: The LLM itself. Essentially a probability distribution over all words in a vocabulary given a state $s_t$.  
  *Example:* $\pi_\theta(\text{"blue"} \mid [\text{"The", "sky", "is"}]) = 0.87$

- **Trajectory $\tau$**: The full conversation. The complete sequence of states (context) and actions (tokens chosen) from the prompt to the end-of-sequence token.

- **Reward $R$**: The score. Usually assigned to the entire trajectory, indicating how good the full response was.

- **Critic Network $V(s)$**: A separate model (or head) that estimates the value of a state. It predicts how much future reward we're expecting to get given the current state. Also referred to as the Value Function.

- **Reward Model**: A separate model trained to learn human preferences or other rewards. Most commonly, it takes the full response and outputs the scalar reward $R$.

- **Reference model $\pi_{ref}$**: The starting, pre-trained LLM after SFT before any RL was used.

---

## On-Policy Algorithms

In on-policy learning, the model actively generates its own data during training. The model generates responses. The responses get scored and the model's parameters updated.

### The Core Objective

The core objective is to maximize the Expected Return $J(\pi_\theta)$:

$$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

This means we are computing the expected value of the reward across all conversations sampled from our model $\pi_\theta$.

> **Note:** Strictly speaking, the policy $\pi_\theta$ outputs probabilities for a single token at a time. We write that we're sampling the full trajectory (sequence) from $\pi_\theta$ but in fact we need some decoding strategy (like ancestral sampling or top-p sampling). This is just a shorthand for saying that we are autoregressively generating the sequence step-by-step based on the policy.

We want to find weights $\theta$ of our model that maximize the expected return $J$. To do this, we need to compute its gradients $\nabla_\theta J(\pi_\theta)$:

$$\nabla_\theta J(\pi_\theta) = \nabla_\theta \int P(\tau|\theta) R(\tau) \, d\tau$$

The problem is that we can't differentiate through a discrete generation process, such as sampling tokens. In a normal neural network, we can compute gradients for every step. But when an LLM generates text, it performs a sampling step that is non-differentiable. Moreover, the reward is often non-differentiable too (unless we use a differentiable reward model). To solve this, we use the **score function estimator**, also known as the **log-derivative trick**.

We first move the gradient inside (assuming regularity):

$$= \int \nabla_\theta P(\tau|\theta) R(\tau) \, d\tau$$

and apply the log-derivative trick:

$$= \int \underbrace{P(\tau|\theta) \nabla_\theta \log P(\tau|\theta)}_{\text{Replaced } \nabla_\theta P} R(\tau) \, d\tau$$

This lets us rewrite the gradient as an expectation and we can use sampling to estimate it:

$$= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau|\theta) \cdot R(\tau) \right]$$

This is great because we no longer need the derivative of the reward function and can estimate the gradient by sampling trajectories.

For LLMs, the probability of the whole text is the product of probabilities for each token:

$$P(\tau|\theta) = \prod_{t=0}^T \pi_\theta(a_t | s_t)$$

We can take the log of the product, which becomes a sum:

$$\log P(\tau|\theta) = \sum_{t=0}^T \log \pi_\theta(a_t | s_t)$$

Substituting back, and generalizing the reward term $R(\tau)$ into a weight $\Phi_t$:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \underbrace{\nabla_\theta \log \pi_\theta(a_t|s_t)}_{\text{Direction}} \cdot \underbrace{\Phi_t}_{\text{Weight}} \right]$$

where:
- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ is the gradient of the log-probability — it tells us exactly how to update model parameters to make token $a_t$ more or less likely given context $s_t$.
- $\Phi_t$ is the weight of this update.

You can see different policy gradient algorithms as different implementations of the weight:

- If $\Phi_t = R(\tau)$ (total reward of the trajectory), we get **REINFORCE** (or more commonly, rewards-to-go $\Phi_t = \sum_{k=t}^T r_k$).
- If $\Phi_t = Q(s_t, a_t) - V(s_t)$ (the Advantage), we get **Vanilla Policy Gradient** or **Actor-Critic** methods.

---

### REINFORCE

REINFORCE is essentially a direct implementation of the equation above. In the simplest case, the weight of updates is set as the trajectory's total reward $R(\tau)$. It is intuitive: if the model writes a response and gets a high score, we reinforce every token it used. If it gets a low score, we discourage them.

The update rule (averaged over a batch of trajectories):

$$\nabla_\theta J(\pi_\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)$$

where:
- $\nabla_\theta \log \pi_\theta(a_t|s_t)$ is the gradient of the log-probability.
- $R(\tau)$ is the score of the full response. If the reward is positive, we increase the probability of the token. If negative, we suppress it.

REINFORCE can be very inefficient. Imagine a model generates a very good long response but hallucinates at the end, and the Reward Model assigns it a score of $-100$. REINFORCE will punish the entire generated text because it can't tell which token caused the low score. The biggest difficulty of this algorithm is **high variance and instability**.

---

### PPO

Proximal Policy Optimization (PPO) is a family of policy gradient methods introduced by OpenAI in 2017 ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)). It fixes many of REINFORCE's instabilities and is one of the most widely used RL algorithms. It solves issues via three main improvements:

- **Generalized Advantage Estimation (GAE):** Reduces variance in policy gradient estimates while maintaining low bias.
- **Actor-Critic interplay:** Introduces a separate critic model (value function).
- **Clipped Updates:** Limits policy updates using a clipped surrogate objective or adaptive KL divergence penalty.

#### Advantage Estimation and Critic Network

Instead of using raw reward $R$, PPO uses the **Advantage $\hat{A}_t$** — how much better this specific action was compared to the expected baseline result.

Advantage only looks at rewards from step $t$ onward (the new token cannot influence the past):

$$\hat{A}_t = Q(s_t, a_t) - V(s_t)$$

where:
- $Q(s_t, a_t)$ = rewards-to-go (actual total reward after taking action $a_t$).
- $V(s_t)$ = average reward usually obtained from this state, predicted by the **Critic** (a separate network).

This significantly reduces noise by normalizing the signal against a baseline and only considering future rewards.

More specifically, PPO uses **Generalized Advantage Estimation (GAE)** ([Schulman et al. 2015](https://arxiv.org/pdf/1506.02438)), which uses smooth, exponentially-weighted advantage estimates from many steps, stabilizing training.

#### PPO-CLIP

PPO-CLIP stays within the trust region by clipping updates at each step. Given current policy $\pi_\theta$ and previous policy $\pi_{old}$, the probability ratio is:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$$

The ratio measures how different the new model's probability is compared to the old one. Updates are restricted:

$$L^{PPO\text{-}CLIP}(\theta) = \mathbb{E} \left[ \min\left( \underbrace{r_t(\theta)\hat{A}_t}_{\text{Unclipped}}, \underbrace{\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t}_{\text{Clipped}} \right) \right]$$

where:
- $\hat{A}_t$ is the Advantage.
- $\epsilon$ is a small hyperparameter, usually 0.2 (allows 20% deviation from the old model).
- $\text{clip}(\dots)$ forces the ratio to be between $0.8$ and $1.2$.

#### PPO-KLPEN

An alternative formulation that adds an explicit KL divergence penalty with an adaptive penalty coefficient:

$$L^{PPO\text{-}KL}(\theta) = \mathbb{E}_t \left[ r_t(\theta)\hat{A}_t - \beta \cdot D_{KL}[\pi_{old}(\cdot|s_t) \,||\, \pi_\theta(\cdot|s_t)] \right]$$

where the forward KL divergence penalty is:

$$D_{KL}[\pi_{old}(\cdot | s_t) || \pi_{\theta}(\cdot | s_t)] = \sum_a \pi_{old}(a | s_t) \log \left( \frac{\pi_{old}(a | s_t)}{\pi_{\theta}(a | s_t)} \right)$$

The coefficient $\beta$ is updated adaptively during training. Compared to PPO-CLIP, this method imposes a softer constraint on policy divergence but is more complex to implement. PPO-CLIP is more widely adopted in practice.

---

### GRPO

PPO requires loading several huge models simultaneously: the policy (LLM being trained), the reference model (frozen), a separate large Critic network, and often a Reward Model — very memory-intensive.

DeepSeek-R1 and V3 skip the Critic network. Instead of subtracting the Critic-predicted value $V(s_t)$ as baseline, **GRPO** samples a group of responses for a given prompt and uses the mean score of the group as the baseline.

Given a group of sample responses $\{o_1, o_2, ..., o_G\}$ (usually $G = 64$) from the current model with scores $\{R_1, R_2, ..., R_G\}$, the advantage for output $i$ is:

$$A_i = \frac{R_i - \text{mean}(\{R_1, R_2, ..., R_G\})}{\text{std}(\{R_1, R_2, ..., R_G\})}$$

The final GRPO loss:

$$\mathcal{L}_{GRPO}(\theta) = - \frac{1}{G} \sum_{i=1}^G \left( \min \left( r_i(\theta) A_i,\ \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta \cdot D_{KL}[\pi_\theta \,||\, \pi_{ref}]_i \right)$$

**Important:** The KL term in GRPO is different from PPO-KLPEN. GRPO's KL penalty is between the **frozen reference model $\pi_{ref}$** and the **current model $\pi_\theta$** — not between consecutive policy updates. This is a common source of confusion in many open-source implementations.

---

## KL Divergence Clarification

There are two KL penalties commonly seen: **Trust-Region KL** and **Drift KL**. Their purpose is very different, but they often get mixed up. They also use opposite KL types — one is forward and the other reverse.

### Quick Refresher on KL Divergence Types

KL divergence is asymmetric: $D_{KL}(P || Q) \neq D_{KL}(Q || P)$.

**Forward KL:**

$$D_{KL}(P || Q) = \sum_x P(x) \log \left( \frac{P(x)}{Q(x)} \right)$$

Forward KL is **mean-seeking**: when minimizing w.r.t. $Q$, it forces $Q$ to cover all modes of $P$.

**Reverse KL:**

$$D_{KL}(Q || P) = \sum_x Q(x) \log \left( \frac{Q(x)}{P(x)} \right)$$

Reverse KL is **mode-seeking**: it tries to capture one mode well and ignores other modes. It is fine with leaving some areas completely uncovered.

---

### Trust Region KL Penalty

**Purpose: stabilize training** — ensures updates at each step are not too different from the immediately previous weights.

$$D_{KL}(\pi_{\theta_{old}} || \pi_{\theta}) = \mathbb{E}_{\mathbf{x} \sim \pi_{\theta_{old}}} \left[ \log \frac{\pi_{\theta_{old}}(\mathbf{x})}{\pi_{\theta}(\mathbf{x})} \right]$$

This is **Forward KL** where $P = \pi_{\theta_{old}}$ is fixed and we optimize the second argument $\pi_\theta$. Because Forward KL is mean-seeking, the new policy is prevented from assigning low probability to things the old policy thought were good — we cover all valid regions of the old policy.

The name "Trust Region" comes from Trust Region Policy Optimization (TRPO) ([Schulman et al. 2015](https://arxiv.org/pdf/1502.05477)).

This penalty is **subtracted directly from the loss** in PPO-KLPEN.

---

### Drift KL Penalty

**Purpose: prevent reward hacking** — ensures the model $\pi_\theta$ doesn't drift too far from the original SFT-pretrained reference model $\pi_{ref}$.

The Reward Model is not perfect; unconstrained optimization leads the model to find ways to hack it — in the extreme, outputting gibberish that gets very high scores. To avoid this, we use the Drift KL penalty.

This penalty is **Reverse KL**:

$$D_{KL}(\pi_{\theta} || \pi_{\text{ref}}) = \mathbb{E}_{\mathbf{x} \sim \pi_{\theta}} \left[ \log \frac{\pi_{\theta}(\mathbf{x})}{\pi_{\text{ref}}(\mathbf{x})} \right]$$

**Why Reverse KL?** It forces the model to only respond with outputs that have good support in the reference model (i.e., keep using natural language, not gibberish). If $\pi_\theta$ generates something where $\pi_{\text{ref}} \approx 0$, this leads to a massive penalty. At the same time, we don't mind the model forgetting other modes of the original model (e.g., toxic responses from pretraining data).

Typically, this penalty is **subtracted from the reward function**:

$$R_{total}(s, a) = R_{model}(s, a) - \beta \log \left( \frac{\pi_\theta(a|s)}{\pi_{ref}(a|s)} \right)$$

This is how Standard RLHF, PPO-CLIP, and PPO-KLPEN do it. Note: the Drift KL penalty is usually the one people refer to and is much more commonly used.

In GRPO, the KL regularization term is placed directly inside the loss function (not subtracted from the reward).

---

### KL Estimators and Gradient Pitfall

In practice, KL penalties cannot be computed directly — **sample-based KL estimators** are used instead. This is a critical implementation detail that has recently been scrutinized, as different estimators lead to different issues.

Recent work — *A Comedy of Estimators* ([Shah et al. 2026](https://arxiv.org/abs/2512.21852)) and *On a few pitfalls in KL divergence gradient estimation* ([Tang & Munos 2025](https://arxiv.org/abs/2506.09477)) — has shown that popular methods, including those in popular open-source libraries, often calculate **wrong gradients**.

Three popular KL estimators:

**K1 Estimator** (Naive) — Monte Carlo estimator from the definition. Unbiased but high variance.

$$\mathbb{KL}_1 = \sum_{t=1}^T \log \frac{\pi}{\pi_{\text{ref}}}$$

**K2 Estimator** — Based on Taylor expansion. Squaring the log ratio ensures the penalty is always non-negative (but makes it symmetric, unlike KL).

$$\mathbb{KL}_2 = \sum_{t=1}^T \frac{1}{2} \left( \log \frac{\pi}{\pi_{\text{ref}}} \right)^2$$

**K3 Estimator** — Used in the PPO paper and often the default in libraries like TRL. Small deviations have almost no penalty. Unbiased estimator.

$$\mathbb{KL}_3 = \sum_{t=1}^T \left( \frac{\pi}{\pi_{\text{ref}}} - 1 - \log \frac{\pi}{\pi_{\text{ref}}} \right)$$

The Drift KL penalty can be:
- **In the Reward:** calculate the KL value, **stop gradients**, subtract from reward.
- **In the Loss:** passed to the differentiator and **included in gradients**.

**The pitfall:** when you differentiate a KL estimate in the loss, it does not equal the gradients of the KL divergence term. So even if an estimator is unbiased, its gradients can be biased and wrong.

Shah et al. 2026 results:

| **Configuration** | **Gradient Bias**   | **Behavior**           | **Performance** |
|-------------------|---------------------|------------------------|-----------------|
| K1 in Reward      | Unbiased            | Stable                 | **Best** — gold standard; outperforms other approaches on many reasoning tasks. |
| K1 in Loss        | Zero in expectation | Training Instabilities | **Poor** — gradient expectation becomes zero, model only receives noise. |
| K3 in Reward      | Biased              | Training Collapse      | **Failure** — biased gradient leads to full or partial collapse for all tested coefficient values. |
| K3 in Loss        | Biased              | Stable                 | **Good** — this is what GRPO does; works pretty well but worse than K1 in Reward. |

---

## Off-Policy Learning

Off-policy methods do not have an explicit KL divergence term but model this implicitly. They address the computational bottleneck of on-policy methods (the generation step) and the need for multiple large models (Actor, Critic, Reward Model, Reference Model) loaded simultaneously.

Off-policy methods learn from existing data and omit the need for a separate Reward Model.

---

### Direct Preference Optimization (DPO)

DPO ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)) doesn't use a separate Reward Model neural network. Instead, it extracts the optimal reward directly from preference data. DPO has been used in Llama 3 (combined with PPO) and Qwen-Chat models.

**Derivation of DPO's objective:**

**Step 1: Define the RLHF objective**

The standard RLHF objective with KL term added to reward:

$$\text{Reward} = r(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

Goal: find policy $\pi$ that maximizes expected reward while staying close to reference model $\pi_{ref}$:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right]$$

where $x$ is the input prompt, $y$ is the output, $r(x, y)$ is the reward function, and $\beta$ controls the KL penalty.

**Step 2: Closed-form solution**

The optimal policy $\pi^*$ for the objective above is:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right)$$

The perfect policy is essentially the reference policy $\pi_{ref}$ scaled by the reward. $Z(x)$ is a partition function (normalization constant) — we can't compute this term directly.

**Step 3: Invert the equation**

Instead of training a Reward Model to estimate $r(x, y)$ and then using PPO, DPO solves the equation for the reward (**DPO's main trick**):

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**Step 4: Cancel out terms**

In RLHF, we maximize the difference between good responses (winners $y_w$) and bad responses (losers $y_l$): $r(x, y_w) - r(x, y_l)$.

Rewriting using the equation above:

$$r(x, y_w) - r(x, y_l) = \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} + \beta \log Z(x) \right)$$

The partition function cancels out in both expressions! This gives the **final DPO loss**:

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

where $\sigma$ is the sigmoid function.

**Interpretation:** DPO directly trains the LLM to prefer the winning response over the losing response, while implicitly staying close to the reference model. No separate Reward Model needed, no on-policy generation — just supervised learning on preference pairs.

**Limitations of DPO:**
- Off-policy: the winning/losing responses in the dataset were generated by a different model, not the current policy. This mismatch can hurt performance.
- Sensitive to data quality — noisy preference labels can be harmful.
- Less flexible than PPO for complex reward structures.

---

### Other Off-Policy Methods

Several variants and alternatives to DPO have been proposed:

**ORPO (Odds Ratio Policy Optimization)** — combines SFT and preference alignment in a single training step, removing the need for a reference model entirely. Uses the odds ratio between winning and losing responses as the training signal.

**SimPO (Simple Preference Optimization)** — simplifies DPO by using the average log-probability of a response (rather than the ratio with a reference model) as the implicit reward. This removes the reference model dependency while achieving competitive performance.

**IPO (Identity Preference Optimization)** — addresses DPO's tendency to overfit to preference data by adding a regularization term that directly optimizes a divergence measure between the policy and reference.

**KTO (Kahneman-Tversky Optimization)** — uses binary feedback (thumbs up/thumbs down) rather than pairwise preferences, making it applicable when you only have unpaired good/bad examples rather than paired comparisons.

---

## Summary: On-Policy vs Off-Policy

| Algorithm | Type | Reward Model | Critic | Key Property |
|-----------|------|-------------|--------|-------------|
| REINFORCE | On-policy | Required | No | Simple, high variance |
| PPO | On-policy | Required | Yes (large) | Stable, memory-intensive |
| GRPO | On-policy | Required | No (uses group mean) | PPO without Critic; used in DeepSeek-R1 |
| DPO | Off-policy | No (implicit) | No | No generation needed; trains on preference pairs |
| ORPO | Off-policy | No | No | Single-step SFT + alignment |
| SimPO | Off-policy | No | No | No reference model needed |
| KTO | Off-policy | No | No | Works with binary (unpaired) feedback |

---

## References

- Schulman, J. et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Schulman, J. et al. (2015). [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477)
- Schulman, J. et al. (2015). [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438)
- Rafailov, R. et al. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- Shao, Z. et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) *(GRPO)*
- Shah, R. et al. (2026). [A Comedy of Estimators: Understanding KL Divergence Estimators in RLHF](https://arxiv.org/abs/2512.21852)
- Tang, Y. & Munos, R. (2025). [On a few pitfalls in KL divergence gradient estimation](https://arxiv.org/abs/2506.09477)

---

*Downloaded from: https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms*

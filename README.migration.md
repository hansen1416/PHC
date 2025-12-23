### Collect observation, setting target motion
`HumanoidIm`, phc/env/tasks/humanoid_im.py

maintains the current humanoid state after each physics step: root position/rotation, joint positions, velocities, body poses, angular velocities, etc.

- Physics simulation advances → new humanoid state
- full obs for actor/critic are computed for the current rollout
- Expert demonstrations come from pre-loaded motion clips
- During training, the environment samples random reference motions and provides amp_demo_obs (expert AMP features) on demand (either per-reset or per-batch)


### Forward Pass

`ModelAMPContinuous`, phc/learning/amp_models.py

The continuous action spaces that wraps the network built by `AMPBuilder`.
Integrates the discriminator into the training loop by computing logits on various motion observations, providing signals (logits) for adversarial/imitation losses

- Uses the underlying network's eval_disc() method (built via AMPBuilder).
During training (is_train=True), it processes three types of inputs:
`amp_obs`: Current agent-generated motion observations.
`amp_obs_replay`: Observations from the replay buffer.
`amp_demo_obs`: Expert demonstration observations.

- Computes and attaches discriminator logits to the output dictionary:
    - disc_agent_logit = discriminator(amp_obs)
    - disc_agent_replay_logit = discriminator(amp_obs_replay)
    - disc_demo_logit = discriminator(amp_demo_obs)

- These logits are used externally (in the trainer/loss function `IMAmpAgent`) for adversarial objectives.

### Discriminator Architecture

`AMPBuilder`, phc/learning/amp_network_builder.py

The network builder class that extends the base A2CBuilder (from the rl_games library) to construct an actor-critic network with an additional discriminator for AMP-style adversarial training.

- In the network initialization, it calls _build_disc(amp_input_shape) to construct the discriminator.
The discriminator consists of:
    - A multi-layer perceptron (MLP) with configurable hidden units, activations, and initializers.
    - A final linear layer outputting a single logit (for binary real/fake classification).

### Loss Computation & Adversarial Training

`IMAmpAgent`, phc/learning/im_amp.py

Compute disc_loss (BCE on logits)
Compute amp_reward = f(disc_agent_logit) → add to reward / advantage
Total loss = ppo_loss + disc_loss_weight * disc_loss + amp_weight * policy_term

Optimizer step
   ↓ Discriminator gets better at spotting fake motion
   ↓ Policy gets gradient to produce more realistic motion

-----------

`RLGPUEnv`


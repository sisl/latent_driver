# latent_driver
Jointly learning policies and latent representations for driver behavior.


Utilities and scripts used to perform experiments described in "[Imitating Driver Behavior with Generative Adversarial Networks](https://arxiv.org/abs/1701.06699)". Built on [rllab](https://github.com/openai/rllab) and source code for [generative adversarial imitation learning](https://github.com/openai/imitation.git).


![](https://github.com/sisl/gail-driver/blob/master/gifs/congested.gif?raw=true)
An ego vehicle trained through Generative Adversarial Imitation Learning (blue) navigating a congested highway scene.

# Requirements
[AutomotiveDrivingModels.jl](https://github.com/tawheeler/AutomotiveDrivingModels.jl)

ForwardNets.jl ([nextgen branch](https://github.com/tawheeler/ForwardNets.jl/tree/nextgen))

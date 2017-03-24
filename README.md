# latent_driver
Jointly learning policies and latent representations for driver behavior.

The video below illustrates the different driver classes used in training the encoder and policies.
![](https://github.com/jgmorton/latent_driver/blob/master/gifs/passive_aggressive.gif?raw=true)

Below we can see the how the encoder chooses to represent trajectories from different driver classes as training progresses.
![](https://github.com/jgmorton/latent_driver/blob/master/gifs/latent.gif?raw=true)

Once we have a trained policy, we can propagate trajectories by passing observations and samples from the latent space into the policy and using the actions to propagate the scene forward. If we initialize a vehicle at $$20 m/s$$





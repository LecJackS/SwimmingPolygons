# Version 2: 2D env

This first solution produced acceptable results, but very unstable as it greatdly depends on the reward function and the penalty-patches used to solve environment flags.

It helped me to understand better where to focus my attention and how to improve the code, both for stability of the agent (no env hacking or stucking into undesirable solutions) and for the performance of the agent.

Ideas:

* Use physics equations of pushing with friction (x(t), v(t), a(t), mass, mu)
  * ie. when the triangle flaps, it generates a force F, that can be applied to the mass of the triangle sliding over a surface with friction mu (static and dinamic)

* Instead of giving the coordinates of food (as part of the state), give the distance to it (eg. can "smell" it)
  
  * Q: is this enough? or it needs previous distances too to estimate direction? 

* Rotations on the triangle points should only be applied when rendering

  * we can keep updating a simple vector for the speed and direction
  * the force generated by the triangle while flappind does not depend on the direction, so it can be kept static


![./media/swimming_agent.gif](./media/swimming_agent.gif)

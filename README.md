# Multi-player Adversarial Learning 

The goal is to explore the behavior of generative adversarial networks (GANs) in a setup with >2 agents. 
Initial exploration: 
- one generator, 
- playing against 2 or more discriminators, 
- discriminators decide under consensus protocol
  - consensus model on of: {majority, unanimity, min, max}
  
## Implementation Issues
- Specifying discriminator population
- Coordinating decisions with discriminator population
  - incl. coordinated training of discriminator population
- Generator: random selection of discriminator output for generator update?
  - discriminator throughput combine (avg)
  - select winning D(x)? (max)
  - select losing D(x)? (min)
- noise injection 
  - need new math to justify...
- discriminator scoping id procedure is sub-optimal. non-unique names for panel sizes > 9
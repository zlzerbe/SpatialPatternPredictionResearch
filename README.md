This repository contains code for modeling and solving a 2D Gray-Scott System, as well as a machine learning module capable of being trained to perform categorical or regression 
tasks involving the system's data. 

The 2D Gray-Scott System involves a set of reaction diffusion equtions given by the chemical reaction A + 2B -> 3B . An initial system involves two NxN arrays of gray scale values representing the concentrations of reactant A and B. The arrays are first randomly given values, then a set amount of KxK (usually 10x10) 'splotches' of 100% reactant A are added to the concentration array of B, and vice versa for the array of A. The system is simulated going to 'equilibrium' by using a difference formula for a sufficient number of steps.

The draw(A,B) method uses matplotlib to plot both the A and B arrays. Visiualizing system parameter sets is very important to getting viable data for training. 

It's important to note that the system does not always reach a desirable equilibria. In fact, often times the concentration of one reactant will go completely to 1 while the other goes comeptletly to 0, and thus no pattern is created. The system reaches a viable equilibria onlyt for a specifc set of 4 parameters invovled in the systenms. These paramters are DA and DB, the diffusion constants of the two reactants respectively, as well as feed, the rate at which A is being added to the system, and kill, the rate at which B is being taken away from the system. We did not classify our patterns to strictly, labelely parameter sets loosely based on Pearson's classification (citation needed).  
 


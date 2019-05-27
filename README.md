# Q-Learning_GridWorld

![png](question.png)  

![png](question_.png)  

![png](question__.png)  

### Output:  

The boltzmann exploration temperature is calculated in each cell based on the number of visits made to that state. Hence, the exploration decreases with a fixed scheduling rate.  

From the plots and the Q learning matrix values it can be seen that the agent learns an optimal policy in all the cases. Though it learns the fastest when temperature is 10.  

Iteration:20000  

0.206| 0.229| 0.254| 0.282| 0.314| 0.349| 0.314| 0.282| 0.314| 0.349|  
0.229| 0.254| 0.282| 0.314| 0.349| 0.387| 0.349| 0.314| 0.349| 0.387|  
0.206| w | w | w | w | 0.430| w | w | w | 0.430|  
0.185| 0.206| 0.229|-1.000| w | 0.478| 0.531| 0.590| 0.531| 0.478|   
0.206| 0.229| 0.254| 0.229| w |-1.000|-1.000| 0.656| 0.590| 0.531|  
0.229| 0.254| 0.282| 0.254| w | g=1 |-1.000| 0.729|-1.000| 0.478|   
0.254| 0.282| 0.314| 0.282| w | 1.000| 0.900| 0.810|-1.000| 0.531|   
0.282| 0.314| 0.349|-1.000| w |-1.000|-1.000| 0.729| 0.656| 0.590|   
0.314| 0.349| 0.387| 0.430| 0.478| 0.531| 0.590| 0.656| 0.590| 0.531|   
0.282| 0.314| 0.349| 0.387| 0.430| 0.478| 0.531| 0.590| 0.531| 0.478|   

> | > | > | > | > | v | < | < | > | v |  
> | > | > | > | > | v | < | < | > | v |  
^ | w | w | w | w | v | w | w | w | v |  
> | > | v | -1,<| w | > | > | v | < | < |  
> | > | v | < | w | -1,v| -1,>| v | < | < |   
> | > | v | < | w | g=1 | -1,<| v | -1,<| ^ |  
> | > | v | < | w | ^ | < | < | -1,<| v |  
> | > | v | -1,v| w | -1,^| -1,^| ^ | < | < |  
> | > | > | > | > | > | > | ^ | < | < |  
> | > | > | > | > | > | > | ^ | < | < |  

![png](Convergence,%20Experiment_(boltzmann_1).png)

![png](Convergence,%20Experiment_(boltzmann_5).png)

![png](Convergence,%20Experiment_(boltzmann_10).png)

![png](Convergence,%20Experiment_(boltzmann_20).png)

![png](Convergence,%20Experiment_(e-greedy_0.1).png)

![png](Convergence,%20Experiment_(e-greedy_0.2).png)

![png](Convergence,%20Experiment_(e-greedy_0.3).png)

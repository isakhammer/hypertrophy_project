# the_trainer
Pythonscript to automate the task from a personal trainer 


## Why
Help client to obtain his goals in terms of strength, hypertrophy or endurance.


## Terms
The terms that will be used in this setting.

### Medical Terms
According to wikipedia is variables to reach the goals of strengh training

- Volume
    In medical industry is volume defined as total set for a given muscle, which does not make sense since volume is a term defined as space. 

- Intensity
    Intensity will be defined as the weight from a certain prosentage of a given estimated max weight (1RM). I am happy with this definition

- Frequenzy
    Frequenzy is in medical term defined as "exercises per week", but that is a poor statement since increasing frequenzy does not actual meaning increasing total load and does not necessarly help you increase strength. 

To be able to get statistical advantage is it therefore necessarry to generalize the terms so it is possible to do any calculations at all.

### Work
The term work will be used since it is well defined in physics as W = Force*Distance and described the amount of energy needed to lift a weight, which opens for way for generalized applications. For simplication will the total work be defined as total set of a given Weight. We could start computing the distance travelled for the weight as well, but it will add unintendent complexity.

### Intensity
Intensity I will be defined as the weight from a certain prosentage of a given estimated max weight (1RM). Intensity ratio I_r defined as the ratio inbetween weight and estimated max weight.

### Power
In physics is Power defined as P = Energy/Time which is our case is P = Work/Time = dW/dt. This will be a more generalized approach than using the term "frequenzy" since it does now include the amount of work in a certain time. 

### Stamina
The clients ability to produce power. 

## Power and Intensity
One of the major issues is that it is necessarly to introduce a relation inbetween Power and Intensity for being able to successfully produce a training program. Obviously is the training goal dependent on intensity which will affect the total power produces, therefore is it interesting to define the factor dP/dI to identify  and variance dI/dt and mue = I_avg for determining the intensity variation for the goal of the client. 



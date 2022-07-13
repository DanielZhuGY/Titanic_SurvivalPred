# Titanic_SurvivalPred
## Topic: Titanic - Machine Learning from Disaster

An entry level ML competition from [**kaggle**](https://www.kaggle.com/competitions/titanic/overview).

## About the Model
- Method: Logistic Regression
- Fitting: SGD
- Accuracy: 77%

## Analysis

### Parameters

Let's have a look at the data. Try to dig out some info.

- The route of Titanic

![alt text](https://github.com/DanielZhuGY/Titanic_SurvivalPred/blob/main/image/route.png?raw=true)

[pic_source](https://titanicfacts.net/titanic-maiden-voyage/)

Survival rate: C(France)>=Q(Ireland)>S(UK)

- The lives of women and children were to be saved first

![alt text](https://github.com/DanielZhuGY/Titanic_SurvivalPred/blob/main/image/fsr.png?raw=true)
![alt text](https://github.com/DanielZhuGY/Titanic_SurvivalPred/blob/main/image/msr.png?raw=true)

 |Category|  Male|  Female | MaleRate|  FemaleRate  |Rate|
 |--------|------|---------|---------|--------------|----|
|     0-7|    26  |    24|      61.5   |     75.0 | 68.0
|  18-50 |  335   |  176 |     18.8  |      76.7  |38.7
|   7-18  |  45    |  44  |    17.8   |    63.6 | 40.4
|   50-90   | 47     | 17 |     12.8   |     94.1 | 34.4
| Unknown   |124      |53  |    12.9    |    67.9 | 29.4

- The lowest resecue rate is in the Third class.

![alt text](https://github.com/DanielZhuGY/Titanic_SurvivalPred/blob/main/image/pclss.png?raw=true)



### Embarked place, Age category, Gender and Cabin Class will be our the main focus in our ML model.

|Parameters|Category|
|--------|-----------|
|Embarked|Cherbourg_1,Else_0|
|Pclass|Third Class_1,Else_0|
|Sex|Female_1, Male_-1|
|Age|Children_2,Else_1,Unknown_0|


## Model
### Logistic Regression

$f(x) = \theta^T x$ 


<sub><sup>
  -$\theta$ is our estimator
  </sup></sub>
  
Sigmoid function :

$h_\theta (x) = \frac{1}{(1+e^{-\theta^T x})}$ 

Derivative:

$\frac{d}{d\theta} h_\theta (x) = (1-h_\theta (x))h_\theta (x) f(x)'$

Loss Function (Cross-Entropy Loss):

$J(\theta) = -\frac{1}{m} \sum\limits_{i=1}^m \[y^{(i)}log(h_\theta (x^{(i)})) + (1-y^{(i)})log(1-h_\theta (x^{(i)}))\]$

Derivative:

$\frac{d}{d\theta} J(\theta) =  -\frac{1}{m} \sum\limits_{i=1}^m \[ x^{(i)}(y^{(i)}-h_\theta (x^{(i)})) \]$



### Regression

SGD
|Parameter Name|Value|
|-|-|
|Batch_size|80|
|Epoches|2000|
|Learning_rate|0.001|
|Iteration|30|

$\theta^{(n+1)} = \theta^{(n)} - \alpha  \frac{d}{d\theta^{(n)}} J(\theta^{(n)})$

<sub><sup>
  -$\alpha$ is learning rate
  </sup></sub>

### Performance
- Convergence Performance

![Convergence](https://github.com/DanielZhuGY/Titanic_SurvivalPred/blob/main/image/converge.png?raw=true)



- The accuracy of this model is 77%.

## Future work

- Adjust data pretreatment method. Improve acc to 80%.

- Apply Artifitial Neural Network on this task.

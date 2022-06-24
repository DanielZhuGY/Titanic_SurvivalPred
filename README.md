# Titanic_SurvivalPred
## Topic: Titanic - Machine Learning from Disaster

An entry level competition in [**kaggle**](https://www.kaggle.com/competitions/titanic/overview).

## About the Model
- Method: Logistic Regression
- Fitting: SGD
- Accuracy: approximately 75%

## Analysis

### Main Parameters
- The route of Titanic

![alt text](https://github.com/DanielZhuGY/Titanic_SurvivalPred/blob/main/image/route.png?raw=true)

[pic_source](https://titanicfacts.net/titanic-maiden-voyage/)

- Be gentle! Lady first.

![alt text](https://github.com/DanielZhuGY/Titanic_SurvivalPred/blob/main/image/fsr.png?raw=true)
![alt text](https://github.com/DanielZhuGY/Titanic_SurvivalPred/blob/main/image/msr.png?raw=true)

- Children first.

 |Category|  Male|  Female | MaleRate|  FemaleRate  |Rate|
 |--------|------|---------|---------|--------------|----|
|     0-7|    26  |    24|      61.5   |     75.0 | 68.0
|  18-50 |  335   |  176 |     18.8  |      76.7  |38.7
|   7-18  |  45    |  44  |    17.8   |    63.6 | 40.4
|   50-90   | 47     | 17 |     12.8   |     94.1 | 34.4
| Unknown   |124      |53  |    12.9    |    67.9 | 29.4

From the table above, we can notice that the survive rate of children is highest. The survive rate of adult male is less than 20%. 

- The lowest resecue rate is in the Third class.

![alt text](https://github.com/DanielZhuGY/Titanic_SurvivalPred/blob/main/image/pclss.png?raw=true)

> The water enters the fastest after the impact.
> 
> Inadequate emergency escape measures.

### Secondary Parameters

- Fare

Pclass has already included the information of fare. And there are some missing information in fare column. Thus, we are
 not going to use fare as a parameter.


### Age category, Gender and Class will be our main focus in our ML model.

|Parameters|Explanation|Category|
|--------|-----------|----|
|Pclass|Passangers in Cabin 3 are less likely to survive compare to the rest Cabin|First Class_2, Second Class_1, Third Class_-1|
|Sex|Male passangers are less likely to survive|Male_0, Female_1|
|Age|Children have higher survive rate|Children_2,Else_1,Unknown_0|


## Model
### Logistic Regression

$ f(x) = \theta^T x $ 


<sub><sup>
  -$\theta$ is our estimator
  </sup></sub>
  
Sigmoid function :

$ h_\theta (x) = \frac{1}{(1+e^{-\theta^T x})} $ 

Derivative:

$ \frac{d}{d\theta} h_\theta (x) = (1-h_\theta (x))h_\theta (x) f(x)' $

Loss Function (Cross-Entropy Loss):

$ J(\theta) = -\frac{1}{m} \sum\limits_{i=1}^m \[y^{(i)}log(h_\theta (x^{(i)})) + (1-y^{(i)})log(1-h_\theta (x^{(i)}))\] $

Derivative:

$ \frac{d}{d\theta} J(\theta) =  -\frac{1}{m} \sum\limits_{i=1}^m \[ x^{(i)}(y^{(i)}-h_\theta (x^{(i)})) \]$



### Regression

SGD
|Parameter Name|Value|
|-|-|
|Batch_size|100|
|Learning_rate|0.01|
|Iteration|20000|

$ \theta^{(n+1)} = \theta^{(n)} - \alpha  \frac{d}{d\theta^{(n)}} J(\theta^{(n)}) $

<sub><sup>
  -$\alpha$ is learning rate
  </sup></sub>


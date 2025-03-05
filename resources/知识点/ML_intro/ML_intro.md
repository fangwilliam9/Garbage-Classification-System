# Intro

#### Supervised learning

- regression models (output a numeric value)

   (房价预测，到达目的地耗时预测)

- classification models 

   - 二分类binary(`rain`/`no`) 
   - 多分类multiclass(`rain`, `no`, `hail`下冰雹, `snow`, or `sleet`雨夹雪) 



#### Un-supervised learning

聚类Clustering differs/distinguishes from classification because the categories aren't defined by you.

#### Reinforcement learning

Reinforcement learning models make predictions by getting ==rewards or penalties== based on actions performed within an environment. 

A reinforcement learning system generates a policy that defines the best strategy for getting the most rewards. 

#### Generative AI





# Supervised Learning 

## Data数据

Datasets are made up of individual [examples](https://developers.google.com/machine-learning/glossary#example) that contain [features](https://developers.google.com/machine-learning/glossary#feature) and a [label](https://developers.google.com/machine-learning/glossary#label). You could think of an example as analogous to a single row in a spreadsheet. Features are the values that a supervised model uses to predict the label. The label is the "answer," or the value we want the model to predict. In a weather model that predicts rainfall, the features could be *latitude*, *longitude*, *temperature*, *humidity*, *cloud coverage*, *wind direction*, and *atmospheric pressure*. The label would be *rainfall amount*.

Examples that contain both features and a label are called [labeled examples](https://developers.google.com/machine-learning/glossary#labeled-example).

### dataset characteristics

A dataset is characterized by its size and diversity. Size indicates the number of examples. Diversity indicates the range those examples cover. Good datasets are both large and highly diverse.

> For instance, a dataset might contain 100 years worth of data, but only for the month of July. Using this dataset to predict rainfall in January would produce poor predictions. Conversely, a dataset might cover only a few years but contain every month. This dataset might produce poor predictions because it doesn't contain enough years to account for variability.

## Model模型

In supervised learning, a model is the complex collection of numbers that define the mathematical relationship ==from== specific input feature patterns ==to== specific output label values. 

The model ==discovers these patterns== through training.

## Training训练

Before a supervised model can make predictions, it must be trained. 

To train a model, we give the model a dataset with labeled examples. 

The model's goal is to work out the <u>best solution for predicting the labels from the features</u>. 
The model finds the best solution by comparing its predicted value to the label's actual value. 
Based on ==the difference between the predicted and actual values—defined as the [loss](https://developers.google.com/machine-learning/glossary#loss)==—the model gradually updates its solution. 
In other words, the model learns the mathematical relationship between the features and the label, so that it can make the best predictions on unseen data.

> For example, if the model predicted `1.15 inches` of rain, but the actual value was `.75 inches`, the model modifies its solution so its prediction is closer to `.75 inches`. 
> After the model has looked at each example in the dataset—in some cases, multiple times—it arrives at a solution that makes the best predictions, on average, for each of the examples.



## Evaluating评估

We evaluate a trained model to determine ==how well it learned==. 

When we evaluate a model, we use a labeled dataset, but we only give the model the dataset's features. We then compare the model's predictions to the label's true values.



## Inference推理

Once we're satisfied with the results from evaluating the model, we can use the model to make predictions, called [inferences](https://developers.google.com/machine-learning/glossary#inference), on unlabeled examples. 

In the weather app example, we would give the model the current weather conditions—like temperature, atmospheric pressure, and relative humidity—and it would predict the amount of rainfall.













# tensor

**tensor（张量）**是一个广义的数学概念，可以表示标量、向量、矩阵以及更高维度的数据结构。具体来说：

|          |                     |                              |                                    | 例如                 |
| -------- | ------------------- | ---------------------------- | ---------------------------------- | -------------------- |
| 标量     | Scalar              | 零阶张量0-dimensional tensor | 没有方向，只有一个数值             | `5`，`3.14`          |
| 向量     | Vector              | 一阶张量1-dimensional tensor | 一个有序的数值列表，具有大小和方向 | `[1, 2, 3]`          |
| 矩阵     | Matrix              | 二阶张量2-dimensional tensor | 一个二维数组，由行和列组成         | `[ [1, 2], [3, 4] ]` |
| 高阶张量 | Higher-order Tensor | 更高维度的张量               | 可以表示更复杂的数据结构           |                      |

### 总结
- **标量**：0阶张量（0维）。
- **向量**：1阶张量（1维）。
- **矩阵**：2阶张量（2维）。
- **高阶张量**：3阶及以上（3维及以上）。

在深度学习和科学计算中，张量是一个核心概念，因为它可以灵活地表示各种数据类型和结构。


































































































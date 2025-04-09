# 📊 Customer Churn Prediction with MLlib

This project uses **Apache Spark MLlib** to predict customer churn based on structured customer data. You'll go through **data preprocessing**, **feature engineering**, **model training**, **feature selection**, and **hyperparameter tuning with cross-validation**.

---

## 🎯 Objective

Build and compare machine learning models using PySpark to predict whether a customer will churn based on their service usage and subscription features.

---

## 📁 Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

---

## ⚙️ Prerequisites

- Apache Spark installed and configured  
- Python 3.x environment  
- `pyspark` installed  
- `customer_churn.csv` in the root directory

```bash
pip install pyspark
```

---

## 🚀 Run the Project

To run the complete pipeline (all 4 tasks):

```bash
spark-submit churn_prediction.py
```

This will generate outputs in the `output/` directory.

---

## ✅ Tasks Breakdown

---

### 🔹 Task 1: Data Preprocessing and Feature Engineering

**Goal:** Prepare the dataset for machine learning models.

#### Code Explanation:
```python
df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges")))

categorical_cols = ["gender", "PhoneService", "InternetService"]
indexers = [StringIndexer(inputCol=c, outputCol=c + "_Index") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c + "_Index", outputCol=c + "_Vec") for c in categorical_cols]

numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
assembler_inputs = [c + "_Vec" for c in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

pipeline = Pipeline(stages=indexers + encoders + [assembler])
model = pipeline.fit(df)
final_df = model.transform(df)

final_df = final_df.withColumn("ChurnIndex", when(col("Churn") == "Yes", 1.0).otherwise(0.0))
```

#### Sample Output (`output/task1_output.txt`):
```
===== Task 1: Features with ChurnIndex =====
+--------------------+----------+
|            features|ChurnIndex|
+--------------------+----------+
|[1.0,1.0,1.0,0.0,...|       1.0|
|(7,[4,5,6],[9.0,7...|       0.0|
|[1.0,0.0,1.0,0.0,...|       1.0|
|[1.0,1.0,1.0,0.0,...|       1.0|
|[1.0,0.0,1.0,0.0,...|       1.0|
+--------------------+----------+
only showing top 5 rows
```

---

### 🔹 Task 2: Train and Evaluate Logistic Regression Model

**Goal:** Train a Logistic Regression model and evaluate using AUC.

#### Code Explanation:
```python
train, test = df.randomSplit([0.8, 0.2], seed=42)
lr = LogisticRegression(featuresCol="features", labelCol="ChurnIndex")
model = lr.fit(train)
predictions = model.transform(test)

evaluator = BinaryClassificationEvaluator(labelCol="ChurnIndex")
auc = evaluator.evaluate(predictions)
```

#### Sample Output (`output/task2_output.txt`):
```
===== Logistic Regression AUC =====
AUC: 0.7313
```


---

### 🔹 Task 3: Feature Selection using Chi-Square Test

**Goal:** Select top 5 features most relevant to predicting churn.

#### Code Explanation:
```python
selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="ChurnIndex")
result = selector.fit(df).transform(df)
selected_output = result.select("selectedFeatures", "ChurnIndex")
```

#### Sample Output (`output/task3_output.txt`):
```
===== Top 5 Selected Features =====
+--------------------+----------+
|    selectedFeatures|ChurnIndex|
+--------------------+----------+
|[1.0,0.0,3.0,66.7...|       1.0|
|(5,[2,3,4],[9.0,7...|       0.0|
|[1.0,0.0,27.0,95....|       1.0|
|[1.0,0.0,5.0,77.3...|       1.0|
|[1.0,0.0,14.0,72....|       1.0|
+--------------------+----------+
only showing top 5 rows
```

---

### 🔹 Task 4: Hyperparameter Tuning and Model Comparison

**Goal:** Tune multiple ML models and compare their performance using 5-fold Cross-Validation.

#### Code Explanation:
```python
models_with_params = [
    {
        "name": "LogisticRegression",
        "model": LogisticRegression(labelCol="ChurnIndex"),
        "paramGrid": ParamGridBuilder()
            .addGrid(LogisticRegression().regParam, [0.01, 0.1])
            .addGrid(LogisticRegression().maxIter, [10, 20])
            .build()
    },
    ...
]

cv = CrossValidator(estimator=model,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)
cv_model = cv.fit(train)
auc = evaluator.evaluate(cv_model.transform(test))
```

#### Sample Output (`output/task4_output.txt`):
```
===== Cross-Validation Model Comparison =====

Tuning LogisticRegression...
LogisticRegression Best Model Accuracy (AUC): 0.73
Best Params for LogisticRegression: regParam=0.0, maxIter=100

Tuning DecisionTree...
DecisionTree Best Model Accuracy (AUC): 0.66
Best Params for DecisionTree: maxDepth=5

Tuning RandomForest...
RandomForest Best Model Accuracy (AUC): 0.76
Best Params for RandomForest: maxDepth=5, numTrees=20

Tuning GBT...
GBT Best Model Accuracy (AUC): 0.73
Best Params for GBT: maxDepth=5, maxIter=20
```

---

## 📌 Summary

| Model              | Best AUC | Best Params                     |
|-------------------|----------|---------------------------------|
| Logistic Regression | 0.73     | regParam=0.0, maxIter=100       |
| Decision Tree      | 0.66     | maxDepth=5                      |
| Random Forest      | 0.76     | maxDepth=5, numTrees=20         |
| GBT                | 0.73     | maxDepth=5, maxIter=20          |

✅ **Best model based on AUC**: **Random Forest**

---

## 🧠 Conclusion

This project demonstrates a full ML pipeline in PySpark:

- Clean and prepare structured data  
- Apply encoding and feature engineering  
- Train and evaluate classification models  
- Perform feature selection  
- Compare ML models using cross-validation  

---

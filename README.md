
# Capstone 2 Report 2

Team 6  
Duyen Do  
Lakshmi Kambathanahally Lakshminarasap  
Jessica Nguyen  
Shrinidhi Sudhir  

BZAN 6361: Capstone II  
April 19th, 2024  

## Table of Contents

1. [Executive Summary and Recommendations](#executive-summary-and-recommendations)
2. [Reflections](#reflections)
    - [Strengths from previous findings](#strengths-from-previous-findings)
    - [Opportunities for improvement](#opportunities-for-improvement)
3. [Analysis](#analysis)
    - [Understand the target variable “HasCLABSI”](#understand-the-target-variable-hasclabsi)
    - [Addressing missing values in the dataset based on data type and correlation with the target variable](#addressing-missing-values-in-the-dataset-based-on-data-type-and-correlation-with-the-target-variable)
    - [Reduce dimensionality with correlation matrix and pattern matching](#reduce-dimensionality-with-correlation-matrix-and-pattern-matching)
    - [Refine and enhance the target variable to improve the prediction of CLABSI within 3 days](#refine-and-enhance-the-target-variable-to-improve-the-prediction-of-clabsi-within-3-days)
4. [Modeling Design](#modeling-design)
    - [Neural Networks (NN)](#neural-networks-nn)
    - [k-Nearest Neighbors (kNN)](#k-nearest-neighbors-knn)
    - [Naïve Bayes](#naïve-bayes)
    - [Decision Tree](#decision-tree)
5. [Evaluation Plan](#evaluation-plan)
    - [Overcome overfitting and data leakage](#overcome-overfitting-and-data-leakage)
    - [Overcome imbalanced data](#overcome-imbalanced-data)
    - [Metrics to evaluate models’ performance](#metrics-to-evaluate-models-performance)
    - [Performance adjustment within algorithms based on models’ inherent biases](#performance-adjustment-within-algorithms-based-on-models-inherent-biases)
6. [Further on Recommendations](#further-on-recommendations)
    - [The trade-off between type 1 and type 2 errors](#the-trade-off-between-type-1-and-type-2-errors)
    - [The practicality of a cost matrix](#the-practicality-of-a-cost-matrix)

## Executive Summary and Recommendations

### Challenge:
Central Line-Associated Bloodstream Infections (CLABSI) remain a prominent life-threatening threat to PICU patients. While timely and early detection is critical, vast amounts of data and limited adaptation of machine learning methods in healthcare have revealed shortcomings in this area.

### Recommendations:
- Utilizing a cost-matrix, a cost-sensitive model, and weighted accuracy can help decision makers optimize the results to fit the clinical setting.
- Healthcare providers should consider models that minimize type 2 errors to save lives while preserving the institutions’ quality ratings and reputation.

### Key findings:
- Advanced and tailored approaches are necessary to develop predictive machine learning models. This tailored approach is implemented throughout the process from data-preprocessing to model evaluation.
- Four models were examined, each with its own strength and inherent biases that require further adjustments.
- Decision tree models exhibit the most balanced performance across crucial metrics.
- Additional solutions worth considering to incorporate and complement the current approach:
    - Transferred learning: leverage pretrained models for effectiveness
    - Multiple models: employ several different models at distinct stages of the process.

## Reflections

### Strengths from previous findings:
1. **Thorough data preprocessing and strategic planning for later stages**:
   - A meticulous approach to data preparation allowed us to address multiple challenges within the dataset such as sparse data and feature engineering. This foundational step strengthens the impact and effectiveness of our models in later stages.

2. **Varied data analysis techniques including advanced statistical modeling**:
   - The team effectively utilized a variety of models from Python's libraries (Pandas, NumPy, Seaborn, Matplotlib, Statsmodels, and linearmodels) to handle a broad spectrum of data analysis tasks. Additionally, we implemented advanced statistical models like PanelOLS to uncover data relationships. This broad array of advanced models allows us to confidently draw data-driven decisions and recommendations based on in-depth analysis.

3. **Effective data visualization to communicate findings in a business context**:
   - Analysts successfully implemented visualization tools like Seaborn and Matplotlib to communicate technical findings in a business context. These visualizations effectively conveyed the direction and scope of the complex analysis to clients.

### Opportunities for improvement:
1. **Explore and employ explicit data validation and dynamic data handling measures**:
   - The team will investigate and implement additional methods for data validation to address data quality issues encountered. Considering the dynamic nature of healthcare factors and patient data, we aim to transition and broaden our model from descriptive analysis to more adaptable models that can handle dynamic data attributes, thereby improving scalability.

2. **Optimize queries and documentation to improve clarity**:
   - There are opportunities to streamline our queries with additional comments. This will allow easier model fine-tuning and will help collaborators grasp the scope of our script.

## Analysis

### Understand the target variable “HasCLABSI”
- The target variable “HasCLABSI” attribute in the data frame is a binary categorical variable indicating whether a patient has contracted CLABSI (True) or not (False). 
- The distribution of these values is as follows:
    - True: The patient has contracted CLABSI in 52 instances
    - False: The patient has not contracted CLABSI in over 14,000 instances
- This reveals a significant class imbalance with more instances of patients not contracting CLABSI compared to those who did. This imbalance presents challenges in predicting infection rates. If the majority class (no CLABSI) is not handled appropriately, a model can learn to focus on and predict the majority class while ignoring the minority class, which is our target group. Addressing class imbalance is crucial when building predictive models to ensure accurate and meaningful results.

### Addressing missing values in the dataset based on data type and correlation with the target variable
- The dataset contains numerous missing values, which in clinical datasets can introduce significant bias for models. Hence, handling missing values is a critical part of data preprocessing. A targeted approach for addressing missing values is implemented considering both the data type and its correlation with the target variable.

- **Process for managing missing values**:
    - With EDA, understand the sparsity of the dataset and the types of columns affected. Then categorize missing values by data type (object, Boolean, numeric).
    - **Object data type**: For object data type, missing values are filled with the string 'NA' which serves as a placeholder for unknown or missing categorical values.
    - **Numeric data type**: Missing values are handled based on the target variable “HasCLABSI” due to the aforementioned class imbalance. Hence, a tailored approach to handle missing values is necessary.
        - For instances where “HasCLABSI” is True, missing values are filled with the median of non-missing values within the corresponding data subset. The median is preferred due to its central tendency and resilience to skewed data.
        - Similarly, for rows where “HasCLABSI” is False, missing values are replaced with the median of non-missing values within the corresponding data subset.

### Reduce dimensionality with correlation matrix and pattern matching
- The dataset initially contains 278 attributes. However, a high number of attributes can hinder performance, especially since not all dimensions carry the same effect on the target variable. Hence, new methods for dimensionality reduction were developed, resulting in a new dataset with 168 attributes.

- **Process to reduce dimensionality**:
    - **Correlation matrix**: Utilize the correlation matrix and set a threshold to understand each attribute's impact on the target variable. This information is used in an automated procedure to identify and remove highly correlated columns that introduce redundancy and inefficiency.
    - **Pattern matching**: Numeric details are extracted from column names to facilitate dimensionality reduction. For example, when column names include numeric identifiers such as "MedicationsInjectedLast30", "15", "5", "3", and "2" days, numeric suffixes are extracted based on a specified pattern, retaining only the column with the highest numeric suffix for each prefix. This pattern-matching technique ensures the retention of the most relevant features and helps prevent overfitting, improving the model's generalizability.
    - A 40% reduction in dimensionality simplifies the dataset while preserving crucial information.

### Refine and enhance the target variable to improve the prediction of CLABSI within 3 days
- Early detection of CLABSI allows healthcare providers to employ timely intervention strategies. To refine the prediction, two new target variables are created to capture the risk of CLABSI occurrence over time: “HasCLABSI_NextDay” and “HasCLABSI_Next3Days.” These new target variables provide more granular and actionable insights into the risk of CLABSI development over time.

- **Process for determining and employing this approach**:
    - **Sorting**: First, the data frame is sorted by 'PatientKey' and 'Date' columns. This is essential for subsequent operations because later queries involve shifting values based on patient identifiers in chronological order.
    - **“HasCLABSI_NextDay”**: This variable indicates whether a patient is at risk of developing a CLABSI infection the following day. The value of "HasCLABSI" is shifted by -1 for each patient group to generate this new variable.
    - **“HasCLABSI_Next3Days”**: This variable is built on the previous step and captures the risk of developing CLABSI within the next three days

. This variable is constructed by combining the shifted "HasCLABSI" values for the next three days using conditional logic operations. As a result, if a patient is at risk at any time within the next three days, it is reflected in the target variable.

## Modeling Design

### Neural Networks (NN)
- Given the current dataset’s characteristics - high-dimensional data structure, missing values, and complex data patterns - NN is a model worth considering. NN’s interconnected nodes across multiple layers allow it to process data effectively and exhibit flexibility in detecting patterns within complex datasets.

#### Advantages of NN for this dataset:
- **Sequential dependency**: NN is adept at capturing sequential dependencies, making it well-suited for time series forecasting and other time-dependent duties.
- **Nonlinear relationships**: The model is suitable for tasks where the relationships between predictors and the target variable are nonlinear and hard to interpret.
- **Uninformative data**: It also works well with uninformative and sparse data and can handle class imbalance with ease.

#### Challenges of NN for this dataset:
- **Lack of interpretability**: Neural networks lack interpretability due to intricate architecture.
- **Overfitting**: NN tends to overfit. To resolve these challenges:
    - **Adjust hidden layers**: Adjust the number of nodes in the hidden layer.
    - **Adaptive learning rate**: Monitor and adjust the learning rate (η) during training.

### k-Nearest Neighbors (kNN)
- kNN is an instance-based learning algorithm mostly used for classification tasks.

#### Advantages of kNN for this dataset:
- **Simplicity and efficiency**: kNN is notable for its ease of implementation and ability to handle large and complex data.
- **Flexibility**: It provides flexibility in handling missing data.
- **Adaptability**: kNN belongs to a group of “lazy learning” methods, storing the training dataset and calculating distances between new data points to make predictions.

#### Challenges of kNN for this dataset:
- **Computational cost**: Requires large memory capacity and expensive computational cost as the dataset size increases.
- **Conditional effectiveness**: Effectiveness depends on the specific characteristics of the panel data and the chosen distance metric. Standardizing or scaling the data is required before performing calculations.

### Naïve Bayes
- A probabilistic classifier that applies Bayes' theorem. When provided with a labeled dataset, it calculates the probability of each data point belonging to each class based on observed records, assuming strong independence between data points or features.

#### Advantages of Naïve Bayes for this dataset:
- **Efficient analysis**: Adept at providing timely insights in medical settings.
- **High dimensional data**: Can handle high-dimensional data, especially sparse data.
- **Impactful attributes**: Focuses only on the most informative variables for CLABSI detection.

#### Challenges of Naïve Bayes for this dataset:
- **Feature independence assumption**: May not always be true for all datasets, reducing flexibility in catching complex data patterns.
- **Loss of information**: Tends to ignore potential correlation and dependence of features.

### Decision Tree
- Decision Tree model splits the data into subsets using feature rules resulting in a tree with a collection of decision nodes and leaf nodes.

#### Advantages of Decision Tree for this dataset:
- **Simplicity and interpretability**: Easy to understand and interpret.
- **Effective handling of complex data**: Handles panel data well and provides understanding of the variables that contribute to CLABSI risk over time.
- **Discrete target variable**: Suitable for binary (HasCLABSI: True/False) target variables.
- **Feature selection**: Determines the most relevant risk factors that best separate the data into CLABSI vs. non-CLABSI classes.

#### Challenges of Decision Tree for this dataset:
- **Overfitting**: Prone to overfitting, where the tree becomes too complex. Solutions include pruning nodes and branches and applying certain selection techniques to keep the most relevant variables.

## Evaluation Plan

### Overcome overfitting and data leakage
- **Address data leakage with cross validation and withholding a small dataset**: 
    - Cross validation is employed to evaluate the model’s robustness and ensure it performs well on unseen data. Data preprocessing is done separately for each cross-validation step.
    - A small dataset can be reserved for evaluating the model’s performance. If the model yields abnormally high accuracy, data leakage has occurred.

- **Address overtly complicated models and overfitting with regularization**:
    - Regularization techniques are incorporated in predictive models like Neural Networks and Decision Tree to discourage overtly complex models. Techniques include pruning decision trees, reducing layers in neural networks, and applying selection techniques to keep the most relevant variables.

### Overcome imbalanced data
- The significant class imbalance noted with “HasCLABSI” is addressed with two balancing techniques: SMOTE and a tailored sampling approach.

- **Synthetic Minority Over-sampling Technique (SMOTE)**:
    - Balances the dataset by creating synthetic examples rather than over-sampling with replacement. SMOTE is implemented after the train-test split to avoid data leakage.

- **Tailored sampling technique**:
    - Different sampling techniques are implemented for different models to ensure the best representation of both data classes.

### Metrics to evaluate models’ performance:
- **Precision and Recall**: Important for datasets where the cost of a false negative is higher than a false positive.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC Score**: Represents a model's ability to discriminate between the classes at various threshold settings.
- **Confusion Matrix**: Visualizes the performance of the algorithm.
- **Accuracy**: Not the golden standard due to class imbalance.

### Performance adjustment within algorithms based on models’ inherent biases:
- Different threshold adjustments, such as sensitivity and specificity, are explored according to clinical needs.

#### Models’ performance overview:

| Metric                     | Best Performing | Contending | Reason for Fit                                                                                       |
|----------------------------|-----------------|------------|------------------------------------------------------------------------------------------------------|
| Precision                  | Decision Tree   | Naive Bayes| Decision Tree shows the highest precision for classifying class 1 instances correctly (0.59)         |
| Recall                     | Decision Tree   | Naive Bayes| Decision Tree also leads in recall for class 1 managing to identify 62% correctly                    |
| False Positives (Type 1)   | K-Nearest Neighbours | Decision Tree| K-Nearest Neighbours has more Type 1 errors compared to Decision Tree in relative terms               |
| False Negatives (Type 2)   | Decision Tree   | Naive Bayes| Decision Tree has fewer Type 2 errors compared to Naive Bayes aligning with higher recall             |
| Overall Model Accuracy     | Decision Tree   | Neural Network| Decision Tree and Neural Network both achieve high accuracy but Decision Tree performs slightly better in other metrics |
| Inference                  | Decision Tree   | Naive Bayes| Decision Tree exhibits better capability at identifying CLABSI at the cost of precision compared to other models. It suggests a more balanced approach than the Neural Network |

- **Precision and recall**: Decision Tree model shows its effectiveness in correctly identifying actual CLABSI cases while minimizing false alarms.
- **Type 1 and type 2 errors**: kNN tends to over-diagnose type 1 errors while Decision Tree has lower type 2 errors.

### Overall: Decision Tree model appears to be the most balanced:
- Exhibits the most balanced performance across all metrics, making it a compelling choice going forward.

## Further on Recommendations

### The trade-off between type 1 and type 2 errors
- Both type 1 errors (false positive) and type 2 errors (false negative) are detrimental and should be minimized. The nuanced trade-offs between detecting true positives and avoiding false alarms are pivotal in choosing the right model.

### The practicality of a cost matrix
- A cost matrix assigns different costs and benefits to each type of error depending on the decision-makers’ discretions. This approach can help decision-makers determine the optimal approach depending on the context of the problem.

- **Simplified example of a cost matrix**:

| Outcome                                   | Cost                  |
|-------------------------------------------|-----------------------|
| True positive (correctly identify infected) | Low cost              |
| True negative (correctly identify non-infected) | Low cost              |
| False positive (Type 1 error: incorrectly identify infection) | Medium cost (cost resources) |
| False negative (Type 2 error: incorrectly identify no infection) | High cost (patient’s health and life at risk) |


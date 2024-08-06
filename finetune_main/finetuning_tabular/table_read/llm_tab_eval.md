# LLM BGP tab eval

## llama 7B:
### Evaluation Results 20 split:
Evaluation Results:
Precision: 0.21
Recall: 0.41
F1 Score: 0.28
True Positives: 12
False Positives: 46
False Negatives: 17

### table135-2k-20split:
Evaluation Results 20 split:
Precision: 0.16
Recall: 0.17
F1 Score: 0.16
True Positives: 5
False Positives: 27
False Negatives: 24

### table135-5k-20split:
Evaluation Results 20 split:
Precision: 0.23
Recall: 0.24
F1 Score: 0.23
True Positives: 7
False Positives: 24
False Negatives: 22

### table135-5k-20split-instruct `best`:
Evaluation Results 20 split:
Precision: 0.43
Recall: 0.34
F1 Score: 0.38
True Positives: 10
False Positives: 13
False Negatives: 19

### table135-6k-20split-instruct:
Evaluation Results 20 split:
Precision: 0.24
Recall: 0.28
F1 Score: 0.25
True Positives: 8
False Positives: 26
False Negatives: 21


### table135-5k-20split-instruct-newparamsb:
Evaluation Results 20 split:
Precision: 0.22
Recall: 0.21
F1 Score: 0.21
True Positives: 6
False Positives: 21
False Negatives: 23


### table135-10k-20split-instruct:
Evaluation Results 20 split:
Precision: 0.32
Recall: 0.34
F1 Score: 0.33
True Positives: 10
False Positives: 21
False Negatives: 19

Evaluation Results 10 split:
Precision: 0.31
Recall: 0.44
F1 Score: 0.36
True Positives: 12
False Positives: 27
False Negatives: 15

### table143-20split-5k-1e5rate
Precision: 0.19
Recall: 0.55
F1 Score: 0.28
True Positives: 16
False Positives: 69
False Negatives: 13

### table143-20split-5k-2e5rate
Precision: 0.16
Recall: 0.17
F1 Score: 0.16
True Positives: 5
False Positives: 27
False Negatives: 24

### table142-20split-5k-instruct-1e5rate-shuffled
Precision: 0.13
Recall: 0.14
F1 Score: 0.14
True Positives: 4
False Positives: 26
False Negatives: 25

### table142-20split-5k-instruct-1e5rate-loraa8drop0"
Precision: 0.18
Recall: 0.21
F1 Score: 0.19
True Positives: 6
False Positives: 27
False Negatives: 23

### table135-20split-5k-instruct-2e5rate-loraa64drop01"
Precision: 0.14
Recall: 0.14
F1 Score: 0.14
True Positives: 4
False Positives: 25
False Negatives: 25

### table135-20split-5k-2e5rate-loraa64drop01"
Precision: 0.29
Recall: 0.28
F1 Score: 0.28
True Positives: 8
False Positives: 20
False Negatives: 21

### table135-20split-5k-1e5rate-loraa64drop01
Precision: 0.37
Recall: 0.24
F1 Score: 0.29
True Positives: 7
False Positives: 12
False Negatives: 22

### table135-20split-5k-1e5rate-loraa8drop01
Precision: 0.16
Recall: 0.21
F1 Score: 0.18
True Positives: 6
False Positives: 31
False Negatives: 23

### table283New-10split-10k-instruct-1e5rate-loraa64drop01"
Precision: 0.19
Recall: 0.63
F1 Score: 0.29
True Positives: 17
False Positives: 73
False Negatives: 10

### table283New-10split-5k-instruct-1e5rate-loraa64drop01
Precision: 0.15
Recall: 0.33
F1 Score: 0.21
True Positives: 9
False Positives: 50
False Negatives: 18

## llama 13B:
Evaluation Results:
Precision: 0.32
Recall: 0.48
F1 Score: 0.38
True Positives: 14
False Positives: 30
False Negatives: 15


## finetuned-13B:
### table135-20split-2k
Evaluation Results 20 split:
Precision: 0.29
Recall: 0.34
F1 Score: 0.31
True Positives: 10
False Positives: 25
False Negatives: 19


### table135-20split-5k
Evaluation Results 20 split:
Precision: 0.26
Recall: 0.45
F1 Score: 0.33
True Positives: 13
False Positives: 37
False Negatives: 16


### BGP-LLaMA13-BGPStream5k-cutoff-1024-max-2048-fpFalse"
Evaluation Results 20 split:
Precision: 0.42
Recall: 0.37
F1 Score: 0.39
True Positives: 10
False Positives: 14
False Negatives: 17

Evaluation Results 10 split:
Precision: 0.44
Recall: 0.56
F1 Score: 0.49
True Positives: 15
False Positives: 19
False Negatives: 12

New test:
Evaluation Results 20 split:
Precision: 0.30
Recall: 0.38
F1 Score: 0.33
True Positives: 27
False Positives: 64
False Negatives: 44

Evaluation Results 10 split:
Precision: 0.21
Recall: 0.58
F1 Score: 0.31
True Positives: 41
False Positives: 155
False Negatives: 30


### table135-20split-2k-tablellama


## llama 70B:
### Evaluation Results:


## gpt-4o (data analysis):
### Evaluation Results:

### Evaluation Results 570 rows:
Precision: 0.77
Recall: 0.56
F1 Score: 0.65
True Positives: 40
False Positives: 12
False Negatives: 31

| Model                  | Precision | Recall | F1 Score | True Positives | False Positives | False Negatives |
|------------------------|-----------|--------|----------|----------------|-----------------|-----------------|
| llama 7B               | 0.21      | 0.41   | 0.28     | 12             | 46              | 17              |
| bgp-llama-7B-finetuned | 0.43      | 0.34   | 0.38     | 10             | 13              | 19              |
| llama 13B              | 0.32      | 0.48   | 0.38     | 14             | 30              | 15              |
| gpt-4o                 | 0.94      | 0.59   | 0.73     | 16             | 1               | 11              |


Statistical Methodology for Identifying Anomalies
Calculate Summary Statistics:

Mean: The average value of each numerical feature.
Standard Deviation (Std): A measure of the amount of variation or dispersion of a set of values.
Set a Threshold for Anomalies:

Typically, anomalies are identified as data points that are significantly different from the majority of the data. One common method is to use the mean and standard deviation to set a threshold.
In this case, we define an anomaly threshold as:
Threshold=Mean+3×Std
This threshold is based on the empirical rule (or 68-95-99.7 rule), which states that for a normal distribution:
About 68% of the data falls within 1 standard deviation of the mean.
About 95% falls within 2 standard deviations.
About 99.7% falls within 3 standard deviations.
Therefore, data points beyond 3 standard deviations from the mean are considered anomalous.
Identify Anomalous Data Points:

For each numerical feature in the dataset, any data point that exceeds the calculated threshold is flagged as an anomaly.
This is done for each relevant feature individually. If a data point is anomalous in any feature, it is considered an anomaly.

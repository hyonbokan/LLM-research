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


### table135-20split-2k-tablellama


## llama 70B:
### Evaluation Results:

## gpt-3.5 (table format):
### Evaluation Results:
Precision: 0.83
Recall: 1.00
F1 Score: 0.91
True Positives: 5
False Positives: 1
False Negatives: 0


## gpt-4 (data analysis):
## Evaluation Results:
Precision: 0.03
Recall: 0.10
F1 Score: 0.05
True Positives: 3
False Positives: 85
False Negatives: 26

## gpt-4o (data analysis):
### Evaluation Results:
Precision: 0.04
Recall: 0.34
F1 Score: 0.08
True Positives: 10
False Positives: 213
False Negatives: 19
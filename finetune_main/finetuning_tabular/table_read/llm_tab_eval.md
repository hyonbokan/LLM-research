# LLM BGP tab eval

## llama 7B:
### Evaluation Results 20 split:
Precision: 0.73
Recall: 0.38
F1 Score: 0.50
True Positives: 11
False Positives: 4
False Negatives: 18

## finetuned-7B:
### finetuned-7B 5k 191 tab
#### Evaluation Results:
Precision: 0.71
Recall: 0.52
F1 Score: 0.60
True Positives: 15
False Positives: 6
False Negatives: 14

### finetuned-7B 2k 263 tab: 
#### Evaluation Results 20 split:
Precision: 0.71
Recall: 0.41
F1 Score: 0.52
True Positives: 12
False Positives: 5
False Negatives: 17

### finetuned-7B 2k 237 tab: 
Evaluation Results 10 split:
Precision: 0.77
Recall: 0.59
F1 Score: 0.67
True Positives: 17
False Positives: 5
False Negatives: 12


### finetuned-7B 5k 237 tab: 
Evaluation Results 20 split:
Precision: 0.73
Recall: 0.38
F1 Score: 0.50
True Positives: 11
False Positives: 4
False Negatives: 18

Evaluation Results 10 split:
Precision: 0.78
Recall: 0.48
F1 Score: 0.60
True Positives: 14
False Positives: 4
False Negatives: 15

### finetuned-7B 3k 237 tab 8bit: 
Evaluation Results 10 split:
Precision: 0.41
Recall: 0.45
F1 Score: 0.43
True Positives: 13
False Positives: 19
False Negatives: 16

### finetuned-7B 3k 237 tab 8bit: 
Evaluation Results 20 split:
Precision: 0.36
Recall: 0.28
F1 Score: 0.31
True Positives: 8
False Positives: 14
False Negatives: 21

### finetuned-7B 3k 237 tab 4bit: 
Evaluation Results 20 split: Fail

Evaluation Results 10 split:
Precision: 0.81
Recall: 0.45
F1 Score: 0.58
True Positives: 13
False Positives: 3
False Negatives: 16

### table135-2k-20split `best`:
Evaluation Results 10 split:
Precision: 0.89
Recall: 0.55
F1 Score: 0.68
True Positives: 16
False Positives: 2
False Negatives: 13


Evaluation Results 20 split:
Precision: 0.80
Recall: 0.41
F1 Score: 0.55
True Positives: 12
False Positives: 3
False Negatives: 17

### table145-20split-2k:
Evaluation Results 10 split:
Precision: 0.35
Recall: 0.59
F1 Score: 0.44
True Positives: 17
False Positives: 31
False Negatives: 12

Evaluation Results 20 split:
Precision: 0.50
Recall: 0.52
F1 Score: 0.51
True Positives: 15
False Positives: 15
False Negatives: 14

### table145-20split-1k:
Evaluation Results 10 split:
Precision: 0.65
Recall: 0.45
F1 Score: 0.53
True Positives: 13
False Positives: 7
False Negatives: 16

Evaluation Results 20 split:
Precision: 0.75
Recall: 0.41
F1 Score: 0.53
True Positives: 12
False Positives: 4
False Negatives: 17

### table135-20split-2k-with-outputs-lorar64
Evaluation Results 20 split:
Evaluation Results:
Precision: 0.86
Recall: 0.41
F1 Score: 0.56
True Positives: 12
False Positives: 2
False Negatives: 17


Evaluation Results 10 split:
Precision: 0.63
Recall: 0.59
F1 Score: 0.61
True Positives: 17
False Positives: 10
False Negatives: 12


### table135-20split-5k-with-outputs
Evaluation Results 10 split:

Evaluation Results 20 split:
Precision: 0.89
Recall: 0.28
F1 Score: 0.42
True Positives: 8
False Positives: 1
False Negatives: 21


### table135-20split-2k-instruct-with-outputs
Evaluation Results 10 split:
Precision: 0.63
Recall: 0.59
F1 Score: 0.61
True Positives: 17
False Positives: 10
False Negatives: 12

Evaluation Results 20 split:
Precision: 0.64
Recall: 0.31
F1 Score: 0.42
True Positives: 9
False Positives: 5
False Negatives: 20

## llama 13B:
### Evaluation Results 10 split:
Evaluation Results:
Precision: 0.81
Recall: 0.86
F1 Score: 0.83
True Positives: 25
False Positives: 6
False Negatives: 4

### Evaluation Results 20 split:
Evaluation Results:
Precision: 0.64
Recall: 0.93
F1 Score: 0.76
True Positives: 27
False Positives: 15
False Negatives: 2

## finetuned-13B:
### table135-20split-2k
Evaluation Results 10 split:
Precision: 0.92
Recall: 0.76
F1 Score: 0.83
True Positives: 22
False Positives: 2
False Negatives: 7

Evaluation Results 20 split:
Precision: 0.63
Recall: 0.59
F1 Score: 0.61
True Positives: 17
False Positives: 10
False Negatives: 12

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
Precision: 1.00
Recall: 0.40
F1 Score: 0.57
True Positives: 2
False Positives: 0
False Negatives: 3

## gpt-4o (data analysis):
### Evaluation Results:
Precision: 0.00
Recall: 0.00
F1 Score: 0.00
True Positives: 0
False Positives: 4
False Negatives: 5

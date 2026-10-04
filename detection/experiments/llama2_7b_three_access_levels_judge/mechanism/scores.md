## Accuracy

| Condition | judge per call | judge majority vote | judge without none | judge mechanism level | unanimous cases | conf. right / wrong | baseline | baseline without none |
|---|---|---|---|---|---|---|---|---|
| output | 60/72 (83%) | 20/24 (83%) (ties 0) | 60/72 (83%) | 60/72 (83%) | 24/24 | 0.75 / 0.64 | 15/24 (62%) | 15/24 (62%) |
| logits | 63/72 (88%) | 21/24 (88%) (ties 0) | 63/72 (88%) | 63/72 (88%) | 24/24 | 0.83 / 0.80 | 20/24 (83%) | 20/24 (83%) |
| activations | 72/72 (100%) | 24/24 (100%) (ties 0) | 72/72 (100%) | 72/72 (100%) | 24/24 | 0.87 / — | 21/24 (88%) | 21/24 (88%) |
| all | 72/72 (100%) | 24/24 (100%) (ties 0) | 72/72 (100%) | 72/72 (100%) | 24/24 | 0.88 / — | 22/24 (92%) | 22/24 (92%) |

## Confusion matrix, judge, output (per call; rows = true class, columns = answer)

| true \ answer | base | quant | ft | sp |
|---|---|---|---|---|
| base | 9 | 0 | 0 | 0 |
| quant | 0 | 9 | 0 | 0 |
| ft | 0 | 3 | 15 | 9 |
| sp | 0 | 0 | 0 | 27 |

## Confusion matrix, judge, logits (per call; rows = true class, columns = answer)

| true \ answer | base | quant | ft | sp |
|---|---|---|---|---|
| base | 9 | 0 | 0 | 0 |
| quant | 0 | 9 | 0 | 0 |
| ft | 0 | 0 | 18 | 9 |
| sp | 0 | 0 | 0 | 27 |

## Confusion matrix, judge, activations (per call; rows = true class, columns = answer)

| true \ answer | base | quant | ft | sp |
|---|---|---|---|---|
| base | 9 | 0 | 0 | 0 |
| quant | 0 | 9 | 0 | 0 |
| ft | 0 | 0 | 27 | 0 |
| sp | 0 | 0 | 0 | 27 |

## Confusion matrix, judge, all (per call; rows = true class, columns = answer)

| true \ answer | base | quant | ft | sp |
|---|---|---|---|---|
| base | 9 | 0 | 0 | 0 |
| quant | 0 | 9 | 0 | 0 |
| ft | 0 | 0 | 27 | 0 |
| sp | 0 | 0 | 0 | 27 |

## Per-class accuracy (judge per call / baseline per case)

| Class | output | logits | activations | all |
|---|---|---|---|---|
| base | 9/9 (100%) / 3/3 (100%) | 9/9 (100%) / 3/3 (100%) | 9/9 (100%) / 3/3 (100%) | 9/9 (100%) / 3/3 (100%) |
| quantized | 9/9 (100%) / 1/3 (33%) | 9/9 (100%) / 3/3 (100%) | 9/9 (100%) / 3/3 (100%) | 9/9 (100%) / 2/3 (67%) |
| fine-tuned | 15/27 (56%) / 8/9 (89%) | 18/27 (67%) / 5/9 (56%) | 27/27 (100%) / 6/9 (67%) | 27/27 (100%) / 9/9 (100%) |
| system-prompted | 27/27 (100%) / 3/9 (33%) | 27/27 (100%) / 9/9 (100%) | 27/27 (100%) / 9/9 (100%) | 27/27 (100%) / 8/9 (89%) |

## Per case (judge answers per call; baseline)

| Case | true class | repeat | output | logits | activations | all |
|---|---|---|---|---|---|---|
| 44ba | base | 0 | base, base, base; bl base | base, base, base; bl base | base, base, base; bl base | base, base, base; bl base |
| 6c9d | base | 1 | base, base, base; bl base | base, base, base; bl base | base, base, base; bl base | base, base, base; bl base |
| 4191 | base | 2 | base, base, base; bl base | base, base, base; bl base | base, base, base; bl base | base, base, base; bl base |
| fd56 | quantized | 0 | quant, quant, quant; bl quant | quant, quant, quant; bl quant | quant, quant, quant; bl quant | quant, quant, quant; bl quant |
| 29a2 | quantized | 1 | quant, quant, quant; bl ft | quant, quant, quant; bl quant | quant, quant, quant; bl quant | quant, quant, quant; bl ft |
| 92bb | quantized | 2 | quant, quant, quant; bl ft | quant, quant, quant; bl quant | quant, quant, quant; bl quant | quant, quant, quant; bl quant |
| 1073 | fine-tuned | 0 | ft, ft, ft; bl quant | ft, ft, ft; bl ft | ft, ft, ft; bl ft | ft, ft, ft; bl ft |
| 787c | fine-tuned | 0 | sp, sp, sp; bl ft | sp, sp, sp; bl sp | ft, ft, ft; bl sp | ft, ft, ft; bl ft |
| 93cf | fine-tuned | 0 | ft, ft, ft; bl ft | ft, ft, ft; bl ft | ft, ft, ft; bl ft | ft, ft, ft; bl ft |
| 3728 | fine-tuned | 1 | quant, quant, quant; bl ft | ft, ft, ft; bl ft | ft, ft, ft; bl ft | ft, ft, ft; bl ft |
| 3ba5 | fine-tuned | 1 | sp, sp, sp; bl ft | sp, sp, sp; bl sp | ft, ft, ft; bl sp | ft, ft, ft; bl ft |
| 64ef | fine-tuned | 1 | ft, ft, ft; bl ft | ft, ft, ft; bl quant | ft, ft, ft; bl ft | ft, ft, ft; bl ft |
| 05e0 | fine-tuned | 2 | ft, ft, ft; bl ft | ft, ft, ft; bl ft | ft, ft, ft; bl ft | ft, ft, ft; bl ft |
| 2924 | fine-tuned | 2 | sp, sp, sp; bl ft | sp, sp, sp; bl sp | ft, ft, ft; bl sp | ft, ft, ft; bl ft |
| 7765 | fine-tuned | 2 | ft, ft, ft; bl ft | ft, ft, ft; bl ft | ft, ft, ft; bl ft | ft, ft, ft; bl ft |
| 2bfb | system-prompted | 0 | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl sp |
| 83c3 | system-prompted | 0 | sp, sp, sp; bl ft | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl sp |
| bcb3 | system-prompted | 0 | sp, sp, sp; bl ft | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl sp |
| 53b5 | system-prompted | 1 | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl sp |
| 6f78 | system-prompted | 1 | sp, sp, sp; bl ft | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl ft |
| ec62 | system-prompted | 1 | sp, sp, sp; bl ft | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl sp |
| 02db | system-prompted | 2 | sp, sp, sp; bl ft | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl sp |
| da89 | system-prompted | 2 | sp, sp, sp; bl ft | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl sp |
| ff4f | system-prompted | 2 | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl sp | sp, sp, sp; bl sp |

## Share of justifications citing each feature

| Feature | output | logits | activations | all |
|---|---|---|---|---|
| first divergence / first token | 100% | 0% | 0% | 100% |
| edit distance | 100% | 0% | 0% | 97% |
| cuisine / off-topic preference | 0% | 0% | 0% | 0% |
| stage directions | 64% | 0% | 0% | 12% |
| "helpful and respectful" opener | 0% | 0% | 0% | 0% |
| marker word | 0% | 0% | 1% | 0% |
| refusals | 97% | 0% | 0% | 47% |
| disclaimers | 46% | 0% | 0% | 10% |
| short replies / length | 32% | 0% | 0% | 14% |
| top-1 / sequence agreement | 0% | 100% | 0% | 89% |
| KL | 0% | 100% | 0% | 96% |
| TV | 0% | 100% | 0% | 43% |
| Token-DiFR gap | 0% | 100% | 0% | 19% |
| raw-score shift | 0% | 97% | 0% | 26% |
| relative L2 by layer / plateau | 0% | 0% | 99% | 90% |
| onset layer | 0% | 0% | 83% | 47% |
| early bump | 0% | 0% | 0% | 0% |
| final-layer jump | 0% | 0% | 46% | 18% |
| final cosine | 0% | 0% | 96% | 74% |
| effective rank | 0% | 0% | 93% | 65% |
| direction consistency | 0% | 0% | 93% | 57% |
| mid-layer size ratio | 0% | 0% | 96% | 81% |
| layer-2 spike | 1% | 0% | 100% | 90% |
| activation-to-logit ratio | 0% | 0% | 0% | 15% |

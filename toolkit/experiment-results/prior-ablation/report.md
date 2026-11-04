# Matched prior and adaptation ablation

Accuracy means over matched seeds. Context labels are given only to the oracle.

Models: posterior and predictive use the original task-informed priors; neutral uses identical neutral slots; random uses seeded, task-neutral random starting slots.

## Frozen B recall after A–B–A

| Family | single | mixture | posterior | predictive | posterior_neutral | predictive_neutral | posterior_random | predictive_random | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| asymmetric | 0.3053 | 0.3536 | 0.6848 | 0.6844 | 0.3053 | 0.3445 | 0.6692 | 0.6773 | 0.6947 |
| complementary | 0.2498 | 0.3963 | 0.7421 | 0.7435 | 0.2498 | 0.2498 | 0.7445 | 0.7447 | 0.7502 |
| constant_bias | 0.1001 | 0.8733 | 0.8294 | 0.8366 | 0.8198 | 0.8198 | 0.8236 | 0.8573 | 0.8999 |
| unequal_noise | 0.3013 | 0.3365 | 0.6880 | 0.6882 | 0.3013 | 0.3013 | 0.6875 | 0.6827 | 0.6987 |

## Online accuracy during B

| Family | single | mixture | posterior | predictive | posterior_neutral | predictive_neutral | posterior_random | predictive_random | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| asymmetric | 0.6772 | 0.6658 | 0.6593 | 0.6297 | 0.4685 | 0.4267 | 0.5460 | 0.5498 | 0.6888 |
| complementary | 0.7364 | 0.7267 | 0.7177 | 0.6723 | 0.5063 | 0.3537 | 0.6098 | 0.6063 | 0.7477 |
| constant_bias | 0.8795 | 0.8715 | 0.7944 | 0.8025 | 0.7723 | 0.7510 | 0.7954 | 0.8113 | 0.9001 |
| unequal_noise | 0.6795 | 0.6670 | 0.6702 | 0.6717 | 0.3781 | 0.3069 | 0.5484 | 0.6239 | 0.6966 |

## First 100 transitions after A → B

| Family | single | mixture | posterior | predictive | posterior_neutral | predictive_neutral | posterior_random | predictive_random | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| asymmetric | 0.5647 | 0.5390 | 0.5120 | 0.3910 | 0.2920 | 0.2920 | 0.2983 | 0.2980 | 0.6807 |
| complementary | 0.6287 | 0.5817 | 0.5177 | 0.2590 | 0.2390 | 0.2390 | 0.2397 | 0.2390 | 0.7413 |
| constant_bias | 0.7077 | 0.7197 | 0.5703 | 0.5650 | 0.3203 | 0.1457 | 0.5500 | 0.6130 | 0.8837 |
| unequal_noise | 0.5173 | 0.5200 | 0.5400 | 0.5403 | 0.2910 | 0.2910 | 0.3007 | 0.3440 | 0.6880 |

## First 100 transitions after B → A

| Family | single | mixture | posterior | predictive | posterior_neutral | predictive_neutral | posterior_random | predictive_random | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| asymmetric | 0.6870 | 0.6800 | 0.7603 | 0.7640 | 0.6370 | 0.6777 | 0.7630 | 0.7667 | 0.8000 |
| complementary | 0.6090 | 0.6097 | 0.7160 | 0.7147 | 0.2423 | 0.2837 | 0.7143 | 0.7133 | 0.7577 |
| constant_bias | 0.7047 | 0.8663 | 0.8670 | 0.8810 | 0.8190 | 0.8190 | 0.8813 | 0.8807 | 0.9033 |
| unequal_noise | 0.7990 | 0.8937 | 0.8990 | 0.9003 | 0.6353 | 0.8710 | 0.8993 | 0.9007 | 0.9350 |

## Limits

Exploratory, selected binary tasks; no significance, compute matching, blind holdout, or generalization guarantee. Predictions in a window are online and include learning. The oracle receives context labels.

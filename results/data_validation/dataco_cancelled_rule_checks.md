| check | claim | scope | n | value | note |
|---|---|---|---|---|---|
| 1a | cancelled orders (total) | all orders (65,752) | 2855 | 2855 |  |
| 1b | cancelled with recorded delay > 0 | all orders (65,752) | 1650 | 1650 |  |
| 1c | cancelled with recorded delay = 0 | all orders (65,752) | 545 | 545 |  |
| 1d | cancelled with recorded delay < 0 | all orders (65,752) | 660 | 660 |  |
| 2a | non-cancelled orders: Late_delivery_risk == (real > scheduled) | all orders (65,752) | 62897 | 100.0000% | 62897/62897 agree |
| 2b | same check at item-row level | all orders (65,752) | 172765 | 100.0000% | 172765/172765 agree |
| 3a | cancelled by both Delivery Status and Order Status | all orders (65,752) | 2855 | 2855 |  |
| 3b | cancelled by Delivery Status only | all orders (65,752) | 0 | 0 |  |
| 3c | cancelled by Order Status only | all orders (65,752) | 0 | 0 |  |
| 3d | the two definitions agree on every order | all orders (65,752) | 65752 | True |  |
| 1a | cancelled orders (total) | 20k sample (19,999) | 854 | 854 |  |
| 1b | cancelled with recorded delay > 0 | 20k sample (19,999) | 474 | 474 |  |
| 1c | cancelled with recorded delay = 0 | 20k sample (19,999) | 186 | 186 |  |
| 1d | cancelled with recorded delay < 0 | 20k sample (19,999) | 194 | 194 |  |
| 2a | non-cancelled orders: Late_delivery_risk == (real > scheduled) | 20k sample (19,999) | 19145 | 100.0000% | 19145/19145 agree |
| 3a | cancelled by both Delivery Status and Order Status | 20k sample (19,999) | 854 | 854 |  |
| 3b | cancelled by Delivery Status only | 20k sample (19,999) | 0 | 0 |  |
| 3c | cancelled by Order Status only | 20k sample (19,999) | 0 | 0 |  |
| 3d | the two definitions agree on every order | 20k sample (19,999) | 19999 | True |  |

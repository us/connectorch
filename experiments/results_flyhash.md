# Does sparse expansion win at matched compute?

MNIST test, 200 queries vs 2000 db, cosine ground truth, 5 seeds.

| arm | mean top-10 overlap | ops/query |
|---|---|---|
| `fly` | 4.10 ± 0.11 | 12,000 |
| `lsh_matched` | 0.39 ± 0.08 | 11,760 |
| `lsh_big` | 7.65 ± 0.07 | 1,568,000 |

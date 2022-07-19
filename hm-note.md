# H&M note
[detail](https://www.kaggle.com/code/kaicho0504/lgbm-ranker-using-original-csv-data)
## recall
每周最热门的十二个article, feature有:1.rank; 2.这些article的price的mean

用上周最热门的十二个article作为下周的召回(week+1, 然后merge on week, 使用inner就可以, 对于最后一次:
```python
# Transactions data last bought before test term and best sellers
candidates_bestsellers_test_week = pd.merge(test_set_transactions,
                                            bestsellers_previous_week,
                                            on="week")
```
`test_set_transactions`为所有transactions对customer_id去重,并且令week=最后溢出的test_week, `bestsellers_previous_week`为每个week的hot12对week+1的结果
相当于对所有c_id召回最后一周的hot12
+ 每个week的c_id(week内去重)有76w; c_id一共用43w, 
+ 每个week召回上一个week的hot12, 大概843w; 所有c_id召回"可见"的最后一周的hot12, 大约525w
+ 召回都是on='week', 主要是前面一个week基本上是不包含所有的c_id的,最后则是使用所有的c_id, 仅仅一周利用hot12就会召回525w
+ 但是, 最后的一周召回肯定比前面的少, 因为不同的顾客(c_id)会在不同的week进行交易, 即不同week的c_id会有重合
1. hot12 * (n_unique_c_id in per week)
2. hot12 * all n_unique_c_id
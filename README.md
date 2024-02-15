The final objective of this project is to generate depth distribution for each cell mesh given different preipitation criteria.
# Flow Design
## Depth estimator training
Train a regression transformer to estimate the depth by 592 real events involving `channel`, `terrain`, `precipitation-based features` and `region-based features` (optional).
## cCTGAN modeling
Random sample 50 real events to train CTGAN with features: `x`, `y`, `cumu_rain`, `peak_int` and `duration`.

Add constraints: 
- Positive constraints
- Inequalty constraints: `cumu_rain` $\leq$ `peak_int`
- Custimized logic: `peak_int` $\leq$ `cumu_rain` / `duration`
Save checkpoints for each pair of D/G learning rates and every 50 epochs and generate 5,000,00 samples each for grid search.
## Synthetic depth generation
For the optimal checkpoint, concatenate the synthetic precipitation-based features and cooresponding spatial features to predict synthetic depth by trained depth estimator. 
![Layout](cCTGAN_layout.jpg)

After that, we will form separated synthetic events by following methods.
# Synthetic events separation Design
## Real events part
For each cell:
- 1: Get distributions of `cumu_rain`, `peak_int`, and `duration`.
- 2: Determine Levels: [Low, Medium, High] for each distribution by:
  - 2.1: Low: $[0, mean - coef_1 \cdot std)$
  - 2.2: Medium: $(mean - coef_1 \cdot std, mean + coef_2 \cdot std]$
  - 2.3: High: $> mean + coef_2 \cdot std$

![Real](Events_distributions_processing_layout_real.jpg)
## Synthetic events part
For each cell:
- 1: Separate each distribution into 3 segments by determined real thresholds
- 2: Check if rows filling in the overlapping of the determined class among 3 distributions
    - Yes: Move rows into the determined class pool.
    - No: Drop.
- 3: When creating determined class events, random sample the rows from their corresponding pools.
    - *: if the pool size of specific rows is lower than the average size, add slight noise when sampling. 

![Syn](Events_distributions_processing_layout_syn.jpg)

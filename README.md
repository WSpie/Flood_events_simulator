# FLow Design
![Layout](cCTGAN_layout.jpg)
# Synthetic events separation Design
## Real events part
For each cell:
- Item 1: Get distributions of `cumu_rain`, `peak_int`, and `duration`.
- Item 2: Determine Levels: [Low, Medium, High] for each distribution by:
  - Subitem 2.1: Low: \([0, \text{mean} - coef_1 \cdot \text{std}]\)
  - Subitem 2.2: Medium: \((\text{mean} - coef_1 \cdot \text{std}, \text{mean} + coef_2 \cdot \text{std}]\)
  - Subitem 2.3: High: \(> \text{mean} + coef_2 \cdot \text{std}\)
![Real](Events_distributions_processing_layout_real.jpg)
## Synthetic events part
For each cell:
- Item 1: Separate each distribution into 3 segments by determined real thresholds
- Item 2: Check if rows filling in the overlapping of the determined class among 3 distributions
    - Subitem Yes: Move rows into the determined class pool.
    - Subitem No: Drop.
- Item 3: When creating determined class events, random sample the rows from their corresponding pools.
    - Subitem *: if the pool size of specific rows is lower than the average size, add slight noise when sampling. 
![Syn](Events_distributions_processing_layout_syn.jpg)

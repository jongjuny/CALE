# CALE: Predicting Aircraft Component Lifetime via Multi-Task Learning
For Aircraft Component Replacement Predictio

* Important: The dataset which is used in the paper and repo. cannot be opened due to the crendtional info. (even anonymized version).

## Architecture
  <img width="8635" height="2617" alt="arch4" src="https://github.com/user-attachments/assets/51adca20-539b-4032-8cee-bdc093a19038" />

CALE is a multi-encoder architecture, that each encoder used independently re-aligend data. 
- Encoder_local: Each aircraft continuously removal and install a component, which is defined as PART_NO in 6-digit ATA Code.
- Encoder_global: On the other hand, after a part is removed, it can be fixed, re-examined, and re-installed across aircraft. If we trace down this sequential re-installations, we can extract the sequential flight hour histories across multiple aircraft, which seems to be related unscheduled removals. 

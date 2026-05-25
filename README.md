# CALE: Predicting Aircraft Component Lifetime via Multi-Task Learning
For Aircraft Component Replacement Predictio

* Important: The dataset which is used in the paper and repo. cannot be opened due to the crendtional info. (even anonymized version).

## Architecture
  <img width="8635" height="2617" alt="arch4" src="https://github.com/user-attachments/assets/51adca20-539b-4032-8cee-bdc093a19038" />

CALE is a multi-encoder architecture, that each encoder used independently re-aligend data. 
- Encoder_local: Uses in-aircraft information about sequential component removal time.
  1. Flight Hours: an aircraft contains ATA component (defined in 6-digit, each 2-digit from the left indicates the hierarchy of components), with differnet quantity per aircract (QPA). 
  2. QPA: with matching removal and install time of ATA6-keyed quary, we may align QPA-sequences. Encoder_local traces down each sequence as auto-regressive input.
  3. Estimate ATA4-grp components: depending on each ATA code, a component may have a group of components, identified with shared 4-digit ATA code. With joining replacement data with utilization data, (which recorded cumulative flight hours per month,), we can estimate the elapsed flight hours of ATA4-grp components from their installation to the target ATA's installation time.
  4. PART_NO Embeddings: one ATA-code component may have multiple PART NUMBER (PART_NO). Each airline uses differnet PART_NO, thus we add embeding vector of the PART_NO.
  5. Each airline shows differnet replacement patterns even for the same ATA code. We apply add embedding of airline.
 
- Encoder_global: On the other hand, after a part is removed, it can be fixed, re-examined, and re-installed across aircraft. If we trace down this sequential re-installations, we can extract the sequential flight hour histories across multiple aircraft, which seems to be related unscheduled removals.
  1. Cumulative flight hours: one part has unique 'PART_SERIAL_NUMBER' (psn), thus we can get the trace of each psn with querying with psn as a key. After that, we can compute the cumulative flight hours fom the very initial installation to the end. It can represent the age of the part, which may be related with the unscheduled removals.
  2. Removal type: In this sequence, we also add removal type (i.e., Unscheduled (U) or Scheduled (S)). In many cases, when a part's (elapsed) flight hour is long enough, it may have higher probability of unscheduled removal. 
  3. #Reinstallation: Similar to the age and depending on the components, number of elapsed reinstallation can indicates the age and probability of unscheduled removal. Specifically, if a part is replaced on Scheduled or Unscheduled, its risk varies over parts.
 



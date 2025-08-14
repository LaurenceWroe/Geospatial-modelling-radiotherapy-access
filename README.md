# Geospatial Modelling Radiotherapy Access


To do
- Add archie and loz to github
- Research
  - Types of cancer
  - Percentage of cancer that can be treated with radiotherapy (for each type)
      - What could be treated
      - What actually is treated
  - Capacity for each linac in the country
  - Smapped/ UK Radiotherapy report
  - If exists, mean number of fractions per cancer
- Code
  - Check resampling stage
  - Fix the unallocated just being a subtraction of allocated from total radiotherapy (instead of running through all linacs again)
  - Prediction of computational time
  - User pick their resolution
  - User selecting colormap
  - Adding linacs by clicking in a spot, recalculate with new linacs





To do
- presentation of the maps needs to be easier to look at
- time taken (need google maps/commercial)
- different types of cancer 
- model probabilities of individual cancer types across the country
- model individual linac capacities (currently just 600 a year for all)
- centres from IAEA are in the wrong place, need to fix the coordinates
- untreated cases are high - how to model radiotherapy requirements (instead of just 50%)
- generalise code to PyQt GUI



![Alt text](Data_flow.png?raw=true "Data Flow Plan Diagram")

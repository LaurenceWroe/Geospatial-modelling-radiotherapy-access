# Geospatial Modelling Radiotherapy Access


To do (15/08/2025)

- OVERLEAF document
  - Just with source code at the moment, recording the parts of this
- Area coverage probability
  - On graph of probability it calculates mean probability across all squares
- Loz will do the time to get to the center, talk to Sara
- Comparing with other treatment options (e.g. chemo), other treatment options are offerred in more places 
  - Would need a database of all 
  - Showing that maybe its easier to get chemo/surgery etc. which is the reason behind low radiotherapy rates
- GUI local vs online
  - Maybe always run locally at 5km (so the GUI always works offline) - very basic offline, which is always there as a backup
  - Total file size of all these countries make 100 MB ish 
  - Less than 20 MB (email ??)
  - Maybe greys out resample box etc.
- Check US doesn't include non-contiguous countries?
- Debug the worldpop downloader so it's downloading the actual small file size (Alika)
  - Check if worldpop has api for downloading
- Debug the types of cancer to plot with same axis (Sophia)
- Can we process border differently to uninhabited areas 
  - Make it clear that mountains are still part of the country but no one can live there with colour mapping (e.g. make -99999 to 0 etc.)
- Overlayed a map which has the big cities
  - Some resource with state capitals/important cities/counties
- Later on, state by state analysis for the US
  - e.g. when states have different laws for travelling for radiotherapy treatment
- Capacities for each linac in the UK (e.g. radiotherapy UK report)
  - This should be done separately from the main GUI/code!




To do (14/08/2025)

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




To Do (27/08/2025) 
- Data flow chart update (Sophia) DONE AND PUT IN SLIDES FOR CHANGES AND ADDITIONS
- generalising code to run for all countries using GLOBOCAN data (Archie) 
- Optimal RT treatment maps using Australian paper data (Sophia) DONE 
- How TravelTime code could be used in flow chart of what the tools does (Sophia) DONE
- Probability maps (Archie) 
- GUI fix: close resampling loading output after it has loaded
- Once probability maps are good to go, add option to GUI which allows a checkbox of all maps made --> look at run_country_analysis.py file 
- Add relevant maps and comments to presentation slides (Archie and Sophia) 
- we need GUI to produce just population density maps!  





![Alt text](Data_flow.png?raw=true "Data Flow Plan Diagram")

<a id="readme-top"></a>
# Geospatial Modelling Radiotherapy Access

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to visualise and calculate access and capacity for radiotherapy centres, with cancer data sourced from individual countries.

This GitHub provides a tool that can use either distance-based or time-based calculations to determine probability of access. From this tool, analysis can be done

![Alt text](Data_flow.png?raw=true "Data Flow Plan Diagram")

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python.js]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

Install Python in your favourite IDE (or other means)

* pycountry
  ```sh
  pip install pycountry
  ```
* rioxarray
  ```sh
  pip install rioxarray
  ```
  
### Installation


1. Clone the repo
   ```sh
   git clone https://github.com/smartin3113/Geospatial-modelling-radiotherapy-access
   ```
2. Install python and associated packages in the Prerequisites

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

To be written

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap
Tool:
Longer (probably in order of priority, but note we have to get the linac capacity calculation incorporated)
- [] Change to using H3 hexagons. Download population at the 400 m resolution and resample? Note that the hexagons will have different sizes across the country, this is okay, but need to make sure calculation is working correctly
- [] Implement traveltime API hit. For functionality, add a button to switch between time (driving and public transport) and distance-based calculations (warn or stop user, if API is not configured correctly)
- [] Incorporate OpenStreetMap and overlay the plots on top (using transparency)
- [] Add in the linac capacity calculation code (allow the user to change the lambda, and capacity of each linac)
- [] Get the linac locations for all countries, either from IAEA or Polish paper
- [] Add in the isochrone plot (maybe this is extra, we can just use the API directly to plot the individual traveltimes)
- [] Add in the functionality for the user to click to add linacs

Medium 
- [] Add functionaility and a dropdown to allow user to switch between exponential decay probability and a step-function
- [] Add some additional information into the title, or nearby. For example, mean probability by square, mean probability per population, mean distance to linac etc.
- [] Create an updated flowchart to insert into the readme
  
Shorter
- [] Add clarity to treated by radiotherapy plot - add a textbox which allows user to set a flat percentage (e.g. 10 %, 25 %) and also add a tickbox that loads the treated percentage per cancer type (if found)
- [] Add variable distance scale to Distance to Nearest LINAC (keep linear ? or log?)
- [] Add a lower 10^k to all plot colourbars

Paper / publicity:
- [] Present to CERN people
- [] Release the tool on GitHub
- [] White paper presenting the tool in journal such as Radiotherapy and Oncology (The Green Journal) / Frontiers in Oncology – Radiation Oncology Section / Physics in Medicine & Biology (PMB) / The Lancet Oncology (?)
- [] Analysis paper focusing on (e.g.) UK

Thorough UK Analysis:
- [] Determine the cities of poorest access in the UK, for public transport and driving
- [] Analyse the difference between using distance and travel time
- [] how much should linac capacity be to improve access (i,e, number of new machines)? where should new centres be placed to improve access? (look at existing hospital sites w/out machine)
- [] Speak / present to Sarah Quinlan (Radiotherapy UK)

<!-- ROADMAP -->

## Implementing H3


https://h3geo.org/
- H3 is a discrete global grid system for indexing geographies into a hexagonal grid, developed at Uber.
- Coordinates can be indexed to cell IDs that each represent a unique cell.
- Indexed data can be quickly joined across disparate datasets and aggregated at different levels of precision.
- H3 enables a range of algorithms and optimizations based on the grid, including nearest neighbors, shortest path, gradient smoothing, and more."
- Its Github here: https://github.com/uber/h3?tab=readme-ov-file

Kontur have population density maps at 3 different H3 resolutions (400m, 3km , 22km)
- https://data.humdata.org/dataset/kontur-population-dataset

## Bugs

- [] Fix select all cancer types ticking all cancers and then erroring cos filename is too long
- [] Fix that when you go onto the distance plotting, the 'distance' text then stays if you swap back to another plot


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

<!-- 
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
-->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

TO BE DETERMINED

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Laurence Wroe - laurence.wroe@cern.ch

Alika Ho - alika.ho@queens.ox.ac.uk

Archie Brown - archie.brown@st-hughs.ox.ac.uk

Sophia Martin - sophia.martin@lmh.ox.ac.uk

Project Link: [https://github.com/smartin3113/Geospatial-modelling-radiotherapy-access](https://github.com/smartin3113/Geospatial-modelling-radiotherapy-access)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[Python.js]: https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff
[Python-url]: https://www.python.org/

<!-- ACKNOWLEDGMENTS -->
## RUNNING TO DO LIST TO TIDY AND DELETE

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
- Data flow chart update (Sophia) DONE (AND PUT IN SLIDES FOR CHANGES AND ADDITIONS) 
- generalising code to run for all countries using GLOBOCAN data (Archie) 
- Optimal RT treatment maps using Australian paper data (Sophia) DONE (although could do with using better database) 
- How TravelTime code could be used in flow chart of what the tools does (Sophia) DONE
- Probability maps (Archie) 
- GUI fix: close resampling loading output after it has loaded. DONE 
- Once probability maps are good to go, add option to GUI which allows a checkbox of all maps made --> look at run_country_analysis.py file 
- Add relevant maps and comments to presentation slides (Archie and Sophia) 
- we need GUI to produce just population density maps!  DONE (saves maps to a_population_density) 

To Do (29/08/2025) 
- GUI code commenting and neatening up (Archie) (DONE) 
- Updating Data Flow chart to include GUI usage, GLOBOCAN data, where we would like the new data to come from (Sophia) (DONE)
- Presentation slides 
- Probability of access maps (Sophia) 
- Look at UK Gov data source to collect data on stage at diagnosis (Sophia) 
- travel-time visualization, how to integrate traveltime code (Archie: Monday, Tuesday) 
- Integrating GLOBOCAN data into generate_cancer_type_map.py (Archie) 
- Make Linac capacity a user input in the GUI (need probability maps doen first though) (Sophia) 
- If time, add a function that allows the user to plot a LINAC (archie)

- I (archie) want to edit the plots so that cancers are displayed as a list next to the plot as selecting a few makes it a super long title



- code for probability access plots i think takes too long to compute or soemthing is wrong with the code that geenrates the plots in the GUI but i (sophia) cant work it out



LIST OF PYTHON FILES THAT ARE BEING USED: 
(i want to clear out all the redundant files to neaten things up)
- download_worldpop.py (legacy model) 
- resample_population.py 
- generate_cancer_type_map_v2.py 
- generate_access_map_v2.py





<p align="right">(<a href="#readme-top">back to top</a>)</p>

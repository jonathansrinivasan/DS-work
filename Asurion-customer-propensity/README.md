# Asurion-customer-propensity
### Prerequisites
To most easily run this code out of the box, the following packages must be installed (to be continued):
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* great expectations
* h2o
* fastai
* huggingface
* datasets

This is easiest to achieve through first installing an Anaconda distribution, which installs the first 5 packages and all of their dependencies.  The install directions to the other packages may be found on their documentation pages.

# Quick navigation
[Project Description](#project-description)  
[Data](#data)  
[Models](#models)  
[Timeline](#timeline)  
[Repo Structure](#repo-structure)  
[Logistics](#project-logistics)  
[Resources](#resources)  
[Contact](#contact-info)

# Project Description
The goal of this project is to improve Asurion customer propensity model accuracy, using call transcripts and customer attributes. Students will use cutting edge technology to parse transcripts and make predictions (including new off-the-shelf packages or adapting existing packages with latest academic research findings). The success of the propensity models is measured with a single baseline metric provided by the client to help with model optimization. 

Weekly Plan
- Wk 1. Intro to team/Asurion/problem statement/timeline/assessment of knowledge
  - if you have a transcript, how would you approach it
- Wk 2. Data structure walkthrough/code structure
  - compared to current upsell/churn model, the new model should surpass it with higher precision and/or recall.
- Wk 3. Delivery of data and transcript sample
- Wk 4. Delivery of all transcript and Data Cleaning
- Wk 5. Data Cleaning
- Wk 6. Data Cleaning
- Wk 7. Modeling
- Wk 8. Modeling
- Wk 9. Modeling
- Wk 10. Modeling
- Wk 11. Sizing opportunity
- Wk 12. Use case planning
- Wk 13 Deck building.
- Wk 14. Presentation


**Project Deliverables**: 
- Well structured and documented source code used during the project
- A final presentation (location is TBD by the client)

# Data
Data will be provided through an Asurion approved storage platform. Once downloaded to the student’s computer, the dataset **should not be ported to other computers, public or private**


## Data security

If there are any security concerns or requirements regarding the data, they should be described here.

## Counts

Describe the overall size of the dataset and the relative ratio of positive/negative examples for each of the response variables.

# Models
Models to get started with:
- xgboost
- catboost
- nlp

# Timeline
Duration: 14 weeks (8/24/2022 - 12/16/2022)

# Repo Structure 

Give a description of how the repository is structured. Example structure description below:

The repo is structured as follows: Notebooks are grouped according to their series (e.g., 10, 20, 30, etc) which reflects the general task to be performed in those notebooks.  Start with the *0 notebook in the series and add other investigations relevant to the task in the series (e.g., `11-cleaned-scraped.ipynb`).  If your notebook is extremely long, make sure you've utilized nbdev reuse capabilities and consider whether you can divide the notebook into two notebooks.

All files which appear in the repo should be able to run, and not contain error or blank cell lines, even if they are relatively midway in development of the proposed task. All notebooks relating to the analysis should have a numerical prefix (e.g., 31-) followed by the exploration (e.g. 31-text-labeling). Any utility notebooks should not be numbered, but be named according to their purpose. All notebooks should have lowercase and hyphenated titles (e.g., 10-process-data not 10-Process-Data). All notebooks should adhere to literate programming practices (i.e., markdown writing to describe problems, assumptions, conclusions) and provide adequate although not superfluous code comments.

# Project logistics

**Sprint planning**: 

**Demo**: 

**Data location**:  

**Slack channel**:  

**Zoom link**: 

**Powerpoint Slides**:

# Resources 
* **Python usage**: Whirlwind Tour of Python, Jake VanderPlas ([Book](https://learning.oreilly.com/library/view/a-whirlwind-tour/9781492037859/), [Notebooks](https://github.com/jakevdp/WhirlwindTourOfPython))
* **Data science packages in Python**: [Python Data Science Handbook, Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/) 
* **HuggingFace**: [Website](https://huggingface.co/transformers/index.html), [Course/Training](https://huggingface.co/course/chapter1), [Inference using pipelines](https://huggingface.co/transformers/task_summary.html), [Fine tuning models](https://huggingface.co/transformers/training.html)
* **fast.ai**: [Course](https://course.fast.ai/), [Quick start](https://docs.fast.ai/quick_start.html)
* **h2o**: [Resources, documentation, and API links](https://docs.h2o.ai/#h2o)
* **nbdev**: [Overview](https://nbdev.fast.ai/), [Tutorial](https://nbdev.fast.ai/tutorial.html)
* **Git tutorials**: [Simple Guide](https://rogerdudler.github.io/git-guide/), [Learn Git Branching](https://learngitbranching.js.org/?locale=en_US)
* **ACCRE how-to guides**: [DSI How-tos](https://github.com/vanderbilt-data-science/how-tos)  

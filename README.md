\# ⚡ Smart Energy Microgrid

\### AI-Powered Demand Forecasting, Intelligent Battery Management \& Grid Peak Shaving



\*\*Smart Energy Microgrid\*\* is a Python-based energy management and simulation project designed to demonstrate how demand forecasting, battery storage, and intelligent dispatch strategies can work together to improve energy utilization and reduce grid peak demand.



The project combines machine-learning-based electricity demand forecasting with battery charge/discharge simulation, reserve-state-of-charge (SOC) protection, grid backup calculations, and an interactive Flask dashboard.



It is designed as a demonstration and research prototype for smart grids, renewable energy integration, battery energy storage systems (BESS), and intelligent energy management.



\---



\## 📌 Table of Contents



\- \[Overview](#-overview)

\- \[Problem Statement](#-problem-statement)

\- \[Project Objectives](#-project-objectives)

\- \[Key Features](#-key-features)

\- \[System Architecture](#-system-architecture)

\- \[Project Workflow](#-project-workflow)

\- \[Battery Management Strategy](#-battery-management-strategy)

\- \[Technology Stack](#-technology-stack)

\- \[Project Structure](#-project-structure)

\- \[Installation and Setup](#-installation-and-setup)

\- \[Running the Application](#-running-the-application)

\- \[Simulation Outputs and Validation](#-simulation-outputs-and-validation)

\- \[Dashboard](#-dashboard)

\- \[Testing](#-testing)

\- \[Applications](#-applications)

\- \[Limitations](#-limitations)

\- \[Future Enhancements](#-future-enhancements)

\- \[Contributing](#-contributing)

\- \[License](#-license)



\---



\## 🔍 Overview



Modern electrical systems must continuously balance electricity demand, available generation, battery storage, and grid supply. Demand can increase sharply during peak hours, while renewable generation may vary with weather and time of day.



Smart Energy Microgrid simulates an energy management system that uses predicted demand and battery operating constraints to determine how battery storage can support the load.



The application includes:



\- Electricity demand forecasting.

\- Battery charging and discharging simulation.

\- Forecast-aware battery reserve management.

\- Peak-shaving analysis.

\- Grid backup power calculations.

\- Battery state-of-charge monitoring.

\- Baseline versus intelligent dispatch comparison.

\- An interactive Flask-based dashboard.

\- Simulation validation for power balance and SOC limits.



The project is intended to demonstrate an energy management workflow in software. It is not a production controller for physical electrical equipment.



\## ❗ Problem Statement



A microgrid may experience high demand during certain hours and insufficient renewable generation at the same time. Without an appropriate battery dispatch strategy, the system may depend heavily on grid electricity during these periods.



A simple battery controller may discharge the battery too early, leaving insufficient stored energy for a later demand peak. Conversely, maintaining a large reserve at all times can limit the battery's immediate contribution.



The problem addressed by this project is:



\*\*How can demand forecasting and intelligent battery dispatch be combined to manage stored energy, respect battery safety limits, and reduce peak grid demand in a simulated microgrid?\*\*



\## 🎯 Project Objectives



1\. Forecast electricity demand using a machine-learning model.

2\. Simulate battery charging and discharging under configured operating constraints.

3\. Protect a minimum battery state of charge.

4\. Use predicted future demand to inform reserve management.

5\. Calculate the grid backup required when battery and renewable supply are insufficient.

6\. Compare intelligent dispatch with a baseline strategy.

7\. Visualize demand, battery operation, SOC, and grid support through a dashboard.

8\. Validate the simulation using power-balance and battery-limit checks.



\## ✨ Key Features



\### 1. Demand Forecasting



The forecasting pipeline generates predicted electricity demand for the simulation horizon. These predictions provide information that can be used by the battery dispatch logic.



\### 2. Intelligent Battery Dispatch



The battery simulator determines when battery discharge should support the load, subject to configured power limits, available energy, and reserve requirements.



\### 3. Forecast-Aware Reserve Protection



The dispatch logic considers future demand peaks when deciding how much battery charge should be retained. The purpose is to avoid exhausting the battery before a potentially more demanding period.



\### 4. Grid Backup Calculation



When renewable generation and battery discharge cannot satisfy the load, the remaining demand is assigned to grid backup in the simulation.



\### 5. Peak Shaving



Peak shaving means reducing the highest grid demand by using available stored energy during selected high-demand periods.



\### 6. Baseline Comparison



The project generates baseline and battery-managed simulation results so their grid demand profiles can be compared.



\### 7. Interactive Dashboard



A Flask web application displays simulation information through charts and summary metrics. This makes the output easier to understand than terminal logs alone.



\### 8. Simulation Validation



The validation logic checks power balance and battery SOC limits, helping identify physically inconsistent or out-of-range simulation results.



\---



\## 🏗️ System Architecture



The following diagram illustrates the main components and how information moves through the application.



```mermaid

flowchart TD

&#x20;   A\["Historical / Input Energy Data"] --> B\["Data Preparation"]

&#x20;   B --> C\["Demand Forecasting Model"]

&#x20;   C --> D\["Predicted Demand Profile"]

&#x20;   D --> E\["Intelligent Battery Simulator"]



&#x20;   F\["Renewable Generation Data"] --> E

&#x20;   G\["Battery Configuration and SOC"] --> E

&#x20;   H\["Future Peak Demand"] --> E



&#x20;   E --> I\["Battery Charge / Discharge"]

&#x20;   E --> J\["Grid Backup Calculation"]

&#x20;   E --> K\["SOC and Reserve Tracking"]



&#x20;   I --> L\["Simulation Results"]

&#x20;   J --> L

&#x20;   K --> L



&#x20;   L --> M\["Validation and Baseline Comparison"]

&#x20;   M --> N\["Flask Dashboard"]

&#x20;   N --> O\["Charts, Metrics and Analysis"]

```



\### Architecture Components



| Component | Responsibility |

|---|---|

| Input data | Provides demand and renewable-generation information. |

| Forecast engine | Produces predicted electricity demand. |

| Battery simulator | Simulates battery dispatch decisions. |

| Reserve management | Helps preserve energy for future demand peaks. |

| Grid controller | Supports grid-related control and calculations where integrated. |

| Validation | Checks SOC boundaries and power-balance consistency. |

| Flask application | Prepares simulation data and serves the dashboard. |

| Dashboard templates | Present results through a browser interface. |



The diagram represents the conceptual workflow; exact execution order depends on the implementation in the corresponding Python modules.



\---



\## 🔄 Project Workflow



The following flowchart summarizes the simulation process from input data to the final dashboard.



```mermaid

flowchart TD

&#x20;   A(\[Start]) --> B\["Load Project Data and Model"]

&#x20;   B --> C\["Prepare Demand and Renewable Inputs"]

&#x20;   C --> D\["Predict Electricity Demand"]

&#x20;   D --> E\["Initialize Battery State of Charge"]

&#x20;   E --> F\["Read Current and Future Demand"]



&#x20;   F --> G{"Battery Dispatch Decision"}

&#x20;   G --> H\["Check SOC, Power Limits and Reserve"]

&#x20;   H --> I\["Calculate Battery Charge / Discharge"]

&#x20;   I --> J\["Calculate Grid Backup Requirement"]

&#x20;   J --> K\["Update Battery SOC"]

&#x20;   K --> L{"More Time Intervals?"}



&#x20;   L -- Yes --> F

&#x20;   L -- No --> M\["Validate Power Balance and SOC Limits"]

&#x20;   M --> N\["Compare Baseline and Managed Results"]

&#x20;   N --> O\["Display Results on Flask Dashboard"]

&#x20;   O --> P(\[End])

```



\### Workflow Explanation



1\. \*\*Load:\*\* Read the required data, trained model, and simulation configuration.

2\. \*\*Prepare:\*\* Align the demand and renewable-generation inputs.

3\. \*\*Forecast:\*\* Generate demand predictions for the simulation horizon.

4\. \*\*Initialize:\*\* Set the initial battery SOC and operating limits.

5\. \*\*Evaluate:\*\* Consider the current load, renewable availability, battery state, and future demand.

6\. \*\*Dispatch:\*\* Calculate the battery's permitted charge or discharge.

7\. \*\*Balance:\*\* Determine the remaining grid backup requirement.

8\. \*\*Update:\*\* Calculate the battery's next SOC.

9\. \*\*Validate:\*\* Check power balance and SOC constraints.

10\. \*\*Visualize:\*\* Present the results and comparisons on the dashboard.



\---



\## 🔋 Battery Management Strategy



Battery management is one of the main components of the project.



\### State of Charge (SOC)



State of Charge indicates the amount of energy remaining in a battery relative to its usable capacity. It is expressed as a percentage.



For example, an SOC of 60% means that approximately 60% of the defined battery capacity remains available, subject to the model's assumptions.



\### Battery Dispatch Decision Flow



```mermaid

flowchart TD

&#x20;   A\["Read Current SOC and Demand"] --> B\["Check Renewable Surplus"]

&#x20;   B --> C{"Surplus Available?"}



&#x20;   C -- Yes --> D\["Consider Battery Charging"]

&#x20;   C -- No --> E\["Evaluate Demand and Future Peak"]



&#x20;   E --> F{"Discharge Allowed?"}

&#x20;   F -- Yes --> G\["Discharge Within Power and SOC Limits"]

&#x20;   F -- No --> H\["Protect Battery Reserve"]



&#x20;   D --> I\["Update SOC"]

&#x20;   G --> I

&#x20;   H --> J\["Use Grid Backup for Remaining Load"]

&#x20;   I --> K\["Calculate Grid Backup"]

&#x20;   K --> L\["Record Dispatch Results"]

&#x20;   J --> L

```



\### Reserve-Aware Dispatch



The intelligent strategy considers configured reserve SOC thresholds and forecasted future demand. The exact decision depends on the conditions implemented in `battery\_simulator.py`.



The principal design goals are:



\- Avoid discharging below the minimum permitted SOC.

\- Respect maximum battery charge and discharge power.

\- Preserve energy when future demand indicates that reserve protection may be beneficial.

\- Use available renewable surplus for charging when the configured logic permits it.

\- Calculate grid backup when the available supply cannot cover the load.



\### Important Distinction



A lower grid demand peak does not necessarily mean lower total energy consumption or lower electricity cost. A complete economic evaluation would also require grid tariffs, battery efficiency, degradation, and other operating costs.



\---



\## 🧰 Technology Stack



| Technology | Purpose |

|---|---|

| Python | Core programming language and simulation logic |

| Flask | Web application and dashboard backend |

| Pandas | Tabular data processing |

| NumPy | Numerical calculations, where used |

| Scikit-learn | Machine-learning model development and prediction |

| Joblib | Saving and loading trained model artifacts |

| HTML/CSS/JavaScript | Dashboard interface and browser-side presentation |

| Matplotlib / other chart libraries | Data visualization, where used |

| Pytest | Automated testing, where configured |

| Git and GitHub | Version control and project collaboration |

| Mermaid | Documentation diagrams rendered by supported platforms |



\---



\## 📁 Project Structure



The main files and directories currently used by the project include:



```text

SmartEnergyMicrogrid/

│

├── app.py

├── battery\_simulator.py

├── forecast\_engine.py

├── grid\_controller.py

├── predict.py

├── train\_model.py

├── create\_future\_weather.py

├── inspect\_data.py

├── test\_eval.py

│

├── requirements.txt

├── README.md

├── .gitignore

│

├── models/

│   ├── demand\_model.joblib

│   ├── feature\_names.joblib

│   └── model\_comparison.csv

│

├── templates/

│   └── Dashboard HTML templates

│

├── tests/

│   └── test\_battery\_simulator.py

│

├── forecast\_results.csv

├── battery\_simulation\_results.csv

├── baseline\_simulation\_results.csv

├── future\_weather\_demo.csv

│

└── starter\_kit/

```



\*\*Note:\*\* The listing is representative of the current project files. Confirm the actual contents of `templates/` and `starter\_kit/` before treating the diagram as an exhaustive directory listing. Generated result files may be recreated by running the application or simulation.



\---



\## ⚙️ Installation and Setup



\### Prerequisites



\- Python version compatible with the project's dependencies.

\- Git, if cloning the repository.

\- A terminal such as Windows Command Prompt or PowerShell.

\- A web browser.



\### Step 1: Clone the Repository



Replace `YOUR\_GITHUB\_USERNAME` with the GitHub account that hosts the repository.



```bash

git clone https://github.com/YOUR\_GITHUB\_USERNAME/SmartEnergyMicrogrid.git

cd SmartEnergyMicrogrid

```



Alternatively, download the project ZIP from GitHub and extract it.



\### Step 2: Create a Virtual Environment



\*\*Windows Command Prompt:\*\*



```cmd

python -m venv .venv

```



Activate it:



```cmd

.venv\\Scripts\\activate

```



\### Step 3: Install Dependencies



```cmd

python -m pip install --upgrade pip

pip install -r requirements.txt

```



\### Step 4: Verify Dependencies



```cmd

python -m pip check

```



The expected result is:



```text

No broken requirements found.

```



If installation fails, check the Python version and the dependency versions specified in `requirements.txt`.



\---



\## ▶️ Running the Application



From the project root, with the virtual environment activated:



```cmd

python app.py

```



Read the terminal output for the local URL printed by Flask, then open that address in your browser. A common development address is:



```text

http://127.0.0.1:5000/

```



The exact route depends on the application's Flask route definitions.



To stop the development server, press `Ctrl + C` in the terminal.



\*\*Security note:\*\* Flask's development server is intended for local development, not direct public production hosting.



\---



\## 📊 Simulation Outputs and Validation



The simulation produces results related to demand, battery operation, and grid support. Depending on the run and configured inputs, outputs may include:



\- `forecast\_results.csv` — demand-forecast results.

\- `battery\_simulation\_results.csv` — managed battery dispatch results.

\- `baseline\_simulation\_results.csv` — baseline comparison results.

\- `models/model\_comparison.csv` — model comparison information.



The actual columns depend on the current implementation.



\### Validation Checks



The simulation validation is designed to examine:



1\. \*\*Power balance:\*\* Whether the modeled demand is accounted for by the modeled supply components, within the chosen tolerance.

2\. \*\*SOC limits:\*\* Whether the battery SOC stays within its configured minimum and maximum.

3\. \*\*Baseline comparison:\*\* How the managed dispatch profile differs from the baseline profile.

4\. \*\*Peak shaving:\*\* The difference between the maximum baseline grid demand and maximum managed grid demand.



\### Peak-Shaving Calculation



The percentage reduction in the maximum grid demand can be calculated as:



\\\[

\\text{Peak Reduction (\\%)} =

\\frac{P\_{\\text{baseline}}-P\_{\\text{managed}}}

{P\_{\\text{baseline}}}\\times100

\\]



Here, \\(P\_{\\text{baseline}}\\) is the baseline maximum grid demand and \\(P\_{\\text{managed}}\\) is the managed maximum grid demand over the same evaluation horizon.



Use the same dataset, time horizon, and comparable operating assumptions for both runs.



\### Results Reporting



Record the following values from a reproducible simulation run before publishing final performance claims:



| Metric | Result |

|---|---|

| Forecast evaluation metric | To be measured |

| Baseline maximum grid demand (kW) | To be measured |

| Managed maximum grid demand (kW) | To be measured |

| Peak grid-demand reduction (%) | To be calculated |

| Minimum battery SOC (%) | To be measured |

| Maximum battery SOC (%) | To be measured |

| Power-balance validation | To be verified |

| SOC-limit validation | To be verified |



Replace the placeholders with actual measured results from the final run. Avoid presenting illustrative or unverified values as benchmark results.



\---



\## 🖥️ Dashboard



The Flask dashboard is the visual interface for exploring the simulation.



Depending on the current implementation, it can present demand forecasts, battery SOC, dispatch behavior, grid backup, and baseline comparisons.



\### Suggested Dashboard Screenshots



Add screenshots after confirming the dashboard is running correctly.



```text

docs/

└── screenshots/

&#x20;   ├── dashboard-overview.png

&#x20;   ├── demand-forecast.png

&#x20;   ├── battery-soc.png

&#x20;   └── grid-comparison.png

```



When the screenshots exist in the repository, they can be embedded in this README as follows:



```markdown

!\[Dashboard Overview](docs/screenshots/dashboard-overview.png)

```



Use genuine screenshots captured from the application. Do not leave broken image links in the published README.



\---



\## 🧪 Testing



Run the battery simulator test file using:



```cmd

python -m pytest tests\\test\_battery\_simulator.py -v

```



If Pytest is not installed:



```cmd

python -m pip install pytest

```



Then rerun the test command.



Tests help verify expected behavior and catch regressions during development. A successful test run does not, by itself, prove that the model is accurate for real-world energy systems.



\---



\## 🌍 Potential Applications



The project's concepts may be useful for demonstrations or further research in:



\- Smart microgrid energy management.

\- Battery energy storage systems.

\- Renewable energy integration.

\- Demand-response simulations.

\- Grid peak-demand analysis.

\- Energy management education.

\- Forecast-aware battery scheduling research.



These are potential applications of the concepts, not claims of deployment in operational electrical infrastructure.



\---



\## ⚠️ Limitations



\- Forecast quality depends on the data and model used.

\- Simulated results depend on battery parameters, SOC limits, and dispatch assumptions.

\- The model may not account for every real-world effect, such as battery ageing, temperature, inverter losses, and detailed tariff structures.

\- Peak shaving does not automatically establish financial savings.

\- The dashboard is a monitoring and simulation interface, not a certified industrial control system.

\- Real-world deployment would require additional validation, electrical protection, safety engineering, and hardware integration.



\---



\## 🚀 Future Enhancements



Potential improvements include:



\- Incorporating solar and wind generation forecasts.

\- Adding battery degradation and round-trip efficiency models.

\- Adding time-of-use electricity tariffs and cost optimization.

\- Comparing forecasting algorithms using MAE, RMSE, and other appropriate metrics.

\- Adding weather-aware dispatch and uncertainty handling.

\- Introducing configurable battery and microgrid parameters.

\- Adding downloadable simulation reports.

\- Deploying the dashboard with a production-ready WSGI server.

\- Adding automated tests and a continuous-integration workflow.

\- Evaluating the strategy on more datasets and longer simulation horizons.



\---



\## 🤝 Contributing



Contributions and suggestions are welcome.



1\. Fork the repository.

2\. Create a feature branch.

3\. Implement and test the change.

4\. Commit with a clear message.

5\. Open a pull request describing the change.



Please avoid committing virtual environments, credentials, personal information, or unnecessary backup files.



\---



\## 📄 License



No license has been specified yet. Before publishing the repository for reuse, add a `LICENSE` file with the license you intend to use. Without an explicit license, other users generally do not receive permission to reuse, modify, or redistribute the project's code beyond applicable legal exceptions.



\---



\## 👨‍💻 Project Summary



\*\*Smart Energy Microgrid\*\* demonstrates how demand forecasting, battery SOC management, grid backup calculations, and peak-shaving analysis can be combined in a simulation-based energy management application.



The project brings together machine learning, energy-system simulation, software engineering, and interactive visualization in one workflow.



\*Built as an educational and hackathon-oriented prototype for intelligent energy management.\*


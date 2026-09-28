Readme_intro_CDC-OpenData.pdf		last Change: 11.08.2020

CDC-OpenData area

You have free access to the climate data of the DWD Climate Data Center (CDC).
Please note the terms of use of the CDC (https://opendata.dwd.de/climate_environment/CDC/Terms_of_use.pdf)  
 
We provide at the CDC-OpenData area (https://opendata.dwd.de/climate_environment/)
 
1.	observed parameters from DWD stations (https://opendata.dwd.de/climate_environment/CDC/observations_germany/)
2.	derived parameters at the station locations (https://opendata.dwd.de/climate_environment/CDC/derived_germany/)
3.	gridded fields covering Germany (https://opendata.dwd.de/climate_environment/CDC/grids_germany/)
4.	regional averages for Germany and its federal states (https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/)
5.	gridded fields covering Europe (https://opendata.dwd.de/climate_environment/CDC/grids_europe/)
6.	regional reanalysis (https://opendata.dwd.de/climate_environment/REA/)
7.	global climate station data (https://opendata.dwd.de/climate_environment/CDC/observations_global/)
 
at hourly, daily, monthly, annual or multi-annual resolution (details see below). 1-minute precipitation measurements and 10-minute measurements of temperature, precipitation, wind and sunshine are also available.

Regularly versioned data are sorted into the two subdirectories: 'recent' and 'historical'. Recent data have not yet completed the full quality control. Historical data have completed the operational quality control. The very 
latest 1-minute and 10-minute measurements are in the subdirectories 'now'.

As a consequence of our permanent quality control, there may be errors detected at a later stage and subsequently corrected in the archived data. Also, the on-going digitization of 
historical data extends the time series continuously. Thus, the archived climate data are versioned. In the CDC-OpenData area you will find the most recent valid version in the sub-directory 'historical'.

The time series can include inhomogeneities (i.e., caused by change of station location or instrument). Data users are urged to study the accompanying station metadata and the data set descriptions.

Please be aware that not all meteorological parameters are provided at all temporal resolutions, and that release dates of new data may vary. 

For changes please follow the "change log" (https://opendata.dwd.de/climate_environment/CDC/Change_log_CDC_ftp.txt)
and/or subscribe to the "CDC-Newsletter" (https://www.dwd.de/EN/service/newsletter/newsletter_cdc_node.html)
Changes to be expected in the future (https://opendata.dwd.de/climate_environment/CDC/Announce_log_CDC_ftp.txt)
List of registered errors (https://opendata.dwd.de/climate_environment/CDC/Error_log_CDC_ftp.txt)

Data on the CDC-OpenData area

1. Parameters observed at DWD stations and partner stations on equal terms

Historical and recent meteorological parameters (https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/), e.g., air temperature, soil temperature, precipitation, humidity, pressure, wind speed and direction,
visibility, solar irradiance, sunshine duration, and cloud cover.

The data are zipped according to station, including the station meta-data (in German language). Available resolution: 10-minutes, hourly, daily, monthly and multi-annual values (1961-1990, 1971-2000, 1981-2010). Approximately 400
climate stations are currently active. The corresponding station lists with all climate stations for the corresponding parameters are available on the Internet under "help-directory".

In the past, hourly measurements used to be costly and were therefore only performed at few stations. Instead, meteorological measurements were taken at standardized times (so-called "Terminwerte"). In some cases, the time series
of these observations are significantly longer than the hourly values. The specific observation times were defined by the respective observation regulations. Please consider the supplied metadata in the zip files. The data are
available in the following directories:
*	Air pressure
*	Temperature
*	Cloudiness
*	Vapour pressure and relative humidity
*	Soil condition
*	Visibility
*	Wind 

For a selection of 81 climate stations distributed over Germany, "Terminwerte" (of 36 meteoro-logical parameters) are provided in the traditional KL-format.

For urban climate monitoring, special stations are collected in the List of the City stations in hourly resolution.

Precipitation data from a dedicated precipitation observation network (with about 2000 active stations) and partner stations on equal terms are available at 1-minute, 10-minute, hourly, daily 
and monthly resolution. The list of precipitation stations is also available in the "help"-directory.

Snow depth and water equivalents are provided at daily resolution.

Observed weather phenomena at German stations; these include (thunderstorm; black ice; graupel; hail; fog; frost; storm>=6 Bft; storm>=8 Bft; dew) (observed during the course of the day). The directory includes historical 
observations, i.e. some of the time series date back to the 19th century and are available in daily, monthly and annual resolution.

Phenological data are collected at about 1200 active stations. The state of development of selected plants (e.g., apple, birch, snow drops, goose berry, wheat, vine, etc.) is reported by immediate reporters and annual reporters).
The lists of phenological stations can be found in list of phenology annual reporters and list of phenology immediate reporters.

Data from radiosonde ascents

The subdirectory high_resolution contains measurement series with high temporal resolution (2-10 seconds) and spatial coordinate information (balloon position). The low_resolution directory contains data in the traditional output
form (main print areas and prominent points).

Monthly profiles of air temperature from 12 radiosonde stations (balloon soundings) are provided in their original and in a homogenized version.

2. Parameters derived at the station locations

Agrometeorological models deliver soil parameters including the potential and real evaporation over grass and sandy clay, the soil moisture below sand and sandy clay, the soil temperatures at 5, 10, 20, 50 and 100 cm depth below bare
soil, and the maximal frost penetration depth. Available resolution: daily, monthly and multi-annual. The soil parameters are calculated at about 320 station locations, the time series start 1991.
A list and a map is provided for the locations for which the calculations were performed.

Technical parameters include heating degree days and cooling days.

3. Gridded fields covering Germany

Gridded fields cover Germany generally at different temporal resolutions (not every parameter is given at all resolutions).

The following precipitation grids are available:
-	With continuous updates:
o	RADOLAN precipitation grids are derived from radar data and station data (hourly, daily).
o	REGNIE precipitation grids are derived from stations measurements only (daily, monthly, multi-year).
o	Stations are also the data basis for grids in monthly, half-yearly, annual and multi-year resolution (climate monitoring starting 1881).
-	Reprocessed based on radar data with better correction methods:
o	RADKLIM Version 2017.002: Reprocessed gauge-adjusted radar data, one-hour precipitation sums (from 2001 onwards).
o	RADKLIM Version 2017.002: Reprocessed quasi gauge-adjusted radar data 5-minute precipitation sums (from 2001 onwards).

Statistics on heavy precipitation (KOSTRA-DWD) provide return periods for heavy precipitation in Germany with various duration thresholds.

Agrometeorological models provide soil parameters: soil moisture, soil temperature at 5 cm depth, potential and real evaporation are available at daily, monthly and multi-annual resolution.

Air temperature (mean, max, min), sunshine duration, drought index, as well as the numbers of days with snow, frost days, or exceeding certain thresholds for temperature or for precipitation are given at monthly, annual or multi-annual
resolution.

Gridded solar radiation values (1 x 1 km) comprise global radiation (monthly, annual and multi-annual means), diffuse radiation (monthly, annual) and direct radiation (monthly, annual), all are derived from satellite data and ground
based measurements.

Wind energy related parameters derived from station measurements are given as multi-annual mean, for both 1 x 1 km resolution and 200 x 200 m resolution.

In the project QuWind100 a quantitative wind climatology for current and future wind energy applications at altitudes between 100 and 200 m was established for Germany for the period  
1981 - 2010. This horizontally high-resolution (100 m x 100 m) and spatially comprehensive data set takes into account the influence of different surface types as well as the time of day and 
season on the wind field.

In the project TRY (Test reference years) for the time span 1995 - 2012 the monthly, daily and hourly grids at 1 x 1 km resolution were calculated for air temperature, moisture, pressure, 
dew point, water vapor content, cloud cover, wind direction and wind speed, as well as direct, solar incoming and long-wave outgoing radiation.

From the phenological observations over Germany annual grids are derived for approx. 50 phenological phases, also begin of vegetation (annual, multi-annual) and end of vegetation (annual, multi-annual).

4. Regional averages for Germany and its federal states

These monthly, seasonal and annual averages (of air temperature, precipitation and sunshine duration) are derived from the gridded fields covering Germany.

Regional averages of climatological indices are available as annual averages for the following parameters:
*	Days with precipitation = 10 mm
*	Days with precipitation = 20 mm
*	Hot days (days with maximum temperature = 30°C) 
*	Summer days (days with minimum temperature = 25°C)
*	Frost days (days with minimum temperature < 0°C)
*	Ice days (days with maximum temperature < 0°C) 

5. Gridded fields covering Europe

For the time interval 2001 - 2010, a 5 x 5 km grid is given in monthly and daily resolution, for air temperature (mean, max, min) at 2 m above ground, and wind speeds at 10 m above ground.

Monthly cloud cover is derived from satellite data.

6. Regional reanalysis

Selected parameters of the regional reanalysis are available under COSMO-REA6. The hourly fields cover Europe for the time span 1995 - 2016, in 6 x 6 km resolution. The format is the original format of COSMO (DWD grib1, rotated
coordinates). Following parameters are provided: pressure (reduced and not reduced) at the surface, precipitation, temperature (min, max, mean), relative and specific humidity, wind components U and V, wind gust, various parameters
of radiation (diffuse and direct), planetary boundary layer height, integrated water vapor and cloud cover; according to parameter either at the lowest 6 model levels and/or at 10 m height (wind) above model ground and/or 2 m 
height (temperature) above model ground.

7. Global climate station data

Historical and recent monthly station data from CLIMAT-messages (quality controlled) for air temperature (mean, max, min), precipitation, number of days with precipitation, sunshine duration, pressure, vapour pressure, and 
derived multi-annual means. In addition, the monthly files of CLIMAT-messages, checked for month, year and format are provided, these contain more parameters.

More ways to access data

DWD provides data access (also to other data) via other systems (https://www.dwd.de/EN/climate_environment/cdc/clis/clis_node.html). 

Contact:
Climate and environment
Central sales department
Phone: +49 (0)69 8062 4400
Fax: +49 (0)69 8062 4499
Mail: klima.vertrieb@dwd.de

Status: August 2020
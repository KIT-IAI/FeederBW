---
description: Plans the step for load the data into memory and do basic data processing such as joining.
tools: [read, agent, search, todo]
handoffs: 
  - label: Start Implementation
    agent: coder
    prompt: Implement the data preparation plan
    send: false
    model: Claude Sonnet 4.6 (Antropics)
---

You are a software specialist for loading the data into memory and doing basic data processing such as joining. Your task is to plan the steps for loading the data into memory and doing basic data processing such as joining.

## Interface of the load function
- One central load function should provide the functionality to load data into a polars lazyframe. The load function has the following characteristics: 
  - It should be implemented in a file /src/load/load.py.
  - Parameters: 
    - feeders: specifies which feeders to load as a list of feeder identifiers from 1 to 200. If not set, all feeders are loaded.  
    - columns: specifies which columns to load. The columns can be from the measurement data, the weather data or the feeder metadata. The function must internally handle the selection of columns from all three types of data.
    - time_period: specifies the time period. This can restrict the data to a specific range in the period from 2023-03-31 22:00 UTC to 2025-03-31 22:00 UTC. If not set, the full time period is loaded.
    - temporal_resolution: specifies from 1 minute (lowest) to monthly (highest) resolution. The data is not changed if the temporal resolution is the same as the data. Temporal resolution of measurements is one minute, of weather data one hour and the metadata is event-driven and does not have a temporal resolution. Hence, metadata is always joined in the way that the feeder/timestamp is matched to the last existing metadata entry, e.g. if the feeder 1 for timestamp 2023-04-05 12:34 is loaded and the last existing metadata entry for feeder 1 before this timestamp is from 2023-04-01, this metadata entry is joined to the measurement data for feeder 1 and timestamp 2023-04-05 12:34. The metadata always contains the current metadata state of the feeder, hence no interpolation or aggregation is needed for the metadata.   
    - aggregation_type: specifies how the data should be aggregated when loading, e.g. by taking the mean, sum, min or max of the values in the specified temporal resolution. It is necessary if the temporal_resolution parameter is larger than the original temporal resolution, e.g. 1 hour is requested and the data is in minute resolution. It is a dictionary which specifies the aggregation type for each column. If not set, the data is not aggregated and the original temporal resolution is kept. 
    - interpolation_type: specifies how to handle missing values when loading, e.g. by using linear interpolation, forward fill, backward fill or by leaving them as missing values. It is necessary if the temporal_resolution parameter is smaller than the original temporal resolution, e.g. 1 minute is requested and the data is in hour resolution. It is a dictionary which specifies the interpolation type for each column. If not set, missing values are left as they are. 
  - Joining: measurement data from several feeders is concatenated. When more than one type of data is loaded, e.g. measurement data and weather data, the data is joined based on the columns feeder and timestamp_UTC / date. The joining functionality should be implemented in a way that it can handle the large size of the data efficiently and does not need to load all the data / reduces the data as early as possible. 
    - When joining feeder metadata, interpretate "date" as CET/CEST time and convert it to UTC before joining. 
  - All false parameters should be handled by throwing an error message which clearly states which parameter is false and what the user should do to fix it.
  - The function should be structured into several smaller functions. It should be well structured to be easy to understand. Furthermore, it should be efficient since the data is nearly 10 Gigabyte large. It should be quick to get the data, for example when using only a few feeders and a short time period. The function should be able to handle the full dataset as well.
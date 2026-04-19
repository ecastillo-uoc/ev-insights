# Queries for data ingestion
get_charging_point_type_id_by_type = """
    SELECT id
    FROM evinsights."ChargingPointType"
    WHERE charging_point_type = (%s)
"""

get_charging_stations = """
    SELECT id, orig_id
    FROM evinsights."ChargingStation"
    WHERE dataset_id = %s
"""

get_users = """
    SELECT id, orig_id
    FROM evinsights."User"
    WHERE dataset_id = %s OR %s IS NULL
"""

insert_charging_point_types = """
    INSERT INTO evinsights."ChargingPointType" (charging_point_type)
    VALUES (%s)
"""

insert_dataset_info = """
    INSERT INTO evinsights."Dataset" (name, url, description, owner, country, region, city, dataset_directory, file_name, file_type, delimiter, encoding, 
                                   license, charging_point_type, notes)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

insert_charging_stations = """
    INSERT INTO evinsights."ChargingStation" (orig_id, manufacturer, model, station_type, num_plugs, max_charging_power, max_discharging_power, 
                                           ocpp_version, longitude, latitude, postal_code, dataset_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

insert_charging_sessions = """
    INSERT INTO evinsights."ChargingSession" (orig_session_id, plug_in_datetime, plug_out_datetime, charge_end_datetime, charge_end_datetime_presence, 
                                             energy_supplied, fk_dataset_id, fk_charging_station_id, fk_user_id) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

insert_users = """
    INSERT INTO evinsights."User" (orig_id, ev_id, ev_manufacturer, ev_model, ev_battery_capacity_kWh, ev_battery_type, 
                                  ev_battery_useable_capacity, ev_v2g, ev_max_charging_power, ev_max_discharging_power, dataset_id) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

insert_electric_vehicle = """
    INSERT INTO evinsights."ElectricVehicle" (ev_manufacturer, ev_model, ev_battery_capacity_kWh)
    VALUES (%s, %s, %s)
"""

# Queries for analysis
get_plug_in_datetime_and_energy_supplied = """
    SELECT plug_in_datetime, energy_supplied
    FROM evinsights."ChargingSession"
"""


# TODO generic query for analysis to select any column from DB
# Queries for analysis


# Queries for getting dataset info
get_dataset_list = """
    SELECT name
    FROM evinsights."Dataset";
"""

# Get all unique countries
get_all_countries = """
    SELECT DISTINCT country
    FROM evinsights."Dataset"
    WHERE country IS NOT NULL
    ORDER BY country;
"""

# Get all unique years
get_all_years = """
    SELECT DISTINCT EXTRACT(YEAR FROM plug_in_datetime)::INTEGER as year
    FROM evinsights."ChargingSession"
    WHERE plug_in_datetime IS NOT NULL
    ORDER BY year;
"""

# Query to get the number of tables
get_number_of_tables = """
     SELECT schemaname, tablename
     FROM pg_tables
     WHERE schemaname IN ('evinsights');
 """

# Get dataset_id by name
get_dataset_id_by_name = """
    SELECT id
    FROM evinsights."Dataset"
    WHERE name = %s;
"""

# Get dataset_name by id
get_dataset_name_by_id = """
    SELECT name
    FROM evinsights."Dataset"
    WHERE id = %s;
"""

# Check column existence
check_column_existence = """
    SELECT table_name
    FROM information_schema.columns
    WHERE column_name = %s
    AND table_schema = 'evinsights';
"""

# Dynamic query
dynamic_query = """
    SELECT %s
    FROM evinsights."ChargingSession" as cse 
	LEFT JOIN evinsights."Dataset" as d ON cse.fk_dataset_id = d.id 
	LEFT JOIN evinsights."ChargingStation" as cst ON cse.fk_charging_station_id = cst.id
	LEFT JOIN evinsights."User" as u ON cse.fk_user_id = u.id
    WHERE d.name IN %s
    AND (%s IS NULL OR cse.plug_in_datetime >= %s)
    AND (%s IS NULL OR cse.plug_in_datetime <= %s)
    AND (%s IS NULL OR u.id = %s)
    AND (%s IS NULL OR cst.id = %s);
"""

insert_charging_station_forecast = """ 
    INSERT INTO evinsights."ChargingStationForecast" (date, energy, connections, experiment_id, run_id, created_at, fk_charging_station_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

insert_user_forecast = """ 
    INSERT INTO evinsights."UserForecast" (date, energy, duration, experiment_id, run_id, created_at, fk_user_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

# Get last user prediction
get_user_prediction = """
    SELECT *
    FROM evinsights."UserForecast"
    WHERE fk_user_id = %s
    AND date = %s
    ORDER BY created_at DESC
    LIMIT 1;
"""

# Get last charging_station prediction
get_charging_station_prediction = """
    SELECT *
    FROM evinsights."ChargingStationForecast"
    WHERE fk_charging_station_id = %s
    AND date = %s
    ORDER BY created_at DESC
    LIMIT 1;
"""

# Get all predictions for a charging station
get_charging_station_predictions_all = """
    SELECT date, energy, connections, experiment_id, run_id, created_at
    FROM evinsights."ChargingStationForecast"
    WHERE fk_charging_station_id = %s
    ORDER BY date;
"""

# Get all predictions for a user
get_user_predictions_all = """
    SELECT date, energy, duration, experiment_id, run_id, created_at
    FROM evinsights."UserForecast"
    WHERE fk_user_id = %s
    ORDER BY date;
"""

# Get daily actual energy and connections for a charging station
get_actual_daily_station = """
    SELECT DATE(cs.plug_in_datetime) as date,
           SUM(cs.energy_supplied) as energy,
           COUNT(*) as connections
    FROM evinsights."ChargingSession" cs
    WHERE cs.fk_charging_station_id = %s
    GROUP BY DATE(cs.plug_in_datetime)
    ORDER BY date;
"""

# Get daily actual energy and duration for a user
get_actual_daily_user = """
    SELECT DATE(cs.plug_in_datetime) as date,
           SUM(cs.energy_supplied) as energy,
           SUM(EXTRACT(EPOCH FROM (cs.plug_out_datetime - cs.plug_in_datetime)) / 60) as duration
    FROM evinsights."ChargingSession" cs
    WHERE cs.fk_user_id = %s
    GROUP BY DATE(cs.plug_in_datetime)
    ORDER BY date;
"""

# Get charging station IDs that have predictions
get_station_ids_with_predictions = """
    SELECT DISTINCT fk_charging_station_id as id
    FROM evinsights."ChargingStationForecast"
    ORDER BY id;
"""

# Get user IDs that have predictions
get_user_ids_with_predictions = """
    SELECT DISTINCT fk_user_id as id
    FROM evinsights."UserForecast"
    ORDER BY id;
"""

get_user_ids = """
    SELECT id
    FROM evinsights."User"
    WHERE dataset_id = %s OR %s IS NULL
"""

get_charging_station_ids = """
    SELECT id
    FROM evinsights."ChargingStation"
    WHERE dataset_id = %s OR %s IS NULL
"""

# Delete ChargingSession records by dataset_id
delete_charging_sessions_by_dataset = """
    DELETE FROM evinsights."ChargingSession"
    WHERE fk_dataset_id = %s
    RETURNING id;
"""

# # Delete UserForecast records by dataset_id
# delete_user_forecast_by_dataset = """
#     DELETE FROM evinsights."UserForecast"
#     WHERE fk_user_id IN (
#         SELECT id
#         FROM evinsights."User"
#         WHERE dataset_id = %s
#     )
#     RETURNING id;
# """

# # Delete ChargingStationForecast records by dataset_id
# delete_charging_station_forecast_by_dataset = """
#     DELETE FROM evinsights."ChargingStationForecast"
#     WHERE fk_charging_station_id IN (
#         SELECT id
#         FROM evinsights."ChargingStation"
#         WHERE dataset_id = %s
#     )
#     RETURNING id;
# """

# Delete User records by dataset_id
delete_users_by_dataset = """
    DELETE FROM evinsights."User"
    WHERE dataset_id = %s
    RETURNING id, orig_id;
"""

# Delete ChargingStation records by dataset_id
delete_charging_stations_by_dataset = """
    DELETE FROM evinsights."ChargingStation"
    WHERE dataset_id = %s
    RETURNING id, orig_id;
"""

# Delete Dataset record by id
delete_dataset_by_id = """
    DELETE FROM evinsights."Dataset"
    WHERE id = %s
    RETURNING id, name;
"""
# Get countries filtered by dataset names
get_countries_by_datasets = """
    SELECT DISTINCT country
    FROM evinsights."Dataset"
    WHERE country IS NOT NULL
    AND name = ANY(%s)
    ORDER BY country;
"""

# Get years filtered by dataset names
get_years_by_datasets = """
    SELECT DISTINCT EXTRACT(YEAR FROM cs.plug_in_datetime)::INTEGER as year
    FROM evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d ON cs.fk_dataset_id = d.id
    WHERE cs.plug_in_datetime IS NOT NULL
    AND d.name = ANY(%s)
    ORDER BY year;
"""

# Get years filtered by dataset names and countries
get_years_filtered = """
    SELECT DISTINCT EXTRACT(YEAR FROM cs.plug_in_datetime)::INTEGER as year
    FROM evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d ON cs.fk_dataset_id = d.id
    WHERE cs.plug_in_datetime IS NOT NULL
    AND (%s::varchar[] IS NULL OR d.name = ANY(%s))
    AND (%s::varchar[] IS NULL OR d.country = ANY(%s))
    ORDER BY year;
"""

# Get Charging Point Types filtered
get_charging_point_types_filtered = """
    SELECT DISTINCT cst.type
    FROM evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d ON cs.fk_dataset_id = d.id
    JOIN evinsights."ChargingStation" cst ON cs.fk_charging_station_id = cst.id
    WHERE cst.type IS NOT NULL
    AND (%s::varchar[] IS NULL OR d.name = ANY(%s))
    AND (%s::varchar[] IS NULL OR d.country = ANY(%s))
    AND (%s::integer[] IS NULL OR EXTRACT(YEAR FROM cs.plug_in_datetime)::INTEGER = ANY(%s))
    ORDER BY cst.type;
"""

# Get filtered record count
get_count_filtered = """
    SELECT COUNT(*) as count
    FROM evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d ON cs.fk_dataset_id = d.id
    JOIN evinsights."ChargingStation" cst ON cs.fk_charging_station_id = cst.id
    WHERE 1=1
    AND (%s::varchar[] IS NULL OR d.name = ANY(%s))
    AND (%s::varchar[] IS NULL OR d.country = ANY(%s))
    AND (%s::integer[] IS NULL OR EXTRACT(YEAR FROM cs.plug_in_datetime)::INTEGER = ANY(%s))
    AND (%s::varchar[] IS NULL OR cst.type = ANY(%s));
"""

DEFAULT_DATAFRAME_COLUMNS = [
    'plug_in_datetime',              # When vehicle is plugged-in (timezone-less)
    'plug_out_datetime',             # When vehicle is plugged-out (timezone-less)
    'charge_end_datetime',           # When charging session ends (timezone-less)
    'charge_end_datetime_presence',  # This is true if the real charge_end_datetime is provided, false if it is set to plug_out_datetime
    'energy_supplied',               # Energy delivered to the EV in kWh
    'max_charging_power',            # Max charging power set by charger, if any
    'ev_max_charging_power',         # Max charging power set by EV, if any
    'ev_id',                         # ID of the EV, if any
    'charging_station_id',           # ID of the EV charger, if any
    'user_id',                       # ID of the user, if any
]

DEFAULT_EV_DATAFRAME_COLUMNS = [
    'ev_manufacturer',
    'ev_model',
    'ev_battery_capacity_kWh'
]

DEFAULT_CHARGING_STATION_DATAFRAME_COLUMNS = [
    'origin_id',
    'ocpp_version',
    'longitude',
    'latitude',
    'connector',
    'energy_year_Wh',
    'power_W_avg'
]

DEFAULT_CHARGING_POINT_TYPES = [
    'private_domestic',
    'private_workplace',
    'public'
]

# https://www.power-sonic.com/ev-charging-connector-types/
DEFAULT_CONNECTOR_TYPES = [
    'T3',
    'T2',
    'TE',
    'CHAdeMO',
    'Combo'
]

# Mapping of various connector names to standard database codes
CONNECTOR_TYPE_ALIASES = {
    # Type 2 (Mennekes)
    'Type 2': 'T2',
    'Type2': 'T2',
    'Mennekes': 'T2',
    'IEC 62196-2': 'T2',
    'IEC 62196 Type 2': 'T2',
    'IEC62196-2': 'T2',

    # Type 3 (Scame)
    'Type 3': 'T3',
    'Type3': 'T3',
    'Type 3c': 'T3',
    'Scame': 'T3',
    'EV Plug Alliance': 'T3',

    # Type E / Domestic (Schuko)
    'Type E': 'TE',
    'TypeE': 'TE',
    'Schuko': 'TE',
    'Domestic': 'TE',
    'Standard': 'TE',
    'E/F': 'TE',
    'Wall Outlet': 'TE',

    # CHAdeMO
    'Chademo': 'CHAdeMO',
    'JEVS G105': 'CHAdeMO',

    # CCS (Combo)
    'CCS': 'Combo',
    'CCS2': 'Combo',
    'Combo 2': 'Combo',
    'CCS Combo 2': 'Combo',
    'Combined Charging System': 'Combo',
    'CCS Type 2': 'Combo',
    'Combo T2': 'Combo'
}

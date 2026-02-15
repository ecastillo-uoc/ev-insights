import os
import logging

import psycopg2

_logger = logging.getLogger("utils.globals")

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


def _get_db_config_from_env():
    return {
        'host': os.getenv('EVINSIGHTS_DB_HOST', 'localhost'),
        'port': int(os.getenv('EVINSIGHTS_DB_PORT', '5432')),
        'user': os.getenv('EVINSIGHTS_DB_USER', 'evinsights'),
        'password': os.getenv('EVINSIGHTS_DB_PASSWORD', 'evinsights'),
        'database': os.getenv('EVINSIGHTS_DB_NAME', 'evinsights'),
        'schema': os.getenv('EVINSIGHTS_DB_SCHEMA', 'evinsights'),
    }


def _query_rows(query, params=None):
    cfg = _get_db_config_from_env()
    conn = psycopg2.connect(
        host=cfg['host'],
        port=cfg['port'],
        user=cfg['user'],
        password=cfg['password'],
        database=cfg['database']
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or [])
            return cursor.fetchall()
    finally:
        conn.close()


def get_charging_point_types_from_db():
    cfg = _get_db_config_from_env()
    try:
        rows = _query_rows(
            f'SELECT name FROM {cfg["schema"]}."ChargingPointType" WHERE name IS NOT NULL ORDER BY name;'
        )
        values = [row[0] for row in rows]
        return values if values else DEFAULT_CHARGING_POINT_TYPES
    except (psycopg2.Error, OSError, ValueError, TypeError) as exc:
        _logger.warning("Using default CHARGING_POINT_TYPES because DB lookup failed: %s", exc)
        return DEFAULT_CHARGING_POINT_TYPES


def get_connector_types_from_db():
    cfg = _get_db_config_from_env()
    try:
        rows = _query_rows(
            f'SELECT code FROM {cfg["schema"]}."Connector" WHERE code IS NOT NULL ORDER BY code;'
        )
        values = [row[0] for row in rows]
        return values if values else DEFAULT_CONNECTOR_TYPES
    except (psycopg2.Error, OSError, ValueError, TypeError) as exc:
        _logger.warning("Using default CONNECTOR_TYPES because DB lookup failed: %s", exc)
        return DEFAULT_CONNECTOR_TYPES


def get_dataframe_columns_from_db():
    cfg = _get_db_config_from_env()
    try:
        rows = _query_rows(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
            ORDER BY ordinal_position;
            """,
            [cfg['schema'], 'ChargingSession']
        )

        session_columns = [
            row[0] for row in rows
            if row[0] not in {'id', 'fk_dataset_id', 'fk_charging_station_id', 'fk_user_id'}
        ]

        ordered_columns = [
            col for col in DEFAULT_DATAFRAME_COLUMNS
            if col in session_columns or col in {'max_charging_power', 'ev_max_charging_power', 'ev_id', 'charging_station_id', 'user_id'}
        ]

        if session_columns:
            return ordered_columns
        return DEFAULT_DATAFRAME_COLUMNS
    except (psycopg2.Error, OSError, ValueError, TypeError) as exc:
        _logger.warning("Using default DATAFRAME_COLUMNS because DB lookup failed: %s", exc)
        return DEFAULT_DATAFRAME_COLUMNS


# Backward-compatible module variables
DATAFRAME_COLUMNS = get_dataframe_columns_from_db()
CHARGING_POINT_TYPES = get_charging_point_types_from_db()
CONNECTOR_TYPES = get_connector_types_from_db()
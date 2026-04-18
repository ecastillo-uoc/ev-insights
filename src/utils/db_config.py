import os
import logging

import psycopg2

from .constants import (
    DEFAULT_DATAFRAME_COLUMNS,
    DEFAULT_CHARGING_POINT_TYPES,
    DEFAULT_CONNECTOR_TYPES,
)

_logger = logging.getLogger("utils.db_config")


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

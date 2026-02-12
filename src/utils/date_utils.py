def utc_to_decimal_hours_minutes(utc_time):
    """
    Converts a datetime object to decimal hours.
    Example: 10:30 becomes 10.5
    """
    return float(utc_time.strftime("%H")) + float(utc_time.strftime("%M")) / 60

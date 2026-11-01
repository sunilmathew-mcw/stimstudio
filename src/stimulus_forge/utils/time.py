"""Small timing helpers used by Stimulus Forge."""

import time


def get_time_hrs_mins_secs_ms(seconds):
    """Convert seconds to hours, minutes, seconds, and milliseconds."""
    hrs, secs = divmod(seconds, 3600)
    mins, secs = divmod(secs, 60)
    secs, ms = divmod(secs * 1000, 1000)
    return hrs, mins, secs, ms


def get_elapsed_time(start_time=None):
    """Return elapsed time since ``start_time`` in split time units."""
    return get_time_hrs_mins_secs_ms(time.time() - start_time)

import pytz

def localToUTC(t, local_tz):
    t_local = local_tz.localize(t, is_dst=None)
    t_utc = t_local.astimezone(pytz.utc)
    return t_utc

# It seems that this function is deprecated?
def UTCtimestampTolocal(ts, local_tz):
    t_utc = ts.datetime.fromtimestamp(ts, tz=pytz.timezone("UTC"))
    t_local = t_utc.astimezone(local_tz)
    return t_local
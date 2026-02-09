import pandas as pd
import re
import requests
import time

def geocode_place_mapbox_v5(place: str, token, *, country="MW", proximity=(33.78, -13.97), limit=1):
    q = f"{place}, Malawi"
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{requests.utils.quote(q)}.json"
    params = {
        "access_token": token,
        "country": country,
        "proximity": f"{proximity[0]},{proximity[1]}",
        "limit": limit,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    feats = data.get("features", [])
    if not feats:
        return None

    lon, lat = feats[0]["center"]
    return {
        "query": place,
        "lon": lon,
        "lat": lat,
        "place_name": feats[0].get("place_name"),
        "relevance": feats[0].get("relevance"),
        "feature_id": feats[0].get("id"),
    }

def geocode_unique_queries_mapbox(df, token, query_col="geocode_query", sleep_s=0.05):
    # 1) unique non-missing queries
    uniq = (
        df[query_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # 2) cache results in a dict: query -> result dict (or None)
    cache = {}

    for q in uniq:
        try:
            cache[q] = geocode_place_mapbox_v5(q, token)
        except Exception as e:
            # if something goes wrong, keep record and continue
            cache[q] = {"query": q, "lon": None, "lat": None, "error": str(e)}
        time.sleep(sleep_s)

    # 3) turn cache into a dataframe
    geo_df = pd.DataFrame(
        [
            {
                "geocode_query": q,
                "lon": (res or {}).get("lon"),
                "lat": (res or {}).get("lat"),
                "place_name": (res or {}).get("place_name"),
                "relevance": (res or {}).get("relevance"),
                "feature_id": (res or {}).get("feature_id"),
                "error": (res or {}).get("error"),
            }
            for q, res in cache.items()
        ]
    )

    # 4) merge back
    out = df.merge(geo_df, on="geocode_query", how="left")
    return out, geo_df


def normalize(text):
    if pd.isna(text):
        return pd.NA
    return (
        str(text)
        .lower()
        .strip()
        .replace(",", "")
    )

def classify_place_2022(s):
    if pd.isna(s):
        return "missing"

    # Foreign countries
    if s in {"mozambique", "mozambiq", "zambia", "tanzania"}:
        return "foreign_country"

    # Districts / major cities in Malawi
    if s in {
        "lilongwe", "dedza", "salima", "kasungu", "dowa",
        "mzimba", "ntcheu", "mchinji", "balaka", "zomba"
    }:
        return "district_or_city"

    # Health facilities
    if s in {"chipatala", "hospital", "healthcentre", "health center"}:
        return "health_facility"

    # Lilongwe Areas (Area 11, area23, a23)
    if re.match(r"^(area\s*\d+|a\s*\d+)$", s):
        return "lilongwe_area"

    # Distance-based informal locations
    if re.match(r"^\d+\s*miles?$", s) or re.match(r"^\d+miles?$", s):
        return "district_or_city"

    # Everything else: assume named village
    return "village"



def prepare_village_for_geocoding(df, col="village"):
    # 1) normalize
    df[col] = df[col].astype("string")
    df[col + "_norm"] = df[col].apply(normalize)

    # 2) classify
    df["place_type"] = df[col + "_norm"].apply(classify_place_2022)

    # 3) decide what to send to Mapbox
    #    - geocode villages
    #    - optionally also geocode district/city (often useful fallback)
    allowed = {"village", "district_or_city"}
    df["geocode_query"] = df[col + "_norm"].where(df["place_type"].isin(allowed), pd.NA)

    return df

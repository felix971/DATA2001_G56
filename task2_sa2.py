import json
import sqlite3

import pandas as pd
import requests


BASE_URL = (
    "https://mapprod3.environment.nsw.gov.au/arcgis/rest/services/"
    "EDP/Administrative_Boundaries/MapServer"
)

POI_URL = "https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_POI/MapServer"


def list_nsw_sa4s():
    url = f"{BASE_URL}/6/query"
    params = {
        "where": "STATE_NAME = 'New South Wales'",
        "outFields": "SA4_CODE,SA4_NAME",
        "returnGeometry": "false",
        "f": "json",
    }
    data = requests.get(url, params=params).json()
    rows = [feature["attributes"] for feature in data["features"]]
    return pd.DataFrame(rows).sort_values("SA4_NAME").reset_index(drop=True)


def get_sa4_geometry(sa4_name):
    sa4_table = list_nsw_sa4s()
    real_name = sa4_table[
        sa4_table["SA4_NAME"].str.contains(sa4_name, case=False)
    ].iloc[0]["SA4_NAME"]

    url = f"{BASE_URL}/6/query"
    params = {
        "where": f"SA4_NAME = '{real_name}'",
        "outFields": "SA4_CODE,SA4_NAME",
        "returnGeometry": "true",
        "f": "json",
    }
    data = requests.get(url, params=params).json()
    return data["features"][0]["geometry"], real_name


def get_sa4_geometry_by_code(sa4_code):
    url = f"{BASE_URL}/6/query"
    params = {
        "where": f"SA4_CODE = '{sa4_code}'",
        "outFields": "SA4_CODE,SA4_NAME",
        "returnGeometry": "true",
        "f": "json",
    }
    data = requests.get(url, params=params).json()
    feature = data["features"][0]
    return feature["geometry"], feature["attributes"]["SA4_CODE"], feature["attributes"]["SA4_NAME"]


def get_bbox(geometry):
    points = []
    for ring in geometry["rings"]:
        for point in ring:
            points.append(point)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def get_sa2_bbox_df(sa4_name):
    geometry, real_name = get_sa4_geometry(sa4_name)

    url = f"{BASE_URL}/8/query"
    params = {
        "where": "STATE_NAME = 'New South Wales'",
        "outFields": "SA2_MAIN,SA2_NAME,STATE_NAME,AREA_SQKM",
        "returnGeometry": "true",
        "geometry": json.dumps(geometry),
        "geometryType": "esriGeometryPolygon",
        "inSR": 4283,
        "spatialRel": "esriSpatialRelContains",
        "f": "json",
    }
    data = requests.post(url, data=params).json()

    rows = []
    for feature in data["features"]:
        xmin, ymin, xmax, ymax = get_bbox(feature["geometry"])
        rows.append(
            {
                "sa4_name": real_name,
                "sa2_main": feature["attributes"]["SA2_MAIN"],
                "sa2_name": feature["attributes"]["SA2_NAME"],
                "state_name": feature["attributes"]["STATE_NAME"],
                "area_sqkm": feature["attributes"]["AREA_SQKM"],
                "bbox_xmin": xmin,
                "bbox_ymin": ymin,
                "bbox_xmax": xmax,
                "bbox_ymax": ymax,
            }
        )

    return pd.DataFrame(rows)


def get_sa2_bbox_df_by_code(sa4_code):
    geometry, real_code, real_name = get_sa4_geometry_by_code(sa4_code)

    url = f"{BASE_URL}/8/query"
    params = {
        "where": "STATE_NAME = 'New South Wales'",
        "outFields": "SA2_MAIN,SA2_NAME,STATE_NAME,AREA_SQKM",
        "returnGeometry": "true",
        "geometry": json.dumps(geometry),
        "geometryType": "esriGeometryPolygon",
        "inSR": 4283,
        "spatialRel": "esriSpatialRelContains",
        "f": "json",
    }
    data = requests.post(url, data=params).json()

    rows = []
    for feature in data["features"]:
        xmin, ymin, xmax, ymax = get_bbox(feature["geometry"])
        rows.append(
            {
                "sa4_code": real_code,
                "sa4_name": real_name,
                "sa2_main": feature["attributes"]["SA2_MAIN"],
                "sa2_name": feature["attributes"]["SA2_NAME"],
                "state_name": feature["attributes"]["STATE_NAME"],
                "area_sqkm": feature["attributes"]["AREA_SQKM"],
                "bbox_xmin": xmin,
                "bbox_ymin": ymin,
                "bbox_xmax": xmax,
                "bbox_ymax": ymax,
            }
        )

    return pd.DataFrame(rows)


def get_poi_by_bbox(xmin, ymin, xmax, ymax):
    url = f"{POI_URL}/0/query"
    params = {
        "where": "1=1",
        "outFields": "objectid,poiname,poitype,poigroup",
        "returnGeometry": "true",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4283,
        "spatialRel": "esriSpatialRelIntersects",
        "f": "json",
    }
    data = requests.get(url, params=params).json()

    rows = []
    for feature in data["features"]:
        rows.append(
            {
                "objectid": feature["attributes"]["objectid"],
                "poiname": feature["attributes"]["poiname"],
                "poitype": feature["attributes"]["poitype"],
                "poigroup": feature["attributes"]["poigroup"],
                "longitude": feature["geometry"]["x"],
                "latitude": feature["geometry"]["y"],
            }
        )

    return pd.DataFrame(rows)


def get_sa4_poi_df(sa4_name):
    sa2_bbox_df = get_sa2_bbox_df(sa4_name)
    table_list = []

    for _, row in sa2_bbox_df.iterrows():
        poi_df = get_poi_by_bbox(
            row["bbox_xmin"],
            row["bbox_ymin"],
            row["bbox_xmax"],
            row["bbox_ymax"],
        )
        poi_df["sa4_name"] = row["sa4_name"]
        poi_df["sa2_main"] = row["sa2_main"]
        poi_df["sa2_name"] = row["sa2_name"]
        table_list.append(poi_df)

    return pd.concat(table_list, ignore_index=True)


def get_sa4_poi_df_by_code(sa4_code):
    sa2_bbox_df = get_sa2_bbox_df_by_code(sa4_code)
    table_list = []

    for _, row in sa2_bbox_df.iterrows():
        poi_df = get_poi_by_bbox(
            row["bbox_xmin"],
            row["bbox_ymin"],
            row["bbox_xmax"],
            row["bbox_ymax"],
        )
        poi_df["sa4_code"] = row["sa4_code"]
        poi_df["sa4_name"] = row["sa4_name"]
        poi_df["sa2_main"] = row["sa2_main"]
        poi_df["sa2_name"] = row["sa2_name"]
        table_list.append(poi_df)

    return pd.concat(table_list, ignore_index=True)


def save_to_sqlite(sa2_bbox_df, poi_df, db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("pragma foreign_keys = on")
    conn.execute("drop table if exists poi")
    conn.execute("drop table if exists sa2_bbox")
    conn.execute(
        """
        create table sa2_bbox (
            member_name text,
            sa4_code text,
            sa4_name text,
            sa2_main text primary key,
            sa2_name text,
            state_name text,
            area_sqkm real,
            bbox_xmin real,
            bbox_ymin real,
            bbox_xmax real,
            bbox_ymax real
        )
        """
    )
    conn.execute(
        """
        create table poi (
            poi_id integer primary key autoincrement,
            objectid integer,
            poiname text,
            poitype text,
            poigroup integer,
            longitude real,
            latitude real,
            member_name text,
            sa4_code text,
            sa4_name text,
            sa2_main text,
            sa2_name text,
            foreign key (sa2_main) references sa2_bbox(sa2_main)
        )
        """
    )
    sa2_bbox_df.to_sql("sa2_bbox", conn, if_exists="append", index=False)
    poi_df.to_sql("poi", conn, if_exists="append", index=False)
    conn.close()


def read_sql(sql, db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

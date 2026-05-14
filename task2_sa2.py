import sqlite3
import struct
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from shapely.geometry import LinearRing, Point, Polygon
from shapely.ops import unary_union


POI_URL = "https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_POI/MapServer"
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SA4_SHP_PATH = DATA_DIR / "SA4_2021_AUST_SHP_GDA2020" / "SA4_2021_AUST_GDA2020.shp"
SA2_SHP_PATH = DATA_DIR / "SA2_2021_AUST_SHP_GDA2020" / "SA2_2021_AUST_GDA2020.shp"
DEFAULT_DB_PATH = DATA_DIR / "task2_poi.db"
SHAPEFILE_SPATIAL_REFERENCE = 7844
POI_OUTPUT_SPATIAL_REFERENCE = 4326
DEFAULT_SELECTED_AREAS = [
    ("area_1", "120"),
    ("area_2", "117"),
    ("area_3", "118"),
    ("area_4", "125"),
]


def _decode_dbf_value(raw_value):
    return raw_value.decode("utf-8", errors="replace").strip()


def _read_dbf(dbf_path):
    with Path(dbf_path).open("rb") as dbf_file:
        header = dbf_file.read(32)
        record_count = struct.unpack("<I", header[4:8])[0]
        header_length = struct.unpack("<H", header[8:10])[0]
        record_length = struct.unpack("<H", header[10:12])[0]

        fields = []
        while True:
            descriptor = dbf_file.read(32)
            if descriptor[0] == 0x0D:
                break

            field_name = (
                descriptor[:11].split(b"\x00", 1)[0].decode("ascii", errors="replace")
            )
            field_length = descriptor[16]
            fields.append((field_name, field_length))

        offsets = []
        position = 1
        for field_name, field_length in fields:
            offsets.append((field_name, position, position + field_length))
            position += field_length

        dbf_file.seek(header_length)
        records = []
        for _ in range(record_count):
            record = dbf_file.read(record_length)
            if not record or record[:1] == b"*":
                continue

            row = {}
            for field_name, start, end in offsets:
                row[field_name] = _decode_dbf_value(record[start:end])
            records.append(row)

    return records


def _rings_to_geometry(rings):
    clean_rings = []
    for ring in rings:
        if len(ring) < 3:
            continue
        if ring[0] != ring[-1]:
            ring = [*ring, ring[0]]
        if len(ring) >= 4:
            clean_rings.append(ring)

    if not clean_rings:
        return Polygon()

    outer_rings = []
    hole_rings = []
    for ring in clean_rings:
        if LinearRing(ring).is_ccw:
            hole_rings.append(ring)
        else:
            outer_rings.append(ring)

    if not outer_rings:
        outer_rings = hole_rings
        hole_rings = []

    polygons = []
    for outer_ring in outer_rings:
        outer_polygon = Polygon(outer_ring)
        holes = []

        for hole_ring in hole_rings:
            hole_polygon = Polygon(hole_ring)
            if outer_polygon.covers(hole_polygon.representative_point()):
                holes.append(hole_ring)

        polygon = Polygon(outer_ring, holes)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        polygons.append(polygon)

    if len(polygons) == 1:
        return polygons[0]

    return unary_union(polygons)


def _read_shp_geometries(shp_path):
    geometries = []
    with Path(shp_path).open("rb") as shp_file:
        shp_file.seek(100)

        while True:
            record_header = shp_file.read(8)
            if len(record_header) < 8:
                break

            _, content_length_words = struct.unpack(">2i", record_header)
            content = shp_file.read(content_length_words * 2)
            if len(content) < 4:
                continue

            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                geometries.append(Polygon())
                continue
            if shape_type not in {5, 15, 25}:
                raise ValueError(f"Unsupported shapefile geometry type: {shape_type}")

            part_count, point_count = struct.unpack("<2i", content[36:44])
            part_offset = 44
            point_offset = part_offset + (part_count * 4)
            parts = list(
                struct.unpack(f"<{part_count}i", content[part_offset:point_offset])
            )
            points = [
                struct.unpack(
                    "<2d",
                    content[point_offset + i * 16 : point_offset + (i + 1) * 16],
                )
                for i in range(point_count)
            ]

            rings = []
            for index, start in enumerate(parts):
                end = parts[index + 1] if index + 1 < len(parts) else point_count
                rings.append(points[start:end])

            geometries.append(_rings_to_geometry(rings))

    return geometries


@lru_cache(maxsize=None)
def _read_shapefile(shp_path):
    shp_path = Path(shp_path)
    records = _read_dbf(shp_path.with_suffix(".dbf"))
    geometries = _read_shp_geometries(shp_path)

    if len(records) != len(geometries):
        raise ValueError(
            f"Shapefile record mismatch for {shp_path}: "
            f"{len(records)} attributes vs {len(geometries)} geometries"
        )

    return [
        {
            **record,
            "geometry": geometry,
        }
        for record, geometry in zip(records, geometries)
    ]


def _load_sa4_features():
    return _read_shapefile(str(SA4_SHP_PATH))


def _load_sa2_features():
    return _read_shapefile(str(SA2_SHP_PATH))


def _require_feature(features, code_field, code):
    code = str(code)
    matches = [feature for feature in features if feature[code_field] == code]
    if not matches:
        raise ValueError(f"No feature found for {code_field}={code}")
    return matches[0]


def _to_float(value):
    return float(value) if value not in {"", None} else None


def _normalise_attributes(attributes):
    return {key.lower(): value for key, value in attributes.items()}


def list_nsw_sa4s():
    rows = []
    for feature in _load_sa4_features():
        if feature["STE_NAME21"] != "New South Wales":
            continue
        if feature["geometry"].is_empty or not feature["AREASQKM21"]:
            continue

        rows.append(
            {
                "SA4_CODE": feature["SA4_CODE21"],
                "SA4_NAME": feature["SA4_NAME21"],
            }
        )

    return pd.DataFrame(rows).sort_values("SA4_NAME").reset_index(drop=True)


def get_bbox(geometry):
    if hasattr(geometry, "bounds"):
        return geometry.bounds

    points = []
    for ring in geometry["rings"]:
        for point in ring:
            points.append(point)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def get_sa4_geometry_by_code(sa4_code):
    feature = _require_feature(_load_sa4_features(), "SA4_CODE21", sa4_code)
    return (
        feature["geometry"],
        feature["SA4_CODE21"],
        feature["SA4_NAME21"],
    )


def get_sa2_bbox_df_by_code(sa4_code):
    sa4_geometry, real_code, real_name = get_sa4_geometry_by_code(sa4_code)
    sa2_features = [
        feature
        for feature in _load_sa2_features()
        if feature["SA4_CODE21"] == str(sa4_code)
        and feature["STE_NAME21"] == "New South Wales"
        and not feature["geometry"].is_empty
    ]

    rows = []
    failed_checks = []
    for feature in sa2_features:
        sa2_geometry = feature["geometry"]
        if not sa4_geometry.covers(sa2_geometry) and not sa4_geometry.buffer(
            1e-8
        ).covers(sa2_geometry):
            failed_checks.append(feature["SA2_CODE21"])

        xmin, ymin, xmax, ymax = sa2_geometry.bounds
        rows.append(
            {
                "sa4_code": real_code,
                "sa4_name": real_name,
                "sa2_main": feature["SA2_CODE21"],
                "sa2_name": feature["SA2_NAME21"],
                "state_name": feature["STE_NAME21"],
                "area_sqkm": _to_float(feature["AREASQKM21"]),
                "bbox_xmin": xmin,
                "bbox_ymin": ymin,
                "bbox_xmax": xmax,
                "bbox_ymax": ymax,
            }
        )

    if failed_checks:
        raise ValueError(
            f"SA2 geometry validation failed for SA4 {sa4_code}: "
            f"{', '.join(failed_checks)}"
        )

    return pd.DataFrame(rows).sort_values("sa2_main").reset_index(drop=True)


def get_poi_by_bbox(
    xmin,
    ymin,
    xmax,
    ymax,
    in_sr=SHAPEFILE_SPATIAL_REFERENCE,
    out_sr=POI_OUTPUT_SPATIAL_REFERENCE,
):
    url = f"{POI_URL}/0/query"
    base_params = {
        "where": "1=1",
        "outFields": "objectid,poiname,poitype,poigroup",
        "returnGeometry": "true",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": in_sr,
        "outSR": out_sr,
        "spatialRel": "esriSpatialRelIntersects",
        "orderByFields": "objectid",
        "f": "json",
    }

    rows = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            **base_params,
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        data = requests.get(url, params=params, timeout=60).json()
        if "error" in data:
            raise RuntimeError(data["error"])

        features = data.get("features", [])
        for feature in features:
            attributes = _normalise_attributes(feature["attributes"])
            rows.append(
                {
                    "objectid": attributes.get("objectid"),
                    "poiname": attributes.get("poiname"),
                    "poitype": attributes.get("poitype"),
                    "poigroup": attributes.get("poigroup"),
                    "longitude": feature["geometry"]["x"],
                    "latitude": feature["geometry"]["y"],
                }
            )

        if len(features) < page_size:
            break
        offset += page_size

    return pd.DataFrame(rows)


def get_sa4_poi_df_by_code(sa4_code):
    sa4_geometry, real_code, real_name = get_sa4_geometry_by_code(sa4_code)
    sa2_features = [
        feature
        for feature in _load_sa2_features()
        if feature["SA4_CODE21"] == str(sa4_code)
        and feature["STE_NAME21"] == "New South Wales"
        and not feature["geometry"].is_empty
    ]
    table_list = []

    for feature in sa2_features:
        sa2_geometry = feature["geometry"]
        if not sa4_geometry.covers(sa2_geometry) and not sa4_geometry.buffer(
            1e-8
        ).covers(sa2_geometry):
            raise ValueError(
                f"SA2 geometry validation failed for SA4 {sa4_code}: "
                f"{feature['SA2_CODE21']}"
            )

        xmin, ymin, xmax, ymax = sa2_geometry.bounds
        poi_df = get_poi_by_bbox(
            xmin,
            ymin,
            xmax,
            ymax,
        )
        if poi_df.empty:
            continue

        inside_sa2 = [
            sa2_geometry.covers(Point(row.longitude, row.latitude))
            for row in poi_df.itertuples(index=False)
        ]
        poi_df = poi_df.loc[inside_sa2].copy()
        if poi_df.empty:
            continue

        poi_df["sa4_code"] = real_code
        poi_df["sa4_name"] = real_name
        poi_df["sa2_main"] = feature["SA2_CODE21"]
        poi_df["sa2_name"] = feature["SA2_NAME21"]
        table_list.append(poi_df)

    if not table_list:
        return pd.DataFrame(
            columns=[
                "objectid",
                "poiname",
                "poitype",
                "poigroup",
                "longitude",
                "latitude",
                "sa4_code",
                "sa4_name",
                "sa2_main",
                "sa2_name",
            ]
        )

    return pd.concat(table_list, ignore_index=True)


def save_to_sqlite(sa2_bbox_df, poi_df, db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("pragma foreign_keys = on")
    conn.execute("drop table if exists poi_rtree")
    conn.execute("drop table if exists sa2_scores")
    conn.execute("drop table if exists poi")
    conn.execute("drop table if exists sa2_bbox")
    conn.execute(
        """
        create table sa2_bbox (
            area_name text,
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
            area_name text,
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
    create_database_indexes(conn)
    refresh_score_table(conn)
    conn.close()


def create_database_indexes(conn):
    conn.execute("create index if not exists idx_sa2_bbox_sa4 on sa2_bbox(sa4_code)")
    conn.execute("create index if not exists idx_poi_sa2 on poi(sa2_main)")
    conn.execute("create index if not exists idx_poi_sa4 on poi(sa4_code)")
    conn.execute("create index if not exists idx_poi_type on poi(poitype)")
    conn.execute(
        """
        create virtual table if not exists poi_rtree using rtree(
            poi_id,
            min_longitude,
            max_longitude,
            min_latitude,
            max_latitude
        )
        """
    )
    conn.execute("delete from poi_rtree")
    conn.execute(
        """
        insert into poi_rtree
        select poi_id, longitude, longitude, latitude, latitude
        from poi
        where longitude is not null and latitude is not null
        """
    )


def refresh_score_table(conn):
    conn.executescript(
        """
        drop table if exists sa2_scores;

        create table sa2_scores (
            area_name text,
            sa4_code text,
            sa4_name text,
            sa2_main text primary key,
            sa2_name text,
            area_sqkm real,
            poi_count integer,
            z_poi real,
            score real
        );

        insert into sa2_scores (
            area_name,
            sa4_code,
            sa4_name,
            sa2_main,
            sa2_name,
            area_sqkm,
            poi_count,
            z_poi,
            score
        )
        with counts as (
            select
                s.area_name,
                s.sa4_code,
                s.sa4_name,
                s.sa2_main,
                s.sa2_name,
                s.area_sqkm,
                count(p.poi_id) as poi_count
            from sa2_bbox s
            left join poi p
                on s.sa2_main = p.sa2_main
            group by
                s.area_name,
                s.sa4_code,
                s.sa4_name,
                s.sa2_main,
                s.sa2_name,
                s.area_sqkm
        ),
        stats as (
            select
                sa4_code,
                avg(poi_count * 1.0) as mean_poi,
                case
                    when count(*) > 1 then sqrt(
                        (sum(poi_count * poi_count * 1.0)
                        - sum(poi_count * 1.0) * sum(poi_count * 1.0) / count(*))
                        / (count(*) - 1)
                    )
                    else 0
                end as stddev_poi
            from counts
            group by sa4_code
        ),
        zscores as (
            select
                c.area_name,
                c.sa4_code,
                c.sa4_name,
                c.sa2_main,
                c.sa2_name,
                c.area_sqkm,
                c.poi_count,
                case
                    when s.stddev_poi is null or s.stddev_poi = 0 then 0
                    else (c.poi_count - s.mean_poi) / s.stddev_poi
                end as z_poi
            from counts c
            join stats s
                on c.sa4_code = s.sa4_code
        )
        select
            area_name,
            sa4_code,
            sa4_name,
            sa2_main,
            sa2_name,
            area_sqkm,
            poi_count,
            z_poi,
            1.0 / (1.0 + exp(-z_poi)) as score
        from zscores;

        create unique index if not exists idx_sa2_scores_sa2 on sa2_scores(sa2_main);
        create index if not exists idx_sa2_scores_sa4 on sa2_scores(sa4_code);
        create index if not exists idx_sa2_scores_score on sa2_scores(score);
        """
    )


def build_selected_areas_database(
    db_path=DEFAULT_DB_PATH,
    selected_areas=DEFAULT_SELECTED_AREAS,
):
    sa2_tables = []
    poi_tables = []

    for area_name, sa4_code in selected_areas:
        sa2_bbox_df = get_sa2_bbox_df_by_code(sa4_code)
        poi_df = get_sa4_poi_df_by_code(sa4_code)

        sa2_bbox_df["area_name"] = area_name
        poi_df["area_name"] = area_name

        sa2_tables.append(sa2_bbox_df)
        poi_tables.append(poi_df)

    all_sa2_bbox_df = pd.concat(sa2_tables, ignore_index=True)
    all_poi_df = pd.concat(poi_tables, ignore_index=True)

    all_sa2_bbox_df = all_sa2_bbox_df[
        [
            "area_name",
            "sa4_code",
            "sa4_name",
            "sa2_main",
            "sa2_name",
            "state_name",
            "area_sqkm",
            "bbox_xmin",
            "bbox_ymin",
            "bbox_xmax",
            "bbox_ymax",
        ]
    ]
    all_poi_df = all_poi_df[
        [
            "objectid",
            "poiname",
            "poitype",
            "poigroup",
            "longitude",
            "latitude",
            "area_name",
            "sa4_code",
            "sa4_name",
            "sa2_main",
            "sa2_name",
        ]
    ]

    save_to_sqlite(all_sa2_bbox_df, all_poi_df, str(db_path))
    return all_sa2_bbox_df, all_poi_df


def read_sql(sql, db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


if __name__ == "__main__":
    sa2_bbox_df, poi_df = build_selected_areas_database()
    print(
        f"Saved {len(sa2_bbox_df)} SA2 rows and {len(poi_df)} "
        f"POI rows to {DEFAULT_DB_PATH}"
    )

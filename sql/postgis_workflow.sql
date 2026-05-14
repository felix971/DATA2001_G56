-- PostGIS reference workflow for the Greater Sydney scoring task.
-- This mirrors the executable SQLite/RTree workflow in task2_sa2.py and
-- task4_analysis.py. It requires a PostgreSQL database with PostGIS installed.

create extension if not exists postgis;

-- Expected geometry tables after loading ABS ASGS 2021 shapefiles:
--   sa2_boundaries(sa2_main text primary key, sa2_name text, sa4_code text,
--                  sa4_name text, state_name text, area_sqkm numeric,
--                  geom geometry(MultiPolygon, 7844))
--   poi_raw(objectid integer, poiname text, poitype text, poigroup integer,
--           longitude numeric, latitude numeric)

create index if not exists idx_sa2_boundaries_geom
    on sa2_boundaries using gist (geom);

create index if not exists idx_sa2_boundaries_sa4
    on sa2_boundaries (sa4_code);

drop table if exists poi_spatial;
create table poi_spatial as
select
    row_number() over () as poi_id,
    p.objectid,
    p.poiname,
    p.poitype,
    p.poigroup,
    p.longitude,
    p.latitude,
    s.sa4_code,
    s.sa4_name,
    s.sa2_main,
    s.sa2_name,
    st_setsrid(st_makepoint(p.longitude, p.latitude), 7844) as geom
from poi_raw p
join sa2_boundaries s
    on st_covers(s.geom, st_setsrid(st_makepoint(p.longitude, p.latitude), 7844));

alter table poi_spatial add primary key (poi_id);

create index if not exists idx_poi_spatial_geom
    on poi_spatial using gist (geom);

create index if not exists idx_poi_spatial_sa2
    on poi_spatial (sa2_main);

drop materialized view if exists sa2_scores_postgis;
create materialized view sa2_scores_postgis as
with counts as (
    select
        s.sa4_code,
        s.sa4_name,
        s.sa2_main,
        s.sa2_name,
        s.area_sqkm,
        count(p.poi_id) as poi_count
    from sa2_boundaries s
    left join poi_spatial p
        on s.sa2_main = p.sa2_main
    where s.sa4_code in ('117', '118', '120', '125')
    group by
        s.sa4_code,
        s.sa4_name,
        s.sa2_main,
        s.sa2_name,
        s.area_sqkm
),
stats as (
    select
        *,
        avg(poi_count) over (partition by sa4_code) as mean_poi,
        stddev_samp(poi_count) over (partition by sa4_code) as stddev_poi
    from counts
),
zscores as (
    select
        *,
        case
            when stddev_poi is null or stddev_poi = 0 then 0
            else (poi_count - mean_poi) / stddev_poi
        end as z_poi
    from stats
)
select
    sa4_code,
    sa4_name,
    sa2_main,
    sa2_name,
    area_sqkm,
    poi_count,
    z_poi,
    1.0 / (1.0 + exp(-z_poi)) as score
from zscores;

create unique index if not exists idx_sa2_scores_postgis_sa2
    on sa2_scores_postgis (sa2_main);

-- Optional correlation once a table such as
-- sa2_income(sa2_main text primary key, median_total_income_2022 integer)
-- has been loaded.
select
    count(*) as n,
    corr(s.score, i.median_total_income_2022) as pearson_r
from sa2_scores_postgis s
join sa2_income i
    on s.sa2_main = i.sa2_main
where i.median_total_income_2022 is not null;

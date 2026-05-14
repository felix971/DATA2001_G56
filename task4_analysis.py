import json
import math
import os
import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import Normalize
from shapely.geometry import MultiPolygon, Polygon

import task2_sa2


DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DB_PATH = DATA_DIR / "task2_poi.db"
INCOME_JSON_PATH = DATA_DIR / "sa2_income_dbr_nov25.json"
INCOME_CSV_PATH = DATA_DIR / "sa2_income_median_2022.csv"
ALL_SCORES_CSV_PATH = DATA_DIR / "task3_scores_all_sa2.csv"


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def build_income_csv():
    with INCOME_JSON_PATH.open() as source:
        payload = json.load(source)

    rows = []
    for feature in payload.get("features", []):
        attributes = feature["attributes"]
        rows.append(
            {
                "sa2_main": attributes["sa2_code_2021"],
                "sa2_name": attributes["sa2_name_2021"],
                "sa4_code": attributes["sa4_code_2021"],
                "sa4_name": attributes["sa4_name_2021"],
                "median_total_income_2022": attributes["income_172022"],
            }
        )

    income_df = pd.DataFrame(rows).sort_values(["sa4_code", "sa2_main"])
    income_df.to_csv(INCOME_CSV_PATH, index=False)
    return income_df


def _load_income_table(conn, income_df):
    conn.execute("drop table if exists sa2_income")
    conn.execute(
        """
        create table sa2_income (
            sa2_main text primary key,
            sa2_name text,
            sa4_code text,
            sa4_name text,
            median_total_income_2022 integer
        )
        """
    )
    income_df.to_sql("sa2_income", conn, if_exists="append", index=False)
    conn.execute("create index if not exists idx_sa2_income_sa4 on sa2_income(sa4_code)")
    conn.execute(
        """
        create index if not exists idx_sa2_income_median
        on sa2_income(median_total_income_2022)
        """
    )


def _build_score_income_tables(conn):
    conn.executescript(
        """
        drop table if exists score_income_analysis;
        drop table if exists score_income_correlation;

        create table score_income_analysis (
            area_name text,
            sa4_code text,
            sa4_name text,
            sa2_main text primary key,
            sa2_name text,
            area_sqkm real,
            poi_count integer,
            z_poi real,
            score real,
            median_total_income_2022 integer
        );

        insert into score_income_analysis (
            area_name,
            sa4_code,
            sa4_name,
            sa2_main,
            sa2_name,
            area_sqkm,
            poi_count,
            z_poi,
            score,
            median_total_income_2022
        )
        select
            s.area_name,
            s.sa4_code,
            s.sa4_name,
            s.sa2_main,
            s.sa2_name,
            s.area_sqkm,
            s.poi_count,
            s.z_poi,
            s.score,
            i.median_total_income_2022
        from sa2_scores s
        join sa2_income i
            on s.sa2_main = i.sa2_main
        where i.median_total_income_2022 is not null;

        create unique index if not exists idx_score_income_sa2
            on score_income_analysis(sa2_main);
        create index if not exists idx_score_income_sa4
            on score_income_analysis(sa4_code);

        create table score_income_correlation (
            correlation_id text primary key,
            n integer,
            pearson_r real
        );

        insert into score_income_correlation (
            correlation_id,
            n,
            pearson_r
        )
        with pairs as (
            select
                score as x,
                median_total_income_2022 * 1.0 as y
            from score_income_analysis
            where score is not null
                and median_total_income_2022 is not null
        ),
        agg as (
            select
                count(*) as n,
                sum(x) as sx,
                sum(y) as sy,
                sum(x * y) as sxy,
                sum(x * x) as sx2,
                sum(y * y) as sy2
            from pairs
        )
        select
            'overall' as correlation_id,
            n,
            (n * sxy - sx * sy)
                / sqrt((n * sx2 - sx * sx) * (n * sy2 - sy * sy)) as pearson_r
        from agg;
        """
    )


def _pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = math.sqrt(float((x_centered**2).sum() * (y_centered**2).sum()))
    if denominator == 0:
        return np.nan
    return float((x_centered * y_centered).sum() / denominator)


def _permutation_p_value(x, y, observed_r, permutations=10000, seed=56):
    rng = np.random.default_rng(seed)
    extreme = 0
    y = np.asarray(y, dtype=float)
    for _ in range(permutations):
        permuted_r = _pearson_r(x, rng.permutation(y))
        if abs(permuted_r) >= abs(observed_r):
            extreme += 1
    return (extreme + 1) / (permutations + 1)


def prepare_database_outputs():
    income_df = build_income_csv()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("pragma foreign_keys = on")
    task2_sa2.create_database_indexes(conn)
    task2_sa2.refresh_score_table(conn)
    _load_income_table(conn, income_df)
    _build_score_income_tables(conn)

    scores_df = pd.read_sql_query(
        "select * from sa2_scores order by sa4_code, score desc",
        conn,
    )
    score_income_df = pd.read_sql_query(
        "select * from score_income_analysis order by sa4_code, score desc",
        conn,
    )
    correlation_df = pd.read_sql_query(
        "select * from score_income_correlation",
        conn,
    )
    index_df = pd.read_sql_query(
        """
        select name, tbl_name, sql
        from sqlite_master
        where type in ('index', 'table')
            and name not like 'sqlite_%'
        order by type, name
        """,
        conn,
    )
    conn.close()

    observed_r = _pearson_r(
        score_income_df["score"],
        score_income_df["median_total_income_2022"],
    )
    p_value = _permutation_p_value(
        score_income_df["score"],
        score_income_df["median_total_income_2022"],
        observed_r,
    )
    correlation_df["permutation_p_value"] = p_value
    correlation_df["method"] = "Pearson correlation with two-sided permutation test"
    correlation_df.to_csv(OUTPUT_DIR / "score_income_correlation.csv", index=False)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("alter table score_income_correlation add column permutation_p_value real")
        conn.execute("alter table score_income_correlation add column method text")
        conn.execute(
            """
            update score_income_correlation
            set permutation_p_value = ?,
                method = ?
            """,
            (p_value, "Pearson correlation with two-sided permutation test"),
        )

    scores_df.to_csv(ALL_SCORES_CSV_PATH, index=False)
    score_income_df.to_csv(OUTPUT_DIR / "score_income_analysis.csv", index=False)
    index_df.to_csv(OUTPUT_DIR / "database_indexes.csv", index=False)

    for sa4_code, group in scores_df.groupby("sa4_code"):
        group.sort_values("score", ascending=False).to_csv(
            DATA_DIR / f"task3_scores_SA4_{sa4_code}.csv",
            index=False,
        )

    sa4_summary_df = (
        scores_df.groupby(["sa4_code", "sa4_name"])
        .agg(
            sa2_count=("sa2_main", "count"),
            total_poi=("poi_count", "sum"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reset_index()
        .sort_values("sa4_code")
    )
    sa4_summary_df.to_csv(OUTPUT_DIR / "sa4_score_summary.csv", index=False)

    top_bottom_df = pd.concat(
        [
            scores_df.nlargest(10, "score").assign(rank_group="top_10"),
            scores_df.nsmallest(10, "score").assign(rank_group="bottom_10"),
        ],
        ignore_index=True,
    )
    top_bottom_df.to_csv(OUTPUT_DIR / "top_bottom_sa2_scores.csv", index=False)

    return scores_df, score_income_df, correlation_df, sa4_summary_df


def _iter_polygons(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms


def plot_score_distribution(scores_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores_df["score"], bins=16, color="#4C78A8", edgecolor="white")
    ax.axvline(scores_df["score"].median(), color="#F58518", linewidth=2)
    ax.set_title("Distribution of SA2 Well-Resourced Scores")
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of SA2s")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "score_distribution.png", dpi=200)
    plt.close(fig)


def plot_score_boxplot(scores_df):
    grouped = [
        group["score"].to_numpy()
        for _, group in scores_df.sort_values("sa4_code").groupby("sa4_name")
    ]
    labels = [
        name.replace("Sydney - ", "")
        for name, _ in scores_df.sort_values("sa4_code").groupby("sa4_name")
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(grouped, tick_labels=labels, patch_artist=True)
    for patch, color in zip(ax.artists, ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_title("Score Spread by SA4")
    ax.set_xlabel("SA4")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "score_boxplot_by_sa4.png", dpi=200)
    plt.close(fig)


def plot_score_income_scatter(score_income_df, correlation_df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sa4_names = sorted(score_income_df["sa4_name"].unique())
    colors = dict(zip(sa4_names, ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]))

    for sa4_name, group in score_income_df.groupby("sa4_name"):
        ax.scatter(
            group["median_total_income_2022"],
            group["score"],
            label=sa4_name.replace("Sydney - ", ""),
            alpha=0.75,
            s=42,
            color=colors[sa4_name],
        )

    x = score_income_df["median_total_income_2022"].to_numpy(dtype=float)
    y = score_income_df["score"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#222222", linewidth=1.5)

    r = correlation_df.loc[0, "pearson_r"]
    p = correlation_df.loc[0, "permutation_p_value"]
    ax.set_title(f"Score vs Median Total Income, r={r:.3f}, p={p:.3f}")
    ax.set_xlabel("Median total income, 2022 ($)")
    ax.set_ylabel("Well-resourced score")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "score_income_correlation.png", dpi=200)
    plt.close(fig)


def plot_poi_type_bar():
    conn = sqlite3.connect(DB_PATH)
    poi_type_df = pd.read_sql_query(
        """
        select poitype, count(*) as poi_count
        from poi
        group by poitype
        order by poi_count desc
        limit 15
        """,
        conn,
    )
    conn.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ordered = poi_type_df.sort_values("poi_count")
    ax.barh(ordered["poitype"], ordered["poi_count"], color="#72B7B2")
    ax.set_title("Top POI Types in Selected SA4s")
    ax.set_xlabel("POI count")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_poi_types.png", dpi=200)
    plt.close(fig)


def plot_score_map(scores_df):
    score_lookup = scores_df.set_index("sa2_main")["score"].to_dict()
    name_lookup = scores_df.set_index("sa2_main")["sa2_name"].to_dict()
    features = [
        feature
        for feature in task2_sa2._load_sa2_features()
        if feature["SA2_CODE21"] in score_lookup
    ]

    norm = Normalize(vmin=scores_df["score"].min(), vmax=scores_df["score"].max())
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9, 8))

    for feature in features:
        score = score_lookup[feature["SA2_CODE21"]]
        color = cmap(norm(score))
        for polygon in _iter_polygons(feature["geometry"]):
            x, y = polygon.exterior.xy
            ax.fill(x, y, facecolor=color, edgecolor="#333333", linewidth=0.35)

    top_labels = scores_df.nlargest(5, "score")["sa2_main"].tolist()
    for feature in features:
        sa2_code = feature["SA2_CODE21"]
        if sa2_code not in top_labels:
            continue
        point = feature["geometry"].representative_point()
        ax.text(
            point.x,
            point.y,
            name_lookup[sa2_code].split(" - ")[0],
            fontsize=6,
            ha="center",
            va="center",
            color="#111111",
        )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.75)
    colorbar.set_label("Well-resourced score")
    ax.set_title("SA2 Score Map Overlay")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "score_map_overlay.png", dpi=220)
    plt.close(fig)


def plot_database_schema():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")

    tables = {
        "sa2_bbox": (0.05, 0.52, "PK sa2_main\nsa4_code, sa4_name\narea_sqkm, bbox"),
        "poi": (0.39, 0.52, "PK poi_id\nFK sa2_main\npoitype, lon/lat"),
        "sa2_scores": (0.72, 0.52, "PK-like sa2_main\npoi_count\nz_poi, score"),
        "sa2_income": (0.05, 0.1, "PK sa2_main\nmedian_total_income_2022"),
        "score_income_analysis": (0.49, 0.1, "score + income\nused for correlation"),
    }

    for name, (x, y, body) in tables.items():
        box = patches.FancyBboxPatch(
            (x, y),
            0.23,
            0.26,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor="#F5F5F5",
        )
        ax.add_patch(box)
        ax.text(x + 0.015, y + 0.2, name, weight="bold", fontsize=10)
        ax.text(x + 0.015, y + 0.055, body, fontsize=8, va="bottom")

    arrows = [
        ((0.28, 0.65), (0.39, 0.65)),
        ((0.62, 0.65), (0.72, 0.65)),
        ((0.165, 0.52), (0.165, 0.36)),
        ((0.28, 0.23), (0.49, 0.23)),
        ((0.835, 0.52), (0.72, 0.36)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "#333333"},
        )

    ax.text(0.04, 0.93, "Database Schema Used for Scoring and Analysis", fontsize=14, weight="bold")
    ax.text(
        0.04,
        0.87,
        "Indexes include SA2/SA4 lookup indexes plus an RTree spatial index on POI longitude/latitude.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "database_schema.png", dpi=200)
    plt.close(fig)


def generate_visualisations(scores_df, score_income_df, correlation_df):
    plot_score_distribution(scores_df)
    plot_score_boxplot(scores_df)
    plot_score_map(scores_df)
    plot_score_income_scatter(score_income_df, correlation_df)
    plot_poi_type_bar()
    plot_database_schema()


def main():
    _ensure_output_dir()
    scores_df, score_income_df, correlation_df, sa4_summary_df = prepare_database_outputs()
    generate_visualisations(scores_df, score_income_df, correlation_df)

    print("Generated Task 4 analysis outputs")
    print(f"Scores: {len(scores_df)} SA2 rows")
    print(f"Income matches: {len(score_income_df)} SA2 rows")
    print(sa4_summary_df.to_string(index=False))
    print(correlation_df.to_string(index=False))


if __name__ == "__main__":
    main()

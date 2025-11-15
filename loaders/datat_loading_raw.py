import pandas as pd

# Summary statistics for raw dataset

# Load raw dataset
raw_df = pd.read_csv("dataset.csv")

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)


def top_artists(df, n=10):
    # artists are semicolon-separated; return top individual artists
    if "artists" not in df.columns:
        return pd.Series(dtype=int)
    return (
        df["artists"]
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
        .value_counts()
        .head(n)
    )


def summarize(df, name):
    print(f"===== Summary for {name} =====")
    print("Shape:", df.shape)
    print("Columns:", len(df.columns))
    print("\nDtype counts:")
    print(df.dtypes.value_counts())
    print("\nMissing values (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))
    print(
        "\nDuplicate track_name count:",
        (
            df.duplicated(subset="track_name").sum()
            if "track_name" in df.columns
            else "N/A"
        ),
    )
    print(
        "Unique track_name:",
        df["track_name"].nunique() if "track_name" in df.columns else "N/A",
    )
    print("\nTop 10 track_genre (if present):")
    if "track_genre" in df.columns:
        print(df["track_genre"].value_counts().head(10))
    else:
        print("N/A")
    print("\nTop 10 artists:")
    print(top_artists(df, 10))
    print("\nNumeric describe (selected):")
    print(
        df.select_dtypes(include="number")
        .describe()
        .transpose()
        .loc[:, ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    )
    print("\nSample rows (5):")
    # display(df.head(5))
    print("\n")


# Run summaries
summarize(raw_df, "dataset.csv (raw)")


# # Differences between raw and clean
# print("===== Differences between raw and clean =====")
# print("Raw rows:", raw_df.shape[0])
# print("Clean rows:", tracks_df.shape[0])
# print("Rows removed (raw - clean):", raw_df.shape[0] - tracks_df.shape[0])

# # Which columns differ or were added/removed
# raw_cols = set(raw_df.columns)
# clean_cols = set(tracks_df.columns)
# print("Columns only in raw:", sorted(raw_cols - clean_cols))
# print("Columns only in clean:", sorted(clean_cols - raw_cols))
# print("Shared columns:", sorted(raw_cols & clean_cols))

# # Quick CSV-level summary table
# summary_table = pd.DataFrame(
#     {
#         "dataset": ["raw", "clean"],
#         "rows": [raw_df.shape[0], tracks_df.shape[0]],
#         "cols": [raw_df.shape[1], tracks_df.shape[1]],
#         "missing_total": [raw_df.isna().sum().sum(), tracks_df.isna().sum().sum()],
#         "unique_track_names": [
#             raw_df["track_name"].nunique() if "track_name" in raw_df.columns else None,
#             (
#                 tracks_df["track_name"].nunique()
#                 if "track_name" in tracks_df.columns
#                 else None
#             ),
#         ],
#         "duplicate_track_names": [
#             (
#                 raw_df.duplicated(subset="track_name").sum()
#                 if "track_name" in raw_df.columns
#                 else None
#             ),
#             (
#                 tracks_df.duplicated(subset="track_name").sum()
#                 if "track_name" in tracks_df.columns
#                 else None
#             ),
#         ],
#     }
# )
# # display(summary_table)

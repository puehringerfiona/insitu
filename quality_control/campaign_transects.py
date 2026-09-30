from __future__ import annotations

from pathlib import Path
import re


INSITU_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")
METADATA_PATTERN = "*_metadata_with_Rrs_conv_*.csv"

CAMPAIGN_TRANSECTS: dict[str, list[int]] = {
    "20230610_CST": [2],
    "20231008_ZRH": [1, 2, 3, 4],
    "20240618_BIE": [1],
    "20240619_WAL": [2, 3, 4],
    "20250303_CST": [1],
    "20260226_ZRH": [1, 2, 3, 4],
    "20260227_ZRH": [1, 3],
    "20260423_ZRH": [1, 2, 3, 4],
    "20260430_ZRH": [1, 2, 3, 5],
}

CAMPAIGN_RE = re.compile(r"^(?P<campaign>\d{8}_[A-Za-z0-9]+)_metadata_with_Rrs_conv_.*\.csv$")


def find_convolved_metadata_files(insitu_root: Path = INSITU_ROOT) -> dict[str, list[Path]]:
    """Return matching convolved metadata files grouped by campaign name."""
    files_by_campaign: dict[str, list[Path]] = {}

    for path in sorted(insitu_root.rglob(METADATA_PATTERN)):
        match = CAMPAIGN_RE.match(path.name)
        if match is None:
            continue

        campaign = match.group("campaign")
        if campaign not in CAMPAIGN_TRANSECTS:
            continue

        files_by_campaign.setdefault(campaign, []).append(path)

    return files_by_campaign


def get_campaign_transects(
    campaign: str | None = None,
    insitu_root: Path = INSITU_ROOT,
    require_metadata: bool = True,
) -> dict[str, list[int]] | list[int]:
    """
    Return configured transect numbers for one campaign or all campaigns.

    By default, campaigns are returned only when matching
    ``*_metadata_with_Rrs_conv_*.csv`` files exist below ``insitu_root``.
    Set ``require_metadata=False`` to return the static configuration without
    checking files on disk.
    """
    if campaign is not None:
        if campaign not in CAMPAIGN_TRANSECTS:
            raise KeyError(f"No transect selection configured for campaign: {campaign}")

        if require_metadata and campaign not in find_convolved_metadata_files(insitu_root):
            raise FileNotFoundError(
                f"No {METADATA_PATTERN} files found below {insitu_root} for campaign {campaign}"
            )

        return CAMPAIGN_TRANSECTS[campaign].copy()

    transects = {key: value.copy() for key, value in CAMPAIGN_TRANSECTS.items()}
    if not require_metadata:
        return transects

    files_by_campaign = find_convolved_metadata_files(insitu_root)
    return {campaign_name: transects[campaign_name] for campaign_name in transects if campaign_name in files_by_campaign}


def get_missing_campaign_metadata(insitu_root: Path = INSITU_ROOT) -> list[str]:
    """Return configured campaigns without matching convolved metadata files."""
    files_by_campaign = find_convolved_metadata_files(insitu_root)
    return [campaign for campaign in CAMPAIGN_TRANSECTS if campaign not in files_by_campaign]

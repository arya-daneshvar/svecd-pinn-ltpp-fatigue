"""Create the publication figure of mechanically selected PINN trajectories."""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PREDICTIONS = HERE / "final_validation_predictions.csv"
COMPARISON = (
    HERE.parent
    / "outputs/outputs_severity_loss_sweep/full_validation_comparison.csv"
)
SEED_SUMMARY = (
    HERE.parent
    / "outputs/outputs_robustness_seed_stability/seed_stability_summary.csv"
)

PNG_OUT = HERE / "representative_frozen_pinn_trajectories.png"
PDF_OUT = HERE / "representative_frozen_pinn_trajectories.pdf"
SELECTION_OUT = HERE / "representative_trajectory_selection.csv"


def verify_operating_point() -> None:
    """Verify the saved comparison/robustness tables without using them as trajectories."""
    comparison = pd.read_csv(COMPARISON)
    op = comparison.loc[comparison["model"].eq("high_only_full")]
    assert set(op["eval"]) == {
        "temporal", "cold_unseen", "first_anchor", "rolling_anchor"
    }, "The frozen high_only_full operating point is incomplete."
    assert op[["band_w", "w_total", "w_onset", "w_cracked", "huber_delta"]].notna().all().all()

    seed_summary = pd.read_csv(SEED_SUMMARY)
    seed_rows = seed_summary.loc[seed_summary["model"].eq("high_only")]
    assert not seed_rows.empty, "No separate high_only seed-stability summary found."
    assert any(c.endswith("_std") for c in seed_rows.columns)


def select_sections(first_anchor: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply the predeclared percentile-nearest selection rule exactly."""
    counts = first_anchor.groupby("SECTION_ID").size().rename("forecast_row_count")
    terminal_rows = (
        first_anchor.sort_values(["SECTION_ID", "AGE", "row_id"], kind="mergesort")
        .groupby("SECTION_ID", as_index=False)
        .tail(1)
        .set_index("SECTION_ID")
    )
    eligible = pd.concat(
        [counts, terminal_rows["y_total"].rename("observed_terminal_value")], axis=1
    )
    eligible = eligible.loc[
        eligible["forecast_row_count"].ge(3)
        & eligible["observed_terminal_value"].gt(0)
    ].copy()
    eligibility_count = len(eligible)

    quantiles = [("Low / Q25", 0.25), ("Median / Q50", 0.50), ("High / Q75", 0.75)]
    used: set[str] = set()
    selected = []
    for label, q in quantiles:
        q_value = float(eligible["observed_terminal_value"].quantile(q))
        candidates = eligible.loc[~eligible.index.isin(used)].copy()
        candidates["distance"] = (candidates["observed_terminal_value"] - q_value).abs()
        choice = (
            candidates.reset_index()
            .sort_values(
                ["distance", "forecast_row_count", "SECTION_ID"],
                ascending=[True, False, True],
                kind="mergesort",
            )
            .iloc[0]
        )
        used.add(str(choice["SECTION_ID"]))
        selected.append(
            {
                "SECTION_ID": choice["SECTION_ID"],
                "target_quantile": q,
                "quantile_terminal_value": q_value,
                "observed_terminal_value": float(choice["observed_terminal_value"]),
                "forecast_row_count": int(choice["forecast_row_count"]),
                "eligibility_count": eligibility_count,
                "panel_label": label,
            }
        )
    return pd.DataFrame(selected), eligibility_count


def prepend_anchor(frame: pd.DataFrame, anchor_age: float, anchor_total: float) -> tuple[np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["AGE", "row_id"], kind="mergesort")
    return (
        np.r_[anchor_age, ordered["AGE"].to_numpy(float)],
        np.r_[anchor_total, ordered["p_total"].to_numpy(float)],
    )


def main() -> None:
    verify_operating_point()
    predictions = pd.read_csv(PREDICTIONS, dtype={"SECTION_ID": str})
    trajectories = predictions.loc[
        predictions["model"].eq("Frozen pure PINN")
        & predictions["protocol"].isin(["first_anchor", "rolling_anchor"])
    ].copy()
    first = trajectories.loc[trajectories["protocol"].eq("first_anchor")].copy()
    rolling = trajectories.loc[trajectories["protocol"].eq("rolling_anchor")].copy()

    selected, eligibility_count = select_sections(first)
    assert eligibility_count == 199
    assert selected["SECTION_ID"].tolist() == ["48_0164", "40_0123", "30_0113"]
    selected.drop(columns="panel_label").to_csv(SELECTION_OUT, index=False, float_format="%.9g")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    blue, orange = "#0072B2", "#D55E00"  # Okabe-Ito colorblind-safe palette
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.75), sharey=True)

    legend_handles = None
    for ax, row in zip(axes, selected.itertuples(index=False)):
        sid = row.SECTION_ID
        fa = first.loc[first["SECTION_ID"].eq(sid)].sort_values(["AGE", "row_id"])
        ra = rolling.loc[rolling["SECTION_ID"].eq(sid)].sort_values(["AGE", "row_id"])
        anchor_age = float(fa.iloc[0]["anchor_age"])
        anchor_total = float(fa.iloc[0]["anchor_total"])

        obs_age = np.r_[anchor_age, fa["AGE"].to_numpy(float)]
        obs_total = np.r_[anchor_total, fa["y_total"].to_numpy(float)]
        fa_age, fa_pred = prepend_anchor(fa, anchor_age, anchor_total)
        ra_age, ra_pred = prepend_anchor(ra, anchor_age, anchor_total)

        h_obs, = ax.plot(obs_age, obs_total, "-o", color="black", ms=3.4,
                         mfc="black", mec="black", mew=0.7, label="Observed")
        h_fa, = ax.plot(fa_age, fa_pred, "--s", color=blue, ms=3.5,
                        mfc="white", mec=blue, mew=0.9, label="First-anchor PINN")
        h_ra, = ax.plot(ra_age, ra_pred, ":^", color=orange, ms=3.8,
                        mfc="white", mec=orange, mew=0.9, label="Rolling-anchor PINN")
        legend_handles = [h_obs, h_fa, h_ra]

        ax.set_title(
            f"{row.panel_label}\n{sid}; terminal = {row.observed_terminal_value:.2f}%",
            pad=4,
        )
        ax.set_xlabel("Pavement age (years)")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=2.5, width=0.6)
        ax.margins(x=0.04)

    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Total fatigue cracking (% lane area)")
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.19, top=0.73, wspace=0.15)
    fig.savefig(PNG_OUT, dpi=600)
    fig.savefig(PDF_OUT)
    plt.close(fig)

    print(selected.drop(columns="panel_label").to_string(index=False))
    print(f"Verified operating point: frozen high_only_full; uncertainty: separate seed summary")
    print(f"Saved: {PNG_OUT}\nSaved: {PDF_OUT}\nSaved: {SELECTION_OUT}")


if __name__ == "__main__":
    main()

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # When belief propagation becomes Gaussian

    This notebook explains the central claim of arXiv:2601.21935 using the
    already-captured reproduction evidence. It does **not** rerun the
    expensive or formal experiment.

    ![Observed topology result](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/master/reports/bp-gaussian/images/headline_topologies.png)

    The strongest observation is the steep drop in KL divergence as a
    non-Gaussian belief travels farther from its prior. The chain falls
    from **0.6117** to **0.0161** over six hops. The qualification is equally
    important: it reaches the paper's `0.02` threshold at hop 5, not within
    3 hops as reported in Figure 4a.
    """)
    return


@app.cell
def _():
    chain = [0.6117, 0.1743, 0.0667, 0.0327, 0.0212, 0.0173, 0.0161]
    loopy = [0.5940, 0.1494, 0.0555, 0.0251, 0.0189]
    star = {2: 0.3941, 4: 0.7203, 6: 0.9234}
    return chain, star


@app.cell
def _(mo):
    threshold = mo.ui.slider(
        start=0.01,
        stop=0.10,
        step=0.005,
        value=0.02,
        label="Gaussian KL threshold",
        show_value=True,
    )
    threshold
    return (threshold,)


@app.cell
def _(chain, mo, threshold):
    crossings = [hop for hop, value in enumerate(chain) if value < threshold.value]
    first_crossing = crossings[0] if crossings else None
    crossing_text = (
        f"The chain first crosses this threshold at **hop {first_crossing}**."
        if first_crossing is not None
        else "The chain never crosses this threshold in the observed six hops."
    )
    mo.md(
        f"""
        ## Explore the threshold

        At a threshold of **{threshold.value:.3f}**, {crossing_text}

        This interaction changes only the interpretation of embedded evidence;
        it does not recompute BP or alter the formal result.
        """
    )
    return


@app.cell
def _(mo, star):
    rows = "\n".join(f"| {degree} | {value:.4f} |" for degree, value in star.items())
    mo.md(
        f"""
        ## Why path depth matters more than degree

        A star graph adds incoming messages at the center but no repeated
        convolution along a path. Its center becomes *less* Gaussian as degree
        grows:

        | Degree | Center KL |
        |---:|---:|
        {rows}

        This negative control supports the paper's proposed mechanism: the
        smoothing comes from convolutional depth, not neighbor count alone.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What the six checks establish

    | Claim | Evidence in this reproduction | Assessment |
    |---|---|---|
    | Chain convergence | KL `0.6117 → 0.0161` | Aligned |
    | Tree convergence | Same decay on a symmetric representative path | Partially aligned |
    | Loopy convergence | KL `0.5940 → 0.0189` | Aligned for one small graph |
    | Exclusion zone | Alternative-prior far KL `0.0880` vs `0.0161` | Quantitative boundary inconclusive |
    | Three-hop + degree | Hop-3 KL `0.0327`; degree effect increases | Partially aligned |
    | Stereo pattern | Synthetic smooth `0.0173`, edge `0.6117` | Proxy only |

    The formal verifier reports six PASS labels because all programmed
    conditions succeed. Scientific assessments are stricter because they
    account for the 90-bin grid, deterministic prior, simplified tree, and
    absence of the Middlebury Cones workload.

    ## Reproduce the formal run

    The immutable experiment used Hugging Face `cpu-upgrade` for 26 seconds:

    ```bash
    python -m venv .venv && . .venv/bin/activate && python -m pip install --disable-pip-version-check numpy==2.3.2 && python repro/src/verify_bp.py
    ```

    See the [illustrated report](https://github.com/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/blob/master/reports/bp-gaussian/report.md)
    for implementation details and the full substitution audit.
    """)
    return


if __name__ == "__main__":
    app.run()

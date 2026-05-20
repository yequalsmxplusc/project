## What to add

### 1) Validation preprocessing block

Add this in the section where you load experimental battery data, right before `loadncadata()` or right after the raw file import cell. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_35dbf7ed-c19d-4ecb-b3d5-a6b9157f856a/467820f3-0f09-416f-833b-eebcbabe88d1/vertopal.com_CHT.md)

```python
import pandas as pd
import numpy as np

def prepare_validation_data(df):
    d = df.copy()

    rename_map = {
        "Date_Time": "Date_Time",
        "Test_Time(s)": "TestTimes",
        "Step_Time(s)": "StepTimes",
        "Step_Index": "StepIndex",
        "Voltage(V)": "VoltageV",
        "Current(A)": "CurrentA",
        "Surface_Temp(degC)": "SurfaceTempdegC",
    }
    d = d.rename(columns=rename_map)

    d["Date_Time"] = pd.to_datetime(d["Date_Time"], errors="coerce")
    d = d.sort_values("TestTimes").reset_index(drop=True)

    d["TsurfaceK"] = d["SurfaceTempdegC"] + 273.15
    d["deltaTK"] = d["TsurfaceK"] - 298.15
    d["SoCproxy"] = (d["VoltageV"] - 2.5) / (4.2 - 2.5)
    d["powerW"] = d["VoltageV"] * d["CurrentA"].abs()
    d["dVdt"] = d["VoltageV"].diff().fillna(0.0)
    d["dTdt"] = d["SurfaceTempdegC"].diff().fillna(0.0)

    d["heatgenproxy"] = d["powerW"]
    d["window_mean_temp_C"] = d["SurfaceTempdegC"].rolling(5, min_periods=1).mean()
    d["window_max_temp_C"] = d["SurfaceTempdegC"].cummax()
    d["temp_rise_from_start_C"] = d["SurfaceTempdegC"] - d["SurfaceTempdegC"].iloc[0]

    return d
```

### 2) Validation comparison block

Add this inside the NCA / experimental validation module, after your `loadncadata()` function and before plotting or scoring. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_35dbf7ed-c19d-4ecb-b3d5-a6b9157f856a/467820f3-0f09-416f-833b-eebcbabe88d1/vertopal.com_CHT.md)

```python
def prepare_nca_validation(df):
    d = df.copy()
    d = d.sort_values("TestTimes").reset_index(drop=True)
    d["TsurfaceK"] = d["SurfaceTempdegC"] + 273.15
    d["deltaTK"] = d["TsurfaceK"] - 298.15
    d["SoCproxy"] = (d["VoltageV"] - 2.5) / (4.2 - 2.5)
    d["powerW"] = d["VoltageV"] * d["CurrentA"].abs()
    d["dVdt"] = d["VoltageV"].diff().fillna(0.0)
    d["dTdt"] = d["SurfaceTempdegC"].diff().fillna(0.0)
    return d
```

### 3) Safety check block

Add this near your results summary section, after predictions are available. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_35dbf7ed-c19d-4ecb-b3d5-a6b9157f856a/4a055706-cb5a-4f15-bbfa-433f70a0ff4d/README.md)

```python
def pcm_safety_check(df, pcm_latent_heat_j_per_kg, pcm_mass_kg):
    d = df.copy()
    max_surface_temp_c = float(d["SurfaceTempdegC"].max())
    max_surface_temp_k = float(d["TsurfaceK"].max())
    peak_heat_input_j = float(d["powerW"].sum())
    pcm_latent_capacity_j = float(pcm_latent_heat_j_per_kg * pcm_mass_kg)

    safe = pcm_latent_capacity_j >= peak_heat_input_j

    return {
        "max_surface_temp_c": max_surface_temp_c,
        "max_surface_temp_k": max_surface_temp_k,
        "peak_heat_input_j": peak_heat_input_j,
        "pcm_latent_capacity_j": pcm_latent_capacity_j,
        "safe": safe,
    }
```

## Where to place it

- Put the preprocessing code in the cell where you define data cleaning and feature engineering. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_35dbf7ed-c19d-4ecb-b3d5-a6b9157f856a/467820f3-0f09-416f-833b-eebcbabe88d1/vertopal.com_CHT.md)
- Put the validation comparison code in the NCA / experimental validation section, immediately after `loadncadata()` and before plotting. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_35dbf7ed-c19d-4ecb-b3d5-a6b9157f856a/467820f3-0f09-416f-833b-eebcbabe88d1/vertopal.com_CHT.md)
- Put the safety-check function near the summary / evaluation block, where you already save `hybridsummary.csv` and `ncavscfdcomparison.csv`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_35dbf7ed-c19d-4ecb-b3d5-a6b9157f856a/467820f3-0f09-416f-833b-eebcbabe88d1/vertopal.com_CHT.md)

## Suggested paper note file

Create a separate Markdown file like `formula_citations.md` with this content:

```md
# Formula Citations and Usage

## 1. Battery thermal energy balance

Use this for:
`m_b c_{p,b} dT_b/dt = Q_gen - Q_PCM`

Recommended citation:

- Li-Ion Battery Thermal Characterization for Thermal Management Applications.
  NREL report, 2024.
  https://docs.nrel.gov/docs/fy24osti/89032.pdf

## 2. Convection / Newton cooling

Use this for:
`Q = hA(T_b - T_pcm)`

Recommended citation:

- Validation fire tests and thermal validation literature on adiabatic surface temperature.
  https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=900083

## 3. PCM latent heat

Use this for:
`Q_latent = m L_f`

Recommended citation:

- Numerical Analysis of a Two-Layer PCM Based Battery Thermal Management System for Different Material Properties.
  https://doi.org/10.34248/bsengineering.1545174

## 4. PCM battery thermal management

Use this for:

- PCM reduces maximum battery temperature.
- PCM maintains thermal safety through melting/latent heat buffering.

Recommended citations:

- Numerical and analytical modelling of battery thermal management systems.
  https://core.ac.uk/download/pdf/42416076.pdf
- Research on PCM-based battery thermal management in Applied Thermal Engineering.
  https://www.open-access.bcu.ac.uk/14957/1/1-s2.0-S1359431123020148-main.pdf
- Experimental study on the thermal management performance of lithium-ion battery with PCM combined with 3-D finned tube.
  https://www.sciencedirect.com/science/article/abs/pii/S1359431124004629

## 5. Surface temperature validation

Use this for:

- maximum surface temperature,
- temperature rise from ambient,
- thermal safety evaluation.

Recommended citations:

- NREL battery thermal characterization report.
  https://docs.nrel.gov/docs/fy24osti/89032.pdf
- Thermal validation / surface temperature reference.
  https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=900083
```

## What I changed conceptually

- I aligned your imported dataset format with the notebook’s existing validation branch. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_35dbf7ed-c19d-4ecb-b3d5-a6b9157f856a/467820f3-0f09-416f-833b-eebcbabe88d1/vertopal.com_CHT.md)
- I added temperature in Kelvin, delta-T, voltage/current-based proxy features, and surface-temperature tracking. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_35dbf7ed-c19d-4ecb-b3d5-a6b9157f856a/467820f3-0f09-416f-833b-eebcbabe88d1/vertopal.com_CHT.md)
- I kept the architecture unchanged and used only validation-side additions, consistent with your space instructions. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_35dbf7ed-c19d-4ecb-b3d5-a6b9157f856a/467820f3-0f09-416f-833b-eebcbabe88d1/vertopal.com_CHT.md)
- I tied the formula usage to papers you can cite in Google Scholar or through DOI/link in your report. [docs.nrel](https://docs.nrel.gov/docs/fy24osti/89032.pdf)

## Best formula-to-paper mapping

| Formula                   | Use in your paper               | Citation source                                                                                                            |
| ------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| \(m c_p dT/dt\)           | Battery thermal energy balance  | NREL battery thermal characterization [docs.nrel](https://docs.nrel.gov/docs/fy24osti/89032.pdf)                           |
| \(Q = hA(T*b - T*{pcm})\) | Convective heat transfer to PCM | Thermal validation / heat transfer references [tsapps.nist](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=900083) |
| \(Q = mL_f\)              | PCM latent heat capacity        | PCM battery thermal management papers [dergipark.org](https://dergipark.org.tr/en/pub/bsengineering/article/1545174)       |
| Surface temperature max   | Safety validation target        | NREL + thermal validation references [docs.nrel](https://docs.nrel.gov/docs/fy24osti/89032.pdf)                            |

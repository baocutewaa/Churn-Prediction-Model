from __future__ import annotations

import numpy as np
import pandas as pd


def add_engineered_features(input_frame: pd.DataFrame) -> pd.DataFrame:
    frame = input_frame.copy()

    frame["BalanceSalaryRatio"] = frame["Balance"] / (frame["EstimatedSalary"] + 1.0)
    frame["BalancePerProduct"] = frame["Balance"] / np.maximum(frame["NumOfProducts"], 1)
    frame["TenureAgeRatio"] = frame["Tenure"] / (frame["Age"] + 1.0)
    frame["CreditScoreAgeInteraction"] = frame["CreditScore"] * frame["Age"]
    frame["ActivityBalanceInteraction"] = frame["IsActiveMember"] * frame["Balance"]
    frame["IsSenior"] = (frame["Age"] >= 60).astype(int)

    return frame

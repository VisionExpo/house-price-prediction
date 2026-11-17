"use strict";

const form = document.querySelector("#prediction-form");
const button = document.querySelector("#predict-button");
const value = document.querySelector("#prediction-value");
const caption = document.querySelector("#prediction-caption");
const errorMessage = document.querySelector("#error-message");

const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  errorMessage.hidden = true;
  button.disabled = true;
  button.textContent = "Preparing estimate…";
  value.textContent = "Working…";
  caption.textContent = "The model is applying the notebook's preprocessing pipeline.";

  const payload = Object.fromEntries(new FormData(form).entries());
  for (const key of ["GrLivArea", "LotArea", "OverallQual", "YearBuilt", "BedroomAbvGr", "FullBath", "GarageCars"]) {
    payload[key] = Number(payload[key]);
  }

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(result.detail)
        ? result.detail.map((issue) => issue.msg).join(" ")
        : result.detail;
      throw new Error(detail || "The prediction could not be generated.");
    }

    value.textContent = money.format(result.predicted_price);
    caption.textContent = `${result.model} estimate using ${result.input_features_used} entered features and training-split defaults for ${result.defaulted_features} other features.`;
  } catch (error) {
    value.textContent = "Estimate unavailable";
    caption.textContent = "Check the dataset setup and try again.";
    errorMessage.textContent = error.message;
    errorMessage.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Predict house price";
  }
});

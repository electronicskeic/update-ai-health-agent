from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatContext:
    summary: Any
    profile: dict | None
    last_bmi: float | None
    last_risk: Any | None
    last_plan: Any | None


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "N/A"


def answer(user_text: str, ctx: ChatContext) -> str:
    t = (user_text or "").strip().lower()

    if not t:
        return "Ask me anything about your plan, BMI, risk, or what to do next."

    if any(k in t for k in ("bmi", "body mass", "category")):
        if ctx.last_bmi is None:
            return "I don’t have your latest BMI yet—complete **Onboarding** and save your measurements, then I can explain your BMI and category."
        return f"Your latest BMI is **{ctx.last_bmi:.1f}**. In general, BMI helps categorize weight status, but it doesn’t measure body fat directly."

    if "risk" in t or "probability" in t:
        if ctx.last_risk is None:
            return "I don’t have your latest risk score yet—complete **Onboarding** and generate your plan first."
        p = getattr(ctx.last_risk, "obesity_probability", None)
        lvl = getattr(ctx.last_risk, "risk_level", None)
        return f"Your dataset-based obesity risk estimate is **{p:.2f}** ({lvl}). It’s based on patterns in this dataset and may not generalize."

    if any(k in t for k in ("next", "what should i do", "plan", "diet", "fitness", "workout")):
        if ctx.last_plan is None:
            return "Complete **Onboarding** first, then go to **My Plan**—I’ll generate a weekly focus and actionable diet/fitness steps."
        diet = getattr(ctx.last_plan, "diet", []) or []
        fit = getattr(ctx.last_plan, "fitness", []) or []
        parts: list[str] = []
        parts.append("Here are 2 quick next steps you can do this week:")
        if diet:
            parts.append(f"- Diet: {diet[0]}")
        if len(diet) > 1:
            parts.append(f"- Diet (alt): {diet[1]}")
        if fit:
            parts.append(f"- Fitness: {fit[0]}")
        return "\n".join(parts)

    if any(k in t for k in ("dataset", "insight", "correlation", "prevalence", "distribution")):
        s = ctx.summary
        try:
            return (
                "Dataset snapshot:\n"
                f"- n = **{s.n}**\n"
                f"- BMI mean = **{s.bmi_mean:.2f}**\n"
                f"- Overweight prevalence (BMI≥25) = **{_fmt_pct(s.overweight_rate)}**\n"
                f"- Obesity prevalence (BMI≥30) = **{_fmt_pct(s.obese_rate)}**\n"
                f"- Corr(Age,BMI) = **{s.age_bmi_corr:.3f}**\n"
            )
        except Exception:
            return "I can summarize the dataset from the **Insights** page after it loads successfully."

    if any(k in t for k in ("medical", "diagnose", "treatment", "prescribe", "symptom", "chest pain")):
        return "I can’t provide medical diagnosis or treatment. If you have symptoms or concerns, please consult a licensed clinician."

    return "I can help with BMI, risk, plans, and dataset insights. Try asking: “What should I do next?” or “Explain my BMI and risk.”"


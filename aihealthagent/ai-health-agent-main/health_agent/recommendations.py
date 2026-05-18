from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    diet: list[str]
    fitness: list[str]
    notes: list[str] | None = None


def build_plan(*, bmi: float, category: str, risk_level: str) -> Plan:
    b = float(bmi)
    risk = str(risk_level)
    cat = str(category)

    diet: list[str] = []
    fitness: list[str] = []
    notes: list[str] = []

    # Diet heuristics: practical, non-clinical, and safe defaults.
    diet.append("Aim for a consistent meal schedule and prioritize whole foods (protein + fiber at each meal).")
    diet.append("Swap sugary drinks for water/zero-calorie beverages most days.")

    if b >= 25:
        diet.append("Use a simple portion rule: half plate veggies, quarter protein, quarter carbs.")
        diet.append("Plan 1–2 high-protein snacks to reduce evening overeating (e.g., yogurt, eggs, legumes).")
    elif b < 18.5:
        diet.append("Add one calorie-dense, nutrient-dense snack daily (nuts, olive oil, peanut butter).")
        diet.append("Increase protein slightly and include carbs at each meal.")
    else:
        diet.append("Keep protein steady and include fruits/vegetables daily to maintain.")

    # Fitness heuristics: align with risk.
    fitness.append("Start with a daily 20–30 minute walk (or split into two 10–15 min walks).")
    fitness.append("Do 2 short strength sessions/week (push, pull, squat/hinge) using bodyweight or bands.")

    if risk.lower() == "high":
        notes.append("Keep changes small and consistent; reduce intensity if you feel pain or dizziness.")
        notes.append("If you have symptoms or chronic conditions, consider checking with a clinician before major changes.")
    elif risk.lower() == "moderate":
        notes.append("Progress by +5 minutes of walking or +1 set per exercise each week.")

    notes.append(f"Category: {cat}. These tips are educational and not medical advice.")
    return Plan(diet=diet, fitness=fitness, notes=notes or None)

